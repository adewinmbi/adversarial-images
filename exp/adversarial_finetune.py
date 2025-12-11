import os
import sys
import re
from typing import List, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import typer
import yaml

# Add parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model_by_name, valid_model_names
from data.specific_string_data import SpecificStringDataset
from data.jailbreak_data import JailbreakDataset
from data.partition import apply_partition

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_CONFIG = "configs/finetune_default.yaml"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, output_path: str):
    """Persist configuration (including CLI overrides)."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def limit_steps(num_examples: int, steps_per_epoch: Optional[int]) -> int:
    """Determine how many samples to use this epoch."""
    if steps_per_epoch is None:
        return num_examples
    return min(num_examples, steps_per_epoch)


def prepare_lora_model(base_model: torch.nn.Module, cfg: dict) -> torch.nn.Module:
    """Wrap the base model with LoRA adapters based on the provided config."""
    
    target_modules = cfg.get("lora_target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("lora_target_modules must be a non-empty list of module names.")

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    print(
        "Enabling LoRA adapters "
        f"(r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, dropout={cfg['lora_dropout']}, "
        f"targets={target_modules})",
        flush=True,
    )
    peft_model = get_peft_model(base_model, lora_config)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()
    return peft_model


@app.command()
def main(
    config: str = typer.Option(
        DEFAULT_CONFIG,
        help="Path to YAML config file"
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        help="Cache directory for models (overrides config)"
    ),
    num_epochs: Optional[int] = typer.Option(
        None,
        help="Number of additional training epochs"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        help="Learning rate"
    ),
    model_name: Optional[str] = typer.Option(
        None,
        help="Model to fine-tune: llava, qwen3inst, qwen3think, qwen25inst, qwen25_32b"
    ),
    loss_type: Optional[str] = typer.Option(
        None,
        help="Loss to use: 'ce' or 'margin'"
    ),
    margin_param: Optional[int] = typer.Option(
        None,
        help="Margin hyperparameter for margin loss"
    ),
    target_str: Optional[str] = typer.Option(
        None,
        help="Target string (specific_string dataset only)"
    ),
    dataset_type: Optional[str] = typer.Option(
        None,
        help="Dataset type: specific_string or jailbreak"
    ),
    partition: Optional[str] = typer.Option(
        None,
        case_sensitive=False,
        help="Dataset partition to use: 'all', 'a', or 'b' (applies to train/val/test)",
    ),
    log_dir: Optional[str] = typer.Option(
        None,
        help="Directory to store logs/checkpoints"
    ),
    steps_per_epoch: Optional[int] = typer.Option(
        None,
        help="How many samples to use per epoch (None = full dataset)"
    ),
    checkpoint_dir: Optional[str] = typer.Option(
        None,
        help="Directory with previous run to resume from"
    ),
    checkpoint_every: Optional[int] = typer.Option(
        None,
        help="Save model checkpoint every N epochs"
    ),
    use_lora: Optional[bool] = typer.Option(
        None,
        help="Enable LoRA adapters for parameter-efficient fine-tuning (requires 'peft').",
    ),
    lora_r: Optional[int] = typer.Option(
        None,
        help="LoRA rank (only used when use_lora is enabled).",
    ),
    lora_alpha: Optional[int] = typer.Option(
        None,
        help="LoRA alpha scaling factor (only used when use_lora is enabled).",
    ),
    lora_dropout: Optional[float] = typer.Option(
        None,
        help="LoRA dropout probability (only used when use_lora is enabled).",
    ),
    lora_target_modules: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of module names to target with LoRA adapters.",
    ),
):
    """
    Fine-tune a VLM directly on the jailbreak/specific string datasets.

    This script performs standard parameter updates (no adversarial image
    optimization) using the same instruction/label pairs as the image attack.
    """

    # Load base configuration
    cfg = load_config(config)

    # Apply CLI overrides
    if cache_dir is not None:
        cfg["cache_dir"] = cache_dir
    if num_epochs is not None:
        cfg["num_epochs"] = num_epochs
    if learning_rate is not None:
        cfg["learning_rate"] = learning_rate
    if model_name is not None:
        cfg["model_name"] = model_name
    if target_str is not None:
        cfg["target_str"] = target_str
    if loss_type is not None:
        cfg["loss_type"] = loss_type
    if margin_param is not None:
        cfg["margin_param"] = margin_param
    if dataset_type is not None:
        cfg["dataset_type"] = dataset_type
    if partition is not None:
        cfg["partition"] = partition
    if log_dir is not None:
        cfg["log_dir"] = log_dir
    if steps_per_epoch is not None:
        cfg["steps_per_epoch"] = steps_per_epoch
    if checkpoint_dir is not None:
        cfg["checkpoint_dir"] = checkpoint_dir
    if checkpoint_every is not None:
        cfg["checkpoint_every"] = checkpoint_every
    if use_lora is not None:
        cfg["use_lora"] = use_lora
    if lora_r is not None:
        cfg["lora_r"] = lora_r
    if lora_alpha is not None:
        cfg["lora_alpha"] = lora_alpha
    if lora_dropout is not None:
        cfg["lora_dropout"] = lora_dropout
    if lora_target_modules is not None:
        cfg["lora_target_modules"] = [
            module.strip()
            for module in lora_target_modules.split(",")
            if module.strip()
        ]

    cfg.setdefault("checkpoint_every", 1)
    cfg.setdefault("max_new_tokens", 128)
    cfg.setdefault("partition", "all")
    cfg.setdefault("use_lora", False)
    cfg.setdefault("lora_r", 64)
    cfg.setdefault("lora_alpha", 128)
    cfg.setdefault("lora_dropout", 0.05)
    cfg.setdefault("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

    if cfg["checkpoint_every"] <= 0:
        raise ValueError("checkpoint_every must be a positive integer.")

    # If we are resuming and no explicit log_dir was provided, continue in-place.
    if cfg.get("log_dir") is None:
        if cfg.get("checkpoint_dir"):
            cfg["log_dir"] = cfg["checkpoint_dir"]
        else:
            cfg["log_dir"] = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(cfg["log_dir"], exist_ok=True)
    
    save_checkpoint_dir = cfg["checkpoint_dir"] if cfg["checkpoint_dir"] is not None else cfg["log_dir"]
    checkpoints_path = os.path.join(save_checkpoint_dir, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    # Persist final config for reproducibility
    save_config(cfg, os.path.join(cfg["log_dir"], "config.yaml"))

    # ======= LOAD MODEL =======
    valid_models = valid_model_names()
    if cfg["model_name"] not in valid_models:
        print(
            f"Error: Invalid model_name '{cfg['model_name']}'. Must be one of {valid_models}",
            flush=True,
        )
        sys.exit(1)

    load_kwargs = dict(cache_dir=cfg["cache_dir"], dtype="auto", device_map="auto", freeze_params=False)
    model_wrapper = load_model_by_name(cfg["model_name"], **load_kwargs)

    if cfg["use_lora"]:
        model_wrapper.model = prepare_lora_model(model_wrapper.model, cfg)

    model_wrapper.model.train()

    # ======= LOAD DATASET =======
    valid_dataset_types = ["specific_string", "jailbreak"]
    if cfg["dataset_type"] not in valid_dataset_types:
        print(
            f"Error: Invalid dataset_type '{cfg['dataset_type']}'. Must be one of {valid_dataset_types}",
            flush=True,
        )
        sys.exit(1)

    if cfg["dataset_type"] == "specific_string":
        dataset = SpecificStringDataset(
            train_split=cfg["train_split"],
            val_split=cfg["val_split"],
            test_split=cfg["test_split"],
            target_string=cfg["target_str"],
        )
    else:
        dataset = JailbreakDataset(
            train_split=cfg["train_split"],
            val_split=cfg["val_split"],
            test_split=cfg["test_split"],
        )

    (
        train_instructions,
        train_labels,
        val_instructions,
        val_labels,
        test_instructions,
        test_labels,
    ) = dataset.get_splits()

    partition_choice = cfg.get("partition", "all")
    train_instructions, train_labels = apply_partition(
        train_instructions, train_labels, partition_choice, split_name="train"
    )
    val_instructions, val_labels = apply_partition(
        val_instructions, val_labels, partition_choice, split_name="val"
    )
    test_instructions, test_labels = apply_partition(
        test_instructions, test_labels, partition_choice, split_name="test"
    )

    effective_train_steps = limit_steps(len(train_instructions), cfg["steps_per_epoch"])
    effective_val_steps = limit_steps(len(val_instructions), cfg["steps_per_epoch"])

    if effective_train_steps == 0 or effective_val_steps == 0:
        raise RuntimeError("Dataset splits are empty; cannot train or validate.")

    # ======= OPTIMIZER / STATE =======
    optimizer = torch.optim.AdamW(model_wrapper.model.parameters(), lr=cfg["learning_rate"])
    start_epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    def load_latest_checkpoint(resume_dir: str):
        ckpt_dir = os.path.join(resume_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return None
        candidates = []
        for filename in os.listdir(ckpt_dir):
            match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", filename)
            if match:
                candidates.append((int(match.group(1)), filename))
        if not candidates:
            return None
        epoch_idx, fname = max(candidates, key=lambda item: item[0])
        ckpt_path = os.path.join(ckpt_dir, fname)
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        checkpoint = torch.load(ckpt_path, map_location=model_wrapper.device)
        model_state = checkpoint["model_state_dict"]
        if cfg["use_lora"]:
            set_peft_model_state_dict(model_wrapper.model, model_state)
        else:
            model_wrapper.model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(model_wrapper.device)
        return checkpoint

    resume_dir = cfg.get("checkpoint_dir")
    if resume_dir is not None:
        checkpoint = load_latest_checkpoint(resume_dir)
        loss_csv_path = os.path.join(resume_dir, "loss.csv")
        if checkpoint is not None:
            start_epoch = int(checkpoint.get("epoch", 0))
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
        elif os.path.exists(loss_csv_path):
            prev_losses = pd.read_csv(loss_csv_path)
            if not prev_losses.empty:
                train_losses = prev_losses["train_loss"].tolist()
                val_losses = prev_losses["val_loss"].tolist()
                start_epoch = int(prev_losses["epoch"].iloc[-1])

    # ======= HELPER FUNCTIONS =======
    def compute_loss(instruction: str, label: str) -> torch.Tensor:
        """Teacher-forced loss on the target label tokens."""
        processor = model_wrapper.processor

        target_tokens = processor.tokenizer(
            label,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(model_wrapper.device)
        target_len = target_tokens.shape[1]

        prompt_inputs = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label},
                    ],
                },
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model_wrapper.device)

        outputs = model_wrapper.model(**inputs, output_hidden_states=False)
        logits = outputs.logits
        logits_for_target = logits[:, prompt_len - 1 : prompt_len + target_len - 1, :]
        logits_flat = logits_for_target.reshape(-1, logits_for_target.shape[-1])
        target_flat = target_tokens.reshape(-1)

        if cfg["loss_type"] == "margin":
            z_y = logits_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)
            logits_copy = logits_flat.clone()
            logits_copy[torch.arange(logits_flat.shape[0]), target_flat] = float("-inf")
            z_runnerup, _ = logits_copy.max(dim=1)
            margin = torch.clamp(cfg["margin_param"] - (z_y - z_runnerup), min=0)
            return margin.mean()

        return F.cross_entropy(logits_flat, target_flat)

    def evaluate(split_instructions, split_labels, limit_override: Optional[int] = None) -> float:
        """Evaluate loss on a split."""
        model_wrapper.model.eval()
        total_loss = 0.0
        limit = limit_override or limit_steps(len(split_instructions), cfg["steps_per_epoch"])
        with torch.no_grad():
            for instruction, label in zip(split_instructions[:limit], split_labels[:limit]):
                total_loss += compute_loss(instruction, label).item()
        model_wrapper.model.train()
        return total_loss / limit

    def save_checkpoint(epoch_idx: int):
        """Persist model/optimizer state for resuming later."""
        if cfg["use_lora"]:
            model_state = get_peft_model_state_dict(model_wrapper.model)
        else:
            model_state = model_wrapper.model.state_dict()

        ckpt = {
            "epoch": epoch_idx,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        ckpt_path = os.path.join(checkpoints_path, f"checkpoint_epoch_{epoch_idx:04d}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}", flush=True)

    def generate_response(instruction: str) -> str:
        """Run model.generate on a text-only prompt."""
        processor = model_wrapper.processor
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model_wrapper.device)
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

        with torch.no_grad():
            generated_ids = model_wrapper.model.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                pad_token_id=pad_token_id,
            )
            continuation = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        return output_text

    print(f"Starting fine-tuning. Saving artifacts to {cfg['log_dir']}", flush=True)

    total_epochs = cfg["num_epochs"]
    for epoch in range(start_epoch, start_epoch + total_epochs):
        epoch_idx = epoch + 1
        print(f"\nEpoch {epoch_idx}/{start_epoch + total_epochs}", flush=True)
        epoch_train_loss = 0.0

        shuffled_indices = np.random.permutation(len(train_instructions))[:effective_train_steps]
        for step_idx, data_idx in enumerate(shuffled_indices, start=1):
            optimizer.zero_grad()
            instruction = train_instructions[data_idx]
            label = train_labels[data_idx]
            loss = compute_loss(instruction, label)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            if step_idx % 10 == 0 or step_idx == effective_train_steps:
                print(
                    f"  Step {step_idx}/{effective_train_steps} - Loss: {loss.item():.4f}",
                    flush=True,
                )

        avg_train_loss = epoch_train_loss / effective_train_steps
        train_losses.append(avg_train_loss)

        print("  Validating...", flush=True)
        avg_val_loss = evaluate(val_instructions, val_labels, effective_val_steps)
        val_losses.append(avg_val_loss)

        print(
            f"  Epoch {epoch_idx} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}",
            flush=True,
        )

        epochs_axis = list(range(1, len(train_losses) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_axis, train_losses, label="Train Loss", marker="o")
        plt.plot(epochs_axis, val_losses, label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(cfg["log_dir"], "loss_plot.png"))
        plt.close()

        loss_df = pd.DataFrame(
            {
                "epoch": epochs_axis,
                "train_loss": train_losses,
                "val_loss": val_losses,
            }
        )
        loss_df.to_csv(os.path.join(cfg["log_dir"], "loss.csv"), index=False)

        if epoch_idx % cfg["checkpoint_every"] == 0:
            save_checkpoint(epoch_idx)

    # Always save a final checkpoint
    final_epoch_idx = start_epoch + total_epochs
    save_checkpoint(final_epoch_idx)
    print(f"Training complete. Final checkpoint saved at epoch {final_epoch_idx}.", flush=True)

    # ======= GENERATE TEST EXAMPLES =======
    print(f"\nGenerating {cfg['num_examples']} held-out responses...", flush=True)
    test_results = []
    for i, instruction in enumerate(test_instructions[: cfg["num_examples"]], start=1):
        output_text = generate_response(instruction)
        test_results.append({"instruction": instruction, "output": output_text})
        print(f"  Example {i}/{cfg['num_examples']}")
        print(f"    Instruction: {instruction}")
        print(f"    Output: {output_text[:100]}...")

    with open(os.path.join(cfg["log_dir"], "test_examples.txt"), "w") as f:
        for i, result in enumerate(test_results, start=1):
            f.write(f"Example {i}\n")
            f.write(f"Instruction: {result['instruction']}\n")
            f.write(f"Output: {result['output']}\n")
            f.write("-" * 80 + "\n\n")

    print(f"Saved test examples to {os.path.join(cfg['log_dir'], 'test_examples.txt')}", flush=True)


if __name__ == "__main__":
    app()
