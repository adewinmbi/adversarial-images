import sys
import os
import json
from pathlib import Path
from typing import Optional
import re

# Add parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import typer
import pandas as pd
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)

# Use the refactored model registry helpers
from models import load_model_by_name, valid_model_names
from data.specific_string_data import SpecificStringDataset
from data.jailbreak_data import JailbreakDataset
from data.partition import apply_partition

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, output_path: str):
    """Save configuration to YAML file"""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _load_teacher_config(checkpoint_path: str, teacher_cfg_path: Optional[str] = None) -> dict:
    """
    Load the config.yaml for the teacher model.
    
    If teacher_cfg_path is provided, use that directly. Otherwise, infer from
    checkpoint path (expects checkpoint at <run_dir>/checkpoints/checkpoint_epoch_XXXX.pt
    and config at <run_dir>/config.yaml).
    """
    if teacher_cfg_path is not None:
        config_path = teacher_cfg_path
        if os.path.isdir(config_path):
            config_path = os.path.join(config_path, "config.yaml")
    else:
        ckpt_dir = os.path.dirname(checkpoint_path)
        run_dir = os.path.dirname(ckpt_dir)
        config_path = os.path.join(run_dir, "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _prepare_lora_model(base_model: torch.nn.Module, teacher_cfg: dict) -> torch.nn.Module:
    """Attach LoRA adapters to the teacher backbone using the teacher's saved config."""
    target_modules = teacher_cfg.get("lora_target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    if not target_modules:
        raise ValueError("lora_target_modules must be a non-empty list when use_lora is enabled.")

    lora_config = LoraConfig(
        r=teacher_cfg.get("lora_r", 64),
        lora_alpha=teacher_cfg.get("lora_alpha", 128),
        lora_dropout=teacher_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    print(
        f"Attaching LoRA adapters to teacher "
        f"(r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}, targets={target_modules})",
        flush=True,
    )
    peft_model = get_peft_model(base_model, lora_config)
    return peft_model


def _checkpoint_contains_lora(state_dict: dict) -> bool:
    """Detect whether a checkpoint state dict stores LoRA adapter weights."""
    return any("lora" in key for key in state_dict.keys())


def _load_teacher_model(
    model_name: str,
    checkpoint_path: str,
    cache_dir: Optional[str],
    teacher_cfg_path: Optional[str] = None,
):
    """
    Load a teacher model and hydrate weights from checkpoint.
    
    Automatically detects whether the checkpoint contains LoRA adapters and
    reads the LoRA hyperparameters from the teacher's config.yaml.
    
    Args:
        model_name: Model architecture key (e.g. 'qwen25_32b').
        checkpoint_path: Path to the .pt checkpoint file.
        cache_dir: HuggingFace cache directory.
        teacher_cfg_path: Optional explicit path to teacher config.yaml or its
            parent directory. If None, inferred from checkpoint_path.
    """
    if checkpoint_path is None:
        raise ValueError("teacher_checkpoint must be provided for distillation.")

    # Load the teacher's training config to get LoRA settings
    teacher_cfg = _load_teacher_config(checkpoint_path, teacher_cfg_path)
    use_lora = teacher_cfg.get("use_lora", False)

    load_kwargs = dict(
        cache_dir=cache_dir,
        dtype="auto",
        device_map="auto",
        freeze_params=True,
    )
    teacher_wrapper = load_model_by_name(model_name, **load_kwargs)

    checkpoint = torch.load(checkpoint_path, map_location=teacher_wrapper.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Auto-detect LoRA if config doesn't specify but checkpoint has adapter keys
    checkpoint_has_lora = _checkpoint_contains_lora(state_dict)
    if checkpoint_has_lora and not use_lora:
        print(
            "Detected LoRA adapter keys in checkpoint but use_lora=False in teacher config. "
            "Enabling LoRA with default parameters.",
            flush=True,
        )
        use_lora = True

    if use_lora:
        teacher_wrapper.model = _prepare_lora_model(teacher_wrapper.model, teacher_cfg)
        set_peft_model_state_dict(teacher_wrapper.model, state_dict)
    else:
        teacher_wrapper.model.load_state_dict(state_dict)

    teacher_wrapper.model.eval()
    return teacher_wrapper


@app.command()
def main(
    config: str = typer.Option(
        "configs/distill_default.yaml",
        help="Path to YAML config file"
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        help="Cache directory for models (overrides config)"
    ),
    num_epochs: Optional[int] = typer.Option(
        None,
        help="Number of training epochs (overrides config)"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        help="Learning rate (overrides config)"
    ),
    base_model_name: Optional[str] = typer.Option(
        None,
        help="Frozen student model to receive adversarial image"
    ),
    teacher_model_name: Optional[str] = typer.Option(
        None,
        help="Model architecture to instantiate for jailbroken checkpoint"
    ),
    teacher_checkpoint: Optional[str] = typer.Option(
        None,
        help="Path to finetuned/jailbroken model checkpoint (.pt)"
    ),
    teacher_cfg_path: Optional[str] = typer.Option(
        None,
        help="Path to teacher config.yaml or its parent directory. If not provided, inferred from teacher_checkpoint."
    ),
    initial_image: Optional[str] = typer.Option(
        None,
        help="Initial image type: black, noise, or expedia (overrides config)"
    ),
    target_str: Optional[str] = typer.Option(
        None,
        help="Target string for adversarial attack (overrides config, only used for specific_string dataset)"
    ),
    dataset_type: Optional[str] = typer.Option(
        None,
        help="Dataset type: specific_string or jailbreak (overrides config)"
    ),
    partition: Optional[str] = typer.Option(
        None,
        case_sensitive=False,
        help="Dataset partition to use: 'all', 'a', or 'b' (applies to train/val/test)",
    ),
    log_dir: Optional[str] = typer.Option(
        None,
        help="Log directory (overrides config)"
    ),
    steps_per_epoch: Optional[int] = typer.Option(
        None
    ),
    checkpoint_dir: Optional[str] = typer.Option(
        None,
        help="""
        If null, starts training from `initial_image`. 
        Otherwise, resumes training from most recent 
        image in checkpoint_dir."""
    ),
    checkpoint_every: Optional[int] = typer.Option(
        None,
        help="Save adversarial image checkpoints every N epochs (default 10)."
    )
):
    """
    Distill a jailbroken (teacher) model into an adversarial image for a frozen
    student model. Teacher provides soft token distributions; the student with
    the adversarial image matches those distributions via KL/Cross-Entropy.
    """
    
    # Load base config
    cfg = load_config(config)
    
    # Override with CLI arguments if provided
    if cache_dir is not None:
        cfg['cache_dir'] = cache_dir
    if num_epochs is not None:
        cfg['num_epochs'] = num_epochs
    if learning_rate is not None:
        cfg['learning_rate'] = learning_rate
    if base_model_name is not None:
        cfg['base_model_name'] = base_model_name
    if teacher_model_name is not None:
        cfg['teacher_model_name'] = teacher_model_name
    if teacher_checkpoint is not None:
        cfg['teacher_checkpoint'] = teacher_checkpoint
    if teacher_cfg_path is not None:
        cfg['teacher_cfg_path'] = teacher_cfg_path
    if initial_image is not None:
        cfg['initial_image'] = initial_image
    if target_str is not None:
        cfg['target_str'] = target_str
    if dataset_type is not None:
        cfg['dataset_type'] = dataset_type
    if partition is not None:
        cfg['partition'] = partition
    if log_dir is not None:
        cfg['log_dir'] = log_dir
    if steps_per_epoch is not None:
        cfg['steps_per_epoch'] = steps_per_epoch
    if checkpoint_dir is not None:
        cfg['checkpoint_dir'] = checkpoint_dir
    if checkpoint_every is not None:
        cfg['checkpoint_every'] = checkpoint_every

    cfg.setdefault('checkpoint_every', 10)
    cfg.setdefault('partition', 'all')
    if cfg['checkpoint_every'] <= 0:
        raise ValueError("checkpoint_every must be a positive integer.")
    
    # Generate log directory if not specified
    if cfg['log_dir'] is None:
        cfg['log_dir'] = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir_path = cfg['log_dir']
    checkpoint_dir_value = cfg.get('checkpoint_dir')
    
    # Create log directories
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(os.path.join(log_dir_path, "images"), exist_ok=True)
    
    # Save the actual config used (including CLI overrides)
    save_config(cfg, os.path.join(log_dir_path, "config.yaml"))
    
    # ======= LOAD MODEL =======
    
    # Validate model_name
    valid_models = valid_model_names()
    if cfg['base_model_name'] not in valid_models:
        print(f"Error: Invalid base_model_name '{cfg['base_model_name']}'. Must be one of {valid_models}", flush=True)
        sys.exit(1)
    teacher_model_name = cfg.get('teacher_model_name', cfg['base_model_name'])
    if teacher_model_name not in valid_models:
        print(f"Error: Invalid teacher_model_name '{teacher_model_name}'. Must be one of {valid_models}", flush=True)
        sys.exit(1)
    if cfg.get('teacher_checkpoint') is None:
        raise ValueError("teacher_checkpoint must be provided for adversarial distillation.")
    
    load_kwargs = dict(
        cache_dir=cfg['cache_dir'],
        dtype="auto",
        device_map="auto",
        freeze_params=True,
    )
    model_wrapper = load_model_by_name(cfg['base_model_name'], **load_kwargs)
    teacher_wrapper = _load_teacher_model(
        teacher_model_name,
        cfg['teacher_checkpoint'],
        cfg['cache_dir'],
        cfg.get('teacher_cfg_path'),
    )
    
    # ======= LOAD AND SPLIT DATASET =======
    
    # Validate dataset_type
    valid_dataset_types = ["specific_string", "jailbreak"]
    if cfg['dataset_type'] not in valid_dataset_types:
        print(f"Error: Invalid dataset_type '{cfg['dataset_type']}'. Must be one of {valid_dataset_types}", flush=True)
    
    # Load the appropriate dataset
    if cfg['dataset_type'] == "specific_string":
        dataset = SpecificStringDataset(
            train_split=cfg['train_split'],
            val_split=cfg['val_split'],
            test_split=cfg['test_split'],
            target_string=cfg['target_str']
        )
    elif cfg['dataset_type'] == "jailbreak":
        dataset = JailbreakDataset(
            train_split=cfg['train_split'],
            val_split=cfg['val_split'],
            test_split=cfg['test_split']
        )
    
    # Get dataset splits
    train_instructions, train_labels, val_instructions, val_labels, test_instructions, test_labels = dataset.get_splits()
    partition_choice = cfg.get('partition', 'all')
    train_instructions, train_labels = apply_partition(
        train_instructions, train_labels, partition_choice, split_name="train"
    )
    val_instructions, val_labels = apply_partition(
        val_instructions, val_labels, partition_choice, split_name="val"
    )
    test_instructions, test_labels = apply_partition(
        test_instructions, test_labels, partition_choice, split_name="test"
    )
    
    # ======= INITIALIZE ADVERSARIAL IMAGE =======
    print(f"Initializing adversarial image with method: {cfg['initial_image']}", flush=True)
    
    start_epoch = 0
    if cfg['checkpoint_dir'] == None:
        if cfg['initial_image'] == "black":
            adv_image_np = np.zeros((224, 224, 3), dtype=np.uint8)
            original_image_np = adv_image_np.copy()
        elif cfg['initial_image'] == "noise":
            adv_image_np = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            original_image_np = adv_image_np.copy()
        elif cfg['initial_image'] == "expedia":
            original_pil = Image.open("data/expedia.png").convert("RGB")
            original_pil = original_pil.resize((224, 224))
            original_image_np = np.array(original_pil)
            adv_image_np = original_image_np.copy()
        else:
            raise ValueError(f"Unknown initial image type: {cfg['initial_image']}")
    else:
        candidates = []
        for filename in os.listdir(cfg['checkpoint_dir'] + '/images'):
            m = re.fullmatch(r"img_(\d+).png", filename)
            if m: 
                start_epoch = int(m.group(1))
                candidates.append( (start_epoch, filename) )
                
        if candidates:
            target_file = max(candidates)[1]
        else:
            print('Could not find an image in checkpoint_dir!')
            
        original_pil = Image.open(cfg['checkpoint_dir'] + f"/images/{target_file}").convert("RGB")
        print(f"Resuming from image: {cfg['checkpoint_dir'] + f"/images/{target_file}"}")
        
        original_pil = original_pil.resize((224, 224))
        original_image_np = np.array(original_pil)
        adv_image_np = original_image_np.copy()
    
    adv_image_tensor = torch.from_numpy(adv_image_np.astype(np.float32) / 255.0).to(model_wrapper.device)
    original_image_tensor = torch.from_numpy(original_image_np.astype(np.float32) / 255.0).to(model_wrapper.device)
    adv_image_tensor.requires_grad = True
    
    optimizer = torch.optim.Adam([adv_image_tensor], lr=cfg['learning_rate'])
    
    # ======= HELPER FUNCTIONS =======
    def tensor_to_pil(tensor):
        """Convert tensor [H, W, C] in range [0, 1] to PIL Image"""
        img_np = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def _tokenize_label(processor, text):
        tokens = processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )
        return tokens.input_ids

    def compute_teacher_probs(instruction: str, label: str):
        """Return teacher probability distribution over target tokens (text-only)."""
        target_tokens = _tokenize_label(teacher_wrapper.processor, label).to(teacher_wrapper.device)
        target_len = target_tokens.shape[1]
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        prompt_inputs = teacher_wrapper.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        prompt_len = prompt_inputs.input_ids.shape[1]
        full_messages = [
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
            }
        ]
        inputs = teacher_wrapper.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(teacher_wrapper.device)
        with torch.no_grad():
            outputs = teacher_wrapper.model(**inputs, output_hidden_states=False)
            logits = outputs.logits[:, prompt_len-1:prompt_len+target_len-1, :]
            probs = F.softmax(logits, dim=-1)
        return probs.squeeze(0).to(model_wrapper.device), target_tokens.to(model_wrapper.device)

    def compute_student_logits(instruction: str, label: str, adv_image_tensor: torch.Tensor):
        """Return student logits aligned to the label tokens."""
        target_tokens = _tokenize_label(model_wrapper.processor, label).to(model_wrapper.device)
        target_len = target_tokens.shape[1]
        img_for_processor = adv_image_tensor.permute(2, 0, 1)
        temp_pil = tensor_to_pil(adv_image_tensor)
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_pil},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        prompt_inputs = model_wrapper.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        prompt_len = prompt_inputs.input_ids.shape[1]
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_pil},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": label},
                ],
            }
        ]
        inputs = model_wrapper.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model_wrapper.device)
        if 'pixel_values' in inputs:
            temp_inputs = model_wrapper.processor.image_processor(img_for_processor, return_tensors="pt")
            inputs['pixel_values'] = temp_inputs['pixel_values'].to(model_wrapper.device)
        outputs = model_wrapper.model(**inputs, output_hidden_states=False)
        logits = outputs.logits[:, prompt_len-1:prompt_len+target_len-1, :]
        return logits.squeeze(0), target_tokens

    def compute_loss(instruction, label, adv_image_tensor):
        """Cross-entropy between teacher soft labels and student logits."""
        teacher_probs, _ = compute_teacher_probs(instruction, label)
        student_logits, _ = compute_student_logits(instruction, label, adv_image_tensor)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss
    
    def evaluate(instructions, labels, adv_image_tensor):
        """
        Evaluate on a set of instructions.
        
        Args:
            instructions: List of instruction/prompt texts
            labels: List of target label/response texts
            adv_image_tensor: The adversarial image tensor
        """
        total_loss = 0.0
        with torch.no_grad():
            for instruction, label in zip(instructions[:cfg['steps_per_epoch']], labels[:cfg['steps_per_epoch']]):
                loss = compute_loss(instruction, label, adv_image_tensor)
                total_loss += loss.item()
        return total_loss / min(len(instructions), cfg['steps_per_epoch'])
    
    # ======= TRAINING LOOP =======
    print(f"Starting training. Saving at {log_dir_path}", flush=True)
    
    checkpoint_path = os.path.join(checkpoint_dir_value, 'loss.csv') if checkpoint_dir_value is not None else None
    if checkpoint_dir_value is not None and checkpoint_path is not None:
        prev_losses = pd.read_csv(checkpoint_path)
        train_losses = prev_losses['train_loss'].tolist()
        val_losses = prev_losses['val_loss'].tolist()
    else:
        train_losses = []
        val_losses = []
        
    for epoch in range(start_epoch, cfg['num_epochs'] + start_epoch):
        print(f"\nEpoch {epoch + 1}/{start_epoch + cfg['num_epochs']}", flush=True)
        epoch_train_loss = 0.0
        
        # Shuffle train data together
        train_indices_shuffle = np.random.permutation(len(train_instructions))
        train_instructions_shuffled = [train_instructions[i] for i in train_indices_shuffle]
        train_labels_shuffled = [train_labels[i] for i in train_indices_shuffle]
        
        # Training
        for i, (instruction, label) in enumerate(zip(train_instructions_shuffled[:cfg['steps_per_epoch']], 
                                                       train_labels_shuffled[:cfg['steps_per_epoch']])):
            optimizer.zero_grad()
            
            loss = compute_loss(instruction, label, adv_image_tensor)
            loss.backward()
            
            optimizer.step()
            
            # Project back to epsilon ball
            with torch.no_grad():
                delta = adv_image_tensor - original_image_tensor
                delta = torch.clamp(delta, -cfg['epsilon'], cfg['epsilon'])
                adv_image_tensor.data = original_image_tensor + delta
                adv_image_tensor.data = torch.clamp(adv_image_tensor.data, 0, 1)
            
            epoch_train_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Step {i + 1}/{min(len(train_instructions), cfg['steps_per_epoch'])}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / min(len(train_instructions), cfg['steps_per_epoch'])
        train_losses.append(avg_train_loss)
        
        # Validation
        print("  Validating...")
        avg_val_loss = evaluate(val_instructions, val_labels, adv_image_tensor)
        val_losses.append(avg_val_loss)
        
        print(f"  Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}", flush=True)
        
        # Save adversarial image for this epoch according to checkpoint cadence
        if (epoch + 1) % cfg['checkpoint_every'] == 0:
            epoch_adv_pil = tensor_to_pil(adv_image_tensor)
            epoch_adv_pil.save(os.path.join(cfg['log_dir'], "images", f"img_{epoch + 1}.png"))
        
        def extend_losses(x):
            """Adjust length of train/val list to accomodate for missing loss.csv from checkpoint"""
            return [x[0]] * (epoch - len(x) + 1) + x
        
        train_losses = extend_losses(train_losses)
        val_losses = extend_losses(val_losses)
        
        # Update and save loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(start_epoch + 1, epoch + start_epoch + 2), train_losses, label='Train Loss', marker='o')
        plt.plot(range(start_epoch + 1, epoch + start_epoch + 2), val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(cfg['log_dir'], "loss_plot.png"))
        plt.close()
        
        # Update and save loss csv
        loss_df = pd.DataFrame({
            'epoch': range(1, epoch+2),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        loss_df.to_csv(os.path.join(cfg['log_dir'], 'loss.csv'), index=False)
    
    # ======= SAVE FINAL ADVERSARIAL IMAGE =======
    final_adv_pil = tensor_to_pil(adv_image_tensor)
    final_adv_pil.save(os.path.join(cfg['log_dir'], "adversarial_image.png"))
    print(f"Results saved in {cfg['log_dir']}")
    
    # ======= GENERATE TEST EXAMPLES =======
    print(f"\nGenerating {cfg['num_examples']} test examples...", flush=True)
    final_adv_pil = tensor_to_pil(adv_image_tensor)
    
    test_results = []
    for i in range(min(cfg['num_examples'], len(test_instructions))):
        instruction = test_instructions[i]
        
        # Using the clean generate interface!
        output_text = model_wrapper.generate(instruction, final_adv_pil, max_new_tokens=128)
        
        test_results.append({
            "instruction": instruction,
            "output": output_text
        })
        
        print(f"  Example {i + 1}/{cfg['num_examples']}")
        print(f"    Instruction: {instruction}")
        print(f"    Output: {output_text[:100]}...")
    
    # Save test results
    with open(os.path.join(cfg['log_dir'], "test_examples.txt"), 'w') as f:
        for i, result in enumerate(test_results):
            f.write(f"Example {i + 1}\n")
            f.write(f"Instruction: {result['instruction']}\n")
            f.write(f"Output: {result['output']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Saved test examples to {os.path.join(cfg['log_dir'], 'test_examples.txt')}", flush=True)
    print("\nTraining complete!", flush=True)


if __name__ == "__main__":
    app()
