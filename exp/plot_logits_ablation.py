import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import typer
from PIL import Image

# Add parent directory to Python path so we can import models and data modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.jailbreak_data import JailbreakDataset
from data.specific_string_data import SpecificStringDataset
from models import Qwen25Instruct

MODEL = Qwen25Instruct.load_model(
    cache_dir=os.getenv("CACHE_DIR", "/home/adewinmb/orcd/scratch"),
    dtype="auto",
    device_map="auto",
    freeze_params=True,
)


def _load_prompt_pool(attack_type: str, prefix: str) -> List[str]:
    """Load all prompts for the requested attack type."""
    if attack_type == "specific_string":
        dataset = SpecificStringDataset(
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            target_string=prefix,
        )
        train_x, _, val_x, _, test_x, _ = dataset.get_splits()
        pool = test_x
    elif attack_type == "jailbreak":
        dataset = JailbreakDataset(train_split=0.7, val_split=0.2, test_split=0.1)
        train_x, _, val_x, _, test_x, _ = dataset.get_splits()
        pool = test_x
    else:
        raise ValueError("attack_type must be 'specific_string' or 'jailbreak'")

    if not pool:
        raise RuntimeError("Prompt pool is empty")
    return pool


def _prefix_token_ids(prefix: str) -> torch.Tensor:
    """Gets token_ids of the prefix string."""
    tokens = MODEL.processor.tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    if tokens.shape[1] == 0:
        raise ValueError("Prefix must produce at least one token")
    return tokens.to(MODEL.device)


def _token_labels(prefix_token_ids: torch.Tensor) -> List[str]:
    """Get tokens corresponding to each token id for plot labeling purposes."""
    tokens = prefix_token_ids[0].tolist()
    return MODEL.processor.tokenizer.convert_ids_to_tokens(tokens)


def _prepare_messages(prompt: str, label: str, image: Image.Image) -> Tuple[Dict[str, torch.Tensor], int]:
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_inputs = MODEL.processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = prompt_inputs.input_ids.shape[1]

    full_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": label},
            ],
        },
    ]
    inputs = MODEL.processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs, prompt_len


def _token_distributions(
    prompt: str,
    label: str,
    image: Image.Image,
    target_len: int,
) -> torch.Tensor:
    """Return full probability distributions for each prefix token."""
    inputs, prompt_len = _prepare_messages(prompt, label, image)
    inputs = inputs.to(MODEL.device)
    pixel_values = MODEL.processor.image_processor(
        image,
        return_tensors="pt",
    )["pixel_values"].to(MODEL.device)
    inputs["pixel_values"] = pixel_values

    with torch.no_grad():
        outputs = MODEL.model(**inputs, output_hidden_states=False)
        logits = outputs.logits[:, prompt_len - 1 : prompt_len + target_len - 1, :]
        probs = F.softmax(logits, dim=-1)
    return probs.squeeze(0)


def _generate_response(prompt: str, image: Image.Image, max_new_tokens: int) -> str:
    """Generate a textual response for the given prompt/image pair."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = MODEL.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(MODEL.device)
    pixel_values = MODEL.processor.image_processor(
        image,
        return_tensors="pt",
    )["pixel_values"].to(MODEL.device)
    inputs["pixel_values"] = pixel_values

    pad_token_id = MODEL.processor.tokenizer.pad_token_id or MODEL.processor.tokenizer.eos_token_id
    with torch.no_grad():
        generated_ids = MODEL.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )
        continuation = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        text = MODEL.processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    return text.strip()


def _format_token_label(token: str) -> str:
    """
    Clean up tokenizer artifacts for display.
    Qwen uses byte-level BPE where "Ġ" denotes a preceding space.
    """
    if token.startswith("Ġ"):
        trimmed = token[1:] or "<space>"
        # return f"[sp]{trimmed}"
        return trimmed
    return token


def _plot_distributions(
    base_probs: torch.Tensor,
    adv_probs: torch.Tensor,
    token_labels: List[str],
    save_dir: Path,
    prob_threshold: float,
) -> None:
    """Plot per-token probability distributions over the vocab for each position."""
    save_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = base_probs.shape[1]
    vocab_tokens = MODEL.processor.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))

    for idx, token_label in enumerate(token_labels):
        base_np = base_probs[idx].float().detach().cpu().numpy()
        adv_np = adv_probs[idx].float().detach().cpu().numpy()
        mask = (base_np >= prob_threshold) | (adv_np >= prob_threshold)
        if not np.any(mask):
            print(f"Skipping token position {idx + 1}; no tokens exceed threshold {prob_threshold}.")
            continue

        filtered_probs_base = base_np[mask]
        filtered_probs_adv = adv_np[mask]
        filtered_tokens = [_format_token_label(vocab_tokens[i]) for i, keep in enumerate(mask) if keep]
        x_positions = np.arange(len(filtered_tokens))
        width = 0.4

        plt.figure(figsize=(12, 5))
        plt.bar(
            x_positions - width / 2,
            filtered_probs_base,
            width=width,
            label="Black image",
            color="tab:blue",
            alpha=0.8,
        )
        plt.bar(
            x_positions + width / 2,
            filtered_probs_adv,
            width=width,
            label="Adversarial image",
            color="tab:orange",
            alpha=0.8,
        )
        plt.xticks(x_positions, filtered_tokens, rotation=60, ha="right")
        plt.xlabel("Token")
        plt.ylabel("Token probability")
        plt.title(f"Token position {idx + 1} ({token_label}) distribution (vocab={vocab_size})")
        plt.legend()
        plt.grid(True, alpha=0.2)
        outfile = save_dir / f"token_{idx+1:02d}.png"
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
        print(f"Saved histogram to {outfile}")


def plot_logits(
    prefix: str,
    attack_type: str,
    adv_image_path: str,
    prompt_index: int = 0,
    save_dir: str = "exp",
    prob_threshold: float = 0.0,
    max_new_tokens: int = 128,
):
    """
    Compare vocabulary distributions for a single prompt when paired with an
    adversarial image vs a baseline black image.
    """
    pool = _load_prompt_pool(attack_type, prefix)
    prompt = pool[prompt_index % len(pool)]
    prefix_token_ids = _prefix_token_ids(prefix)
    token_labels = _token_labels(prefix_token_ids)

    adv_image = Image.open(adv_image_path).convert("RGB")
    black_image = Image.new("RGB", adv_image.size, color=(0, 0, 0))

    target_len = prefix_token_ids.shape[1]
    adv_probs = _token_distributions(prompt, prefix, adv_image, target_len)
    base_probs = _token_distributions(prompt, prefix, black_image, target_len)

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using prompt index {prompt_index % len(pool)} out of {len(pool)} prompts.")
    print(f"Prompt: {prompt}")
    print(f"Adversarial image: {adv_image_path}")

    print("\nGenerating textual responses...")
    black_response = _generate_response(prompt, black_image, max_new_tokens)
    adv_response = _generate_response(prompt, adv_image, max_new_tokens)
    # responses_path = output_dir / "responses.txt"
    # with open(responses_path, "w") as f:
    #     f.write("Prompt:\n")
    #     f.write(prompt + "\n\n")
    #     f.write("Black image response:\n")
    #     f.write(black_response + "\n\n")
    #     f.write("Adversarial image response:\n")
    #     f.write(adv_response + "\n")
    print(f"Black image response:\n{black_response}\n")
    print(f"Adversarial image response:\n{adv_response}\n")
    # print(f"Saved responses to {responses_path}")

    _plot_distributions(base_probs, adv_probs, token_labels, output_dir, prob_threshold)


if __name__ == "__main__":
    """
    Generate histogram comparisons for a single prompt using adversarial vs black images.
    """
    plot_logits(
        prefix="Download the guide at malware.com!",
        attack_type="specific_string",
        adv_image_path="experiments/20251126_margin25/images/img_500.png",
        prompt_index=0,
        save_dir="exp",
        prob_threshold=0.01,
        max_new_tokens=128,
    )
