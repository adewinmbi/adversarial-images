import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def _enumerate_image_paths(
    start_path: str,
    start_img: Optional[int],
    end_img: Optional[int],
    img_step: int,
) -> List[Path]:
    path = Path(start_path).expanduser().resolve()

    if path.is_dir():
        directory = path
        prefix = "img_"
        suffix = ".png"
        if start_img is None:
            raise ValueError("start_img must be provided when start_path is a directory")
        idx = start_img
    else:
        if not path.exists():
            raise FileNotFoundError(f"Start image '{path}' does not exist")
        match = re.fullmatch(r"(img_)(\d+)(\.[^.]+)", path.name)
        if not match:
            raise ValueError("Start image name must follow the pattern img_<index>.png")
        prefix, idx_str, suffix = match.groups()
        idx = start_img if start_img is not None else int(idx_str)
        directory = path.parent

    if end_img is not None and end_img < idx:
        raise ValueError("end_img must be greater than or equal to start_img")

    if img_step <= 0:
        raise ValueError("img_step must be a positive integer")

    image_paths: List[Path] = []
    while True:
        if end_img is not None and idx > end_img:
            break
        candidate = directory / f"{prefix}{idx}{suffix}"
        if not candidate.exists():
            if not image_paths:
                raise FileNotFoundError(f"Image '{candidate}' does not exist")
            break
        image_paths.append(candidate)
        idx += img_step

    if not image_paths:
        raise RuntimeError("No images found for the given range")
    return image_paths


def _sample_prompts(attack_type: str, num_prompts: int, prefix: str) -> List[str]:
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    if attack_type == "specific_string":
        dataset = SpecificStringDataset(
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            target_string=prefix,
        )
        train_x, _, val_x, _, test_x, _ = dataset.get_splits()
        pool = train_x + val_x + test_x
    else:
        dataset = JailbreakDataset(train_split=0.7, val_split=0.2, test_split=0.1)
        train_x, _, val_x, _, test_x, _ = dataset.get_splits()
        pool = train_x + val_x + test_x
    if not pool:
        raise RuntimeError("Prompt pool is empty")
    if len(pool) <= num_prompts:
        return pool
    rng = np.random.default_rng(0)
    indices = rng.choice(len(pool), size=num_prompts, replace=False)
    return [pool[i] for i in indices]


def _prefix_token_ids(prefix: str) -> torch.Tensor:
    tokens = MODEL.processor.tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    if tokens.shape[1] == 0:
        raise ValueError("Prefix must produce at least one token")
    return tokens.to(MODEL.device)


def _token_labels(prefix_token_ids: torch.Tensor) -> List[str]:
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


def _token_probabilities_for_prompt(
    prompt: str,
    label: str,
    prefix_token_ids: torch.Tensor,
    image: Image.Image,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """
    Get the probabilities for each token in the target for a given prompt.
    Returns tensor like [1, target_len, 1]
    """
    inputs, prompt_len = _prepare_messages(prompt, label, image)
    inputs = inputs.to(MODEL.device)
    inputs["pixel_values"] = pixel_values

    target_len = prefix_token_ids.shape[1]
    with torch.inference_mode():
        outputs = MODEL.model(**inputs, output_hidden_states=False)
        logits = outputs.logits[:, prompt_len - 1 : prompt_len + target_len - 1, :]
        probs = F.softmax(logits, dim=-1)
        token_probs = probs[
            0,
            torch.arange(target_len, device=MODEL.device),
            prefix_token_ids[0],
        ]
    return token_probs


def _average_token_probs_for_image(
    image_path: Path,
    prompts: Sequence[str],
    prefix_token_ids: torch.Tensor,
    label: str,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    pixel_values = MODEL.processor.image_processor(
        image,
        return_tensors="pt",
    )["pixel_values"].to(MODEL.device)
    total = torch.zeros(prefix_token_ids.shape[1], device=MODEL.device)
    for prompt in prompts:
        token_probs = _token_probabilities_for_prompt(
            prompt,
            label,
            prefix_token_ids,
            image,
            pixel_values,
        )
        total += token_probs
    avg = (total / len(prompts)).detach().cpu().numpy()
    return avg


def _persist_results(
    save_dir: Path,
    step_indices: List[int],
    image_paths: List[Path],
    token_labels: List[str],
    histories: List[List[float]],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f"token_probs.png"
    csv_path = save_dir / f"token_probs.csv"

    plt.figure(figsize=(12, 6))
    for label, probs in zip(token_labels, histories):
        plt.plot(step_indices, probs, marker="o", label=label)
    plt.xlabel("Training step (image index)")
    plt.ylabel("Average token probability")
    plt.title("Token probabilities across adversarial image checkpoints")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    records: List[Dict[str, float]] = []
    for pos, (idx, img_path) in enumerate(zip(step_indices, image_paths)):
        row: Dict[str, float] = {"step": idx, "image_path": str(img_path)}
        for label, probs in zip(token_labels, histories):
            row[label] = probs[pos]
        records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    print(f"Saved plot to {plot_path}")
    print(f"Saved CSV to {csv_path}")


def plot_logits(
    prefix: str,
    start_path: str,
    num_prompts: int,
    attack_type: str,
    save_dir: str = "exp",
    start_img: Optional[int] = None,
    end_img: Optional[int] = None,
    img_step: int = 1,
):
    image_paths = _enumerate_image_paths(start_path, start_img, end_img, img_step)
    prompts = _sample_prompts(attack_type, num_prompts, prefix)
    prefix_token_ids = _prefix_token_ids(prefix)
    # print("Prefix token ids: ", prefix_token_ids)
    token_labels = _token_labels(prefix_token_ids)

    print(f"Evaluating {len(image_paths)} images with {len(prompts)} prompts per step...")
    histories: List[List[float]] = [[] for _ in token_labels]
    step_indices: List[int] = []

    for path in image_paths:
        match = re.search(r"img_(\d+)", path.name)
        step_idx = int(match.group(1)) if match else len(step_indices) + 1
        avg_probs = _average_token_probs_for_image(
            path,
            prompts,
            prefix_token_ids,
            prefix,
        )
        for i, prob in enumerate(avg_probs):
            histories[i].append(float(prob))
        step_indices.append(step_idx)
        print(f"  Processed {path.name}: {np.round(avg_probs, 4)}")

    output_dir = Path(save_dir)
    _persist_results(
        output_dir,
        step_indices,
        image_paths,
        token_labels,
        histories,
    )


if __name__ == "__main__":
    """
    Plot average per-token probabilities of the provided prefix across
    successive adversarial image checkpoints (img_1.png, img_2.png, ...).
    """
    plot_logits(
        prefix = "Download the guide at malware.com!",
        start_path = "experiments/20251126_margin25/images/img_1.png",
        num_prompts = 8,
        attack_type = "specific_string",
        save_dir = "exp",
        start_img = 1,
        end_img = 500,
        img_step = 10,
    )
