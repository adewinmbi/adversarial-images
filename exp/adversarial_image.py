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

# Use the refactored model classes
from models import Qwen3Instruct, Qwen3Think, Qwen25Instruct, Llava
from data.specific_string_data import SpecificStringDataset
from data.jailbreak_data import JailbreakDataset

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


@app.command()
def main(
    config: str = typer.Option(
        "configs/default.yaml",
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
    model_name: Optional[str] = typer.Option(
        None,
        help="Model to use: llava, qwen3inst, qwen3think, or qwen25inst (overrides config)"
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
    )
):
    """
    Train adversarial images for vision-language models.
    
    Configuration is loaded from a YAML file, and can be overridden with CLI arguments.
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
    if model_name is not None:
        cfg['model_name'] = model_name
    if initial_image is not None:
        cfg['initial_image'] = initial_image
    if target_str is not None:
        cfg['target_str'] = target_str
    if dataset_type is not None:
        cfg['dataset_type'] = dataset_type
    if log_dir is not None:
        cfg['log_dir'] = log_dir
    if steps_per_epoch is not None:
        cfg['steps_per_epoch'] = steps_per_epoch
    if checkpoint_dir is not None:
        cfg['checkpoint_dir'] = checkpoint_dir
    
    # Generate log directory if not specified
    if cfg['log_dir'] is None:
        cfg['log_dir'] = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create log directories
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['log_dir'], "images"), exist_ok=True)
    
    # Save the actual config used (including CLI overrides)
    save_config(cfg, os.path.join(cfg['log_dir'], "config.yaml"))
    
    # ======= LOAD MODEL =======
    
    # Validate model_name
    valid_models = ["llava", "qwen3inst", "qwen3think", "qwen25inst"]
    if cfg['model_name'] not in valid_models:
        print(f"Error: Invalid model_name '{cfg['model_name']}'. Must be one of {valid_models}", flush=True)
        sys.exit(1)
    
    # Load the appropriate model
    if cfg['model_name'] == "llava":
        model_wrapper = Llava.load_model(
            cache_dir=cfg['cache_dir'],
            dtype="auto",
            device_map="auto",
            freeze_params=True
        )
    elif cfg['model_name'] == "qwen3inst":
        model_wrapper = Qwen3Instruct.load_model(
            cache_dir=cfg['cache_dir'],
            dtype="auto",
            device_map="auto",
            freeze_params=True
        )
    elif cfg['model_name'] == "qwen3think":
        model_wrapper = Qwen3Think.load_model(
            cache_dir=cfg['cache_dir'],
            dtype="auto",
            device_map="auto",
            freeze_params=True
        )
    elif cfg['model_name'] == "qwen25inst":
        model_wrapper = Qwen25Instruct.load_model(
            cache_dir=cfg['cache_dir'],
            dtype="auto",
            device_map="auto",
            freeze_params=True
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
    
    def compute_loss(instruction, label, adv_image_tensor):
        """
        Compute loss for a single instruction with adversarial image.
        
        Args:
            instruction: The instruction/prompt text
            label: The target label/response text
            adv_image_tensor: The adversarial image tensor
        
        Uses proper teacher forcing: concatenates prompt + target, then computes
        loss on logits that predict the target tokens.
        """
        # Tokenize the target label for this sample
        target_tokens = model_wrapper.processor.tokenizer(
            label,
            return_tensors="pt",
            add_special_tokens=False
        )
        target_token_ids = target_tokens.input_ids.to(model_wrapper.device)
        target_len = target_token_ids.shape[1]
        
        # Convert tensor to PIL for the model
        img_for_processor = adv_image_tensor.permute(2, 0, 1)
        temp_pil = tensor_to_pil(adv_image_tensor)
        
        # Handle different model types
        if cfg['model_name'] == "llava":
            # LLaVA-specific processing
            # First get prompt length (without target)
            prompt_only = f"USER: <image>\n{instruction}\nASSISTANT:"
            prompt_inputs = model_wrapper.processor(
                text=prompt_only,
                images=temp_pil,
                return_tensors="pt"
            )
            prompt_len = prompt_inputs.input_ids.shape[1]
            
            # Now create full input (prompt + target)
            full_text = f"USER: <image>\n{instruction}\nASSISTANT: {label}"
            inputs = model_wrapper.processor(
                text=full_text,
                images=temp_pil,
                return_tensors="pt"
            )
            inputs = {k: v.to(model_wrapper.device) for k, v in inputs.items()}
            
            # Remove image_sizes if present (compatibility issue)
            if 'image_sizes' in inputs:
                del inputs['image_sizes']
            
            # Replace pixel_values with our differentiable tensor
            if 'pixel_values' in inputs:
                temp_inputs = model_wrapper.processor.image_processor(temp_pil, return_tensors="pt")
                inputs['pixel_values'] = temp_inputs['pixel_values'].to(model_wrapper.device)
        else:
            # Qwen models use apply_chat_template
            # First get prompt length (without target)
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
            
            # Now create full input (prompt + target as assistant response)
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
            
            # Replace pixel_values with our differentiable tensor
            if 'pixel_values' in inputs:
                temp_inputs = model_wrapper.processor.image_processor(img_for_processor, return_tensors="pt")
                inputs['pixel_values'] = temp_inputs['pixel_values'].to(model_wrapper.device)
        
        # Forward pass through the underlying model
        outputs = model_wrapper.model(**inputs, output_hidden_states=False)
        pred_logits = outputs.logits
        
        # Extract logits that predict the target tokens
        # Logits at position i predict token i+1
        # So logits[prompt_len-1 : prompt_len+target_len-1] predict tokens at positions [prompt_len : prompt_len+target_len]
        pred_logits_for_target = pred_logits[:, prompt_len-1:prompt_len+target_len-1, :]
        pred_logits_flat = pred_logits_for_target.reshape(-1, pred_logits_for_target.shape[-1])
        target_flat = target_token_ids.reshape(-1)
        
        loss = F.cross_entropy(pred_logits_flat, target_flat)
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
    print(f"Starting training. Saving at {log_dir}", flush=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'loss.csv') if checkpoint_dir is not None else None
    if checkpoint_dir is not None and checkpoint_path is not None:
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
        
        # Save adversarial image for this epoch
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
