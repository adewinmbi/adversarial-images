import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ======= PARAMETERS =======
CACHE_DIR = '/home/adewinmb/orcd/scratch'
TARGET_STR = "Download the guide at malware.com!"

# Training parameters
NUM_EPOCHS = 500
INITIAL_IMAGE = "black"  # Options: "black", "noise", "expedia"
EPSILON = 16 / 255  # Maximum deviation from original image
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
LOG_DIR = "experiments/" + datetime.now().strftime("%Y%m%d_%H%M%S")
NUM_EXAMPLES = 10  # Number of test examples to generate after training
LEARNING_RATE = 0.01

# Create log directory and subdirectory for images
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "images"), exist_ok=True)

# ======= LOAD MODEL =======
print("Loading model...", flush=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto", cache_dir=CACHE_DIR
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", cache_dir=CACHE_DIR)

print("Freezing model parameters...", flush=True)
# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Force CUDA synchronization
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("Model loaded and frozen.", flush=True)

# ======= LOAD AND SPLIT DATASET =======
print("Loading dataset...", flush=True)
with open('data/alpaca.json', 'r') as f:
    dataset = json.load(f)

# Extract instructions only
instructions = [item['instruction'] for item in dataset]

# Split dataset
np.random.seed(42)
indices = np.random.permutation(len(instructions))
train_end = int(TRAIN_SPLIT * len(indices))
val_end = int((TRAIN_SPLIT + VAL_SPLIT) * len(indices))

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

train_instructions = [instructions[i] for i in train_indices]
val_instructions = [instructions[i] for i in val_indices]
test_instructions = [instructions[i] for i in test_indices]

print(f"Dataset split: {len(train_instructions)} train, {len(val_instructions)} val, {len(test_instructions)} test", flush=True)

# ======= GENERATE TARGET LABEL =======
print("Generating target label...", flush=True)

# Tokenize TARGET_STR to get token IDs for teacher forcing
# We just need the text tokens, no image
target_tokens = processor.tokenizer(
    TARGET_STR,
    return_tensors="pt",
    add_special_tokens=False  # Don't add special tokens, we just want the content
)
target_token_ids = target_tokens.input_ids.to(model.device)  # [1, seq_len]

print(f"Target token IDs shape: {target_token_ids.shape}", flush=True)
print(f"Target tokens: {target_token_ids[0].tolist()}", flush=True)

# ======= INITIALIZE ADVERSARIAL IMAGE =======
print(f"Initializing adversarial image with method: {INITIAL_IMAGE}", flush=True)

if INITIAL_IMAGE == "black":
    # Black image (224x224 RGB)
    adv_image_np = np.zeros((224, 224, 3), dtype=np.uint8)
    original_image_np = adv_image_np.copy()
elif INITIAL_IMAGE == "noise":
    # Random noise image
    adv_image_np = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    original_image_np = adv_image_np.copy()
elif INITIAL_IMAGE == "expedia":
    # Load expedia.png
    original_pil = Image.open("data/expedia.png").convert("RGB")
    original_pil = original_pil.resize((224, 224))
    original_image_np = np.array(original_pil)
    adv_image_np = original_image_np.copy()
else:
    raise ValueError(f"Unknown initial image type: {INITIAL_IMAGE}")

# Convert to tensor and make it a trainable parameter
# Normalize to [0, 1] for optimization
adv_image_tensor = torch.from_numpy(adv_image_np.astype(np.float32) / 255.0).to(model.device)
original_image_tensor = torch.from_numpy(original_image_np.astype(np.float32) / 255.0).to(model.device)
adv_image_tensor.requires_grad = True

# Optimizer for the image
optimizer = torch.optim.Adam([adv_image_tensor], lr=LEARNING_RATE)

# ======= HELPER FUNCTIONS =======
def tensor_to_pil(tensor):
    """Convert tensor [H, W, C] in range [0, 1] to PIL Image"""
    img_np = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)

def compute_loss(instruction, adv_image_tensor):
    """Compute loss for a single instruction with adversarial image using teacher forcing"""
    # Prepare image tensor: processor expects [B, C, H, W] or [C, H, W]
    # Our tensor is [H, W, C] in range [0, 1]
    img_for_processor = adv_image_tensor.permute(2, 0, 1)  # [C, H, W]
    
    # Convert to PIL for apply_chat_template (it needs PIL or path)
    # Note: This breaks gradient flow, so we'll need to replace pixel_values after
    temp_pil = tensor_to_pil(adv_image_tensor)
    
    # Use apply_chat_template to properly format with image placeholder tokens
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": temp_pil},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Replace pixel_values with our differentiable tensor to maintain gradients
    if 'pixel_values' in inputs:
        # Process our adversarial tensor through the image processor
        # This maintains gradients because processor operations on tensors are differentiable
        temp_inputs = processor.image_processor(img_for_processor, return_tensors="pt")
        inputs['pixel_values'] = temp_inputs['pixel_values'].to(model.device) 
    
    # Forward pass - unpack the dict
    outputs = model(**inputs, output_hidden_states=False)
    pred_logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    # Teacher forcing: use target tokens as labels
    # We want the last N tokens of the output to match our target tokens
    # where N = len(target_token_ids)
    target_len = target_token_ids.shape[1]
    
    # Take the last target_len positions from predictions
    # These are the logits that should predict our target tokens
    pred_logits_for_target = pred_logits[:, -target_len:, :]  # [batch, target_len, vocab_size]
    
    # Reshape for cross_entropy: [batch * target_len, vocab_size]
    pred_logits_flat = pred_logits_for_target.reshape(-1, pred_logits_for_target.shape[-1])
    
    # Expand target tokens to match batch if needed and flatten
    target_flat = target_token_ids.expand(pred_logits_for_target.shape[0], -1).reshape(-1)
    
    # Compute teacher-forced cross-entropy loss
    loss = F.cross_entropy(pred_logits_flat, target_flat)
    
    return loss

def evaluate(instructions, adv_image_tensor):
    """Evaluate on a set of instructions"""
    total_loss = 0.0
    with torch.no_grad():
        for instruction in instructions[:100]:  # Limit to 100 for speed
            loss = compute_loss(instruction, adv_image_tensor)
            total_loss += loss.item()
    return total_loss / min(len(instructions), 100)

# ======= TRAINING LOOP =======
print("Starting training...", flush=True)
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}", flush=True)
    epoch_train_loss = 0.0
    
    # Shuffle training data
    np.random.shuffle(train_instructions)
    
    # Training
    for i, instruction in enumerate(train_instructions[:100]):  # Limit for prototyping
        optimizer.zero_grad()
        
        loss = compute_loss(instruction, adv_image_tensor)
        loss.backward()
        
        # Gradient step
        optimizer.step()
        
        # Project back to epsilon ball around original image
        with torch.no_grad():
            delta = adv_image_tensor - original_image_tensor
            delta = torch.clamp(delta, -EPSILON, EPSILON)
            adv_image_tensor.data = original_image_tensor + delta
            adv_image_tensor.data = torch.clamp(adv_image_tensor.data, 0, 1)
        
        epoch_train_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i + 1}/{min(len(train_instructions), 100)}, Loss: {loss.item():.4f}")
    
    avg_train_loss = epoch_train_loss / min(len(train_instructions), 100)
    train_losses.append(avg_train_loss)
    
    # Validation
    print("  Validating...")
    avg_val_loss = evaluate(val_instructions, adv_image_tensor)
    val_losses.append(avg_val_loss)
    
    print(f"  Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}", flush=True)
    
    # Save adversarial image for this epoch
    epoch_adv_pil = tensor_to_pil(adv_image_tensor)
    epoch_adv_pil.save(os.path.join(LOG_DIR, "images", f"img_{epoch + 1}.png"))
    
    # Update and save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epoch + 2), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, "loss_plot.png"))
    plt.close()  # Close the figure to free memory

# ======= SAVE FINAL ADVERSARIAL IMAGE =======
print("\nSaving final adversarial image...", flush=True)
final_adv_pil = tensor_to_pil(adv_image_tensor)
final_adv_pil.save(os.path.join(LOG_DIR, "adversarial_image.png"))
print(f"Saved to {os.path.join(LOG_DIR, 'adversarial_image.png')}", flush=True)
print(f"Loss plot saved to {os.path.join(LOG_DIR, 'loss_plot.png')}", flush=True)
print(f"Per-epoch images saved to {os.path.join(LOG_DIR, 'images/')}", flush=True)

# ======= GENERATE TEST EXAMPLES =======
print(f"\nGenerating {NUM_EXAMPLES} test examples...", flush=True)
final_adv_pil = tensor_to_pil(adv_image_tensor)

test_results = []
for i in range(min(NUM_EXAMPLES, len(test_instructions))):
    instruction = test_instructions[i]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": final_adv_pil},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    test_results.append({
        "instruction": instruction,
        "output": output_text
    })
    
    print(f"  Example {i + 1}/{NUM_EXAMPLES}")
    print(f"    Instruction: {instruction}")
    print(f"    Output: {output_text[:100]}...")

# Save test results
with open(os.path.join(LOG_DIR, "test_examples.txt"), 'w') as f:
    for i, result in enumerate(test_results):
        f.write(f"Example {i + 1}\n")
        f.write(f"Instruction: {result['instruction']}\n")
        f.write(f"Output: {result['output']}\n")
        f.write("-" * 80 + "\n\n")

print(f"Saved test examples to {os.path.join(LOG_DIR, 'test_examples.txt')}", flush=True)
print("\nTraining complete!", flush=True)



