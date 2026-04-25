"""
File: multimodality_pipeline.py

Description:
End-to-end pipeline for fine-tuning Qwen2.5-VL on specific multimodal datasets 
(SAR or IR). Handles data indexing, splitting, LoRA training, and 
immediate validation inference.

Notes: Includes robust path resolution strategies to handle legacy dataset directory structures.
"""

import json
import os
import sys
import random
import argparse
import itertools
import gc
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DATA_DIR = "../dataset/mmrs-1m/gpt_generated_data/"
DATASET_ROOT = "../dataset/mmrs-1m/"
OUTPUT_BASE = "./multimodal_adapters"
RESULTS_DIR = "./results"

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTRUCTION_PROMPT_CAP = "Describe this image in detail."
BATCH_SIZE_INFERENCE = 10

# -----------------------------------------------------------------------------
# Path Resolution & Data Helpers
# -----------------------------------------------------------------------------

def resolve_image_path(path_in_json, image_index):
    """
    Attempts to resolve the absolute path of an image using multiple strategies.
    Useful when JSON metadata contains relative, legacy, or inconsistent paths.
    
    Args:
        path_in_json (str): The path string found in the dataset JSON.
        image_index (dict): A fallback lookup dictionary (filename -> full_path).
    """
    # 1. Check if the path is already absolute and valid
    if os.path.exists(path_in_json): 
        return path_in_json
    
    # 2. Handle legacy naming convention ("mmrs1m_filtered" -> "mmrs-1m")
    clean_path = path_in_json.replace("mmrs1m_filtered", "mmrs-1m")
    if os.path.exists(clean_path): 
        return clean_path
    
    # 3. Handle relative paths starting with "data/"
    rel = clean_path.replace("data/", "", 1) if clean_path.startswith("data/") else clean_path
    if rel.startswith("/"): 
        rel = rel[1:]
    
    # Try joining with the known dataset root
    full = os.path.join(DATASET_ROOT, rel)
    if os.path.exists(full): 
        return full
    
    # 4. Handle potential nested root prefixes
    if rel.startswith("mmrs-1m/"):
        rel = rel.replace("mmrs-1m/", "", 1)
        full = os.path.join(DATASET_ROOT, rel)
        if os.path.exists(full): 
            return full

    # 5. Fallback: Look up by filename in the pre-built index
    filename = os.path.basename(path_in_json)
    return image_index.get(filename)

def build_image_index():
    """
    Recursively scans the dataset root to build a map of filenames to absolute paths.
    This serves as the ultimate fallback for broken paths.
    """
    print(f"[INFO] Indexing images in {DATASET_ROOT}...")
    index = {}
    for root, _, files in os.walk(DATASET_ROOT):
        for file in files:
            if file not in index: 
                index[file] = os.path.join(root, file)
    print(f"[INFO] Indexed {len(index)} images.")
    return index

def batched(iterable, n):
    """Yields successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk: return
        yield chunk

def prepare_data_splits(modality, image_index):
    """
    Loads Caption and VQA data for the specified modality (SAR/IR).
    Resolves image paths and performs a Train/Validation split (90/10).
    """
    print(f"\n[INFO] Loading and Splitting Data ({modality})...")
    grouped_data = defaultdict(list)
    
    # 1. Load Caption Data
    cap_file = os.path.join(DATA_DIR, f"{modality}_gpt_caption.json")
    if os.path.exists(cap_file):
        with open(cap_file, 'r') as f:
            data = json.load(f)
        for item in data:
            raw_path = item.get('image') or item.get('image_path')
            if not raw_path: continue
            
            path = resolve_image_path(raw_path, image_index)
            if path:
                grouped_data[path].append({
                    "image": path,
                    "prompt": INSTRUCTION_PROMPT_CAP,
                    "answer": item.get('caption') or item.get('ground_truth'),
                    "type": "caption",
                    "image_id": os.path.basename(path)
                })

    # 2. Load VQA Data
    vqa_file = os.path.join(DATA_DIR, f"{modality}_gpt_vqa.json")
    if os.path.exists(vqa_file):
        with open(vqa_file, 'r') as f:
            data = json.load(f)
        for item in data:
            raw_path = item.get('image') or item.get('image_path')
            if not raw_path: continue
            
            path = resolve_image_path(raw_path, image_index)
            if path:
                # Handle nested qa_pairs structure or flat structure
                if 'qa_pairs' in item:
                    for pair in item['qa_pairs']:
                        q = f"{pair['question']} Answer concisely in a single word or number."
                        grouped_data[path].append({
                            "image": path,
                            "prompt": q,
                            "answer": str(pair.get('answer') or pair.get('ground_truth')),
                            "type": "vqa",
                            "image_id": os.path.basename(path)
                        })
                elif 'question' in item:
                    q = f"{item['question']} Answer concisely in a single word or number."
                    grouped_data[path].append({
                        "image": path,
                        "prompt": q,
                        "answer": str(item.get('answer') or item.get('ground_truth')),
                        "type": "vqa",
                        "image_id": os.path.basename(path)
                    })

    # 3. Perform Split based on unique images (to avoid data leakage)
    unique_images = list(grouped_data.keys())
    print(f"[INFO] Found {len(unique_images)} unique images with valid data.")
    
    if len(unique_images) == 0:
        print("[FATAL] No valid data found. Check paths and JSON files.")
        sys.exit(1)

    train_imgs, val_imgs = train_test_split(unique_images, test_size=0.1, random_state=42)
    
    train_samples = []
    for img in train_imgs: 
        train_samples.extend(grouped_data[img])
        
    val_samples = []
    for img in val_imgs: 
        val_samples.extend(grouped_data[img])
        
    print(f"[INFO] Training Set: {len(train_samples)} samples")
    print(f"[INFO] Validation Set: {len(val_samples)} samples")
    
    return train_samples, val_samples

# -----------------------------------------------------------------------------
# Dataset & Collator Implementation
# -----------------------------------------------------------------------------

class InMemoryDataset(torch.utils.data.Dataset):
    """
    Dataset that holds metadata in memory and loads images on-the-fly.
    """
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor
        self.eos_token = processor.tokenizer.eos_token

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item['image']).convert("RGB")
        except:
            # Fallback for corrupt images
            new_idx = random.randint(0, len(self.samples)-1)
            return self.__getitem__(new_idx)

        return {
            "image": image,
            "prompt": item['prompt'],
            "answer": item['answer'] + self.eos_token
        }

class CustomDataCollator:
    """
    Collator that handles Chat Template application and label masking.
    Ensures loss is only computed on the assistant's response.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: return {}

        images = [f["image"] for f in features]
        prompts = [f["prompt"] for f in features]
        answers = [f["answer"] for f in features]
        
        prompt_texts, full_texts = [], []
        
        # Apply Chat Template
        for p, a in zip(prompts, answers):
            msg_prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            msg_full = msg_prompt + [{"role": "assistant", "content": [{"type": "text", "text": a}]}]
            
            prompt_texts.append(self.processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(msg_full, tokenize=False))

        # Tokenize
        full_inputs = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        prompt_inputs = self.processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
        
        # Mask Labels
        labels = full_inputs["input_ids"].clone()
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
        
        for i in range(len(labels)): 
            labels[i, :prompt_lens[i]] = -100
        
        batch = full_inputs
        batch["labels"] = labels
        return batch

# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------

def run_inference(model, processor, val_samples, modality):
    """
    Runs inference on the validation set.
    - Uses Batched inference for Captioning tasks.
    - Uses Sequential inference for VQA tasks.
    """
    # Fix padding side for generation
    processor.tokenizer.padding_side = "left"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    cap_data = [d for d in val_samples if d['type'] == 'caption']
    vqa_data = [d for d in val_samples if d['type'] == 'vqa']
    
    # 1. Captioning (Batched)
    if cap_data:
        output_file = os.path.join(RESULTS_DIR, f"caption_{modality}_results.json")
        print(f"\n[INFO] Running Captioning Inference -> {os.path.basename(output_file)}")
        
        results = []
        batches = list(batched(cap_data, BATCH_SIZE_INFERENCE))
        
        for batch in tqdm(batches, desc="Captioning"):
            batch_prompts, batch_images, batch_items = [], [], []
            for item in batch:
                try:
                    img = Image.open(item['image']).convert("RGB")
                    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": INSTRUCTION_PROMPT_CAP}]}]
                    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    
                    batch_prompts.append(text)
                    batch_images.append(img)
                    batch_items.append(item)
                except: continue
            
            if not batch_prompts: continue
            
            try:
                inputs = processor(text=batch_prompts, images=batch_images, padding=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=200)
                
                decoded = processor.batch_decode(ids, skip_special_tokens=True)
                
                for item, ans in zip(batch_items, decoded):
                    # Clean up artifact if model repeats 'assistant' token
                    clean = ans.split("assistant\n")[-1].strip() if "assistant\n" in ans else ans.strip()
                    results.append({
                        "image_id": item['image_id'],
                        "ground_truth": item['answer'],
                        "model_answer": clean
                    })
            except Exception as e: 
                print(f"[WARN] Batch Error: {e}")
            
        with open(output_file, 'w') as f: 
            json.dump(results, f, indent=2)

    # 2. VQA (One-by-One)
    if vqa_data:
        output_file = os.path.join(RESULTS_DIR, f"vqa_{modality}_results.json")
        print(f"\n[INFO] Running VQA Inference -> {os.path.basename(output_file)}")
        
        results = []
        for item in tqdm(vqa_data, desc="VQA"):
            try:
                img = Image.open(item['image']).convert("RGB")
                msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": item['prompt']}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(msgs)
                
                inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=10)
                
                out = ids[0][len(inputs['input_ids'][0]):]
                ans = processor.decode(out, skip_special_tokens=True).strip()
                
                results.append({
                    "image_id": item['image_id'],
                    "question": item['prompt'],
                    "ground_truth": item['answer'],
                    "model_answer": ans
                })
            except Exception as e: 
                print(f"[WARN] VQA Error: {e}")

        with open(output_file, 'w') as f: 
            json.dump(results, f, indent=2)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multimodal Fine-Tuning Pipeline (SAR/IR)")
    parser.add_argument("--modality", type=str, required=True, choices=["sar", "ir"], help="Target modality")
    args = parser.parse_args()
    
    modality = args.modality
    output_dir = os.path.join(OUTPUT_BASE, f"qwen_{modality}_adapter")

    print(f"\n=== Pipeline Started for Modality: {modality.upper()} ===")
    
    # 1. Prepare Data
    img_idx = build_image_index()
    train_samples, val_samples = prepare_data_splits(modality, img_idx)

    # 2. Load Model & Processor
    print("\n[INFO] Loading Model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=quant_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    # 3. Training Setup
    print("\n[INFO] Configuring LoRA and Trainer...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    train_ds = InMemoryDataset(train_samples, processor)
    val_ds = InMemoryDataset(val_samples, processor)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch", 
        eval_strategy="epoch", 
        fp16=False, 
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds, 
        data_collator=CustomDataCollator(processor)
    )
    
    # 4. Train
    print("[INFO] Starting Training...")
    trainer.train()
    
    final_save = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_save)
    print(f"[INFO] Training Complete. Adapter saved to: {final_save}")

    # 5. Inference
    print("\n[INFO] Starting Validation Inference...")
    
    # Clear VRAM for inference
    torch.cuda.empty_cache()
    gc.collect()
    
    model.eval()
    run_inference(model, processor, val_samples, modality)
    
    print(f"\n=== Pipeline Complete for {modality.upper()} ===")

if __name__ == "__main__":
    main()