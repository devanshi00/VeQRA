"""
File: inference_hybrid.py

Description:
Runs inference using a hybrid fine-tuned Qwen2.5-VL model on both 
VRSBench (JSON) and SkySense (CSV) evaluation sets.
Generates Captioning and VQA results for final benchmarking.

Notes:
Loads data from disparate sources, equalizes sample counts, and combines 
them into a single evaluation stream.
"""

import json
import os
import sys
import random
import itertools
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info 

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Path to the final merged/hybrid adapter
ADAPTER_PATH = "./qwen_hybrid_caption_lora/final_hybrid_checkpoint"

# Output Locations
OUTPUT_CAP = "./results/caption_hybrid_final.json"
OUTPUT_VQA = "./results/vqa_hybrid_final.json"

# Source 1: VRSBench Data
VRS_VAL_IMG = "../dataset/vrsbench/Images_val/"
VRS_CAP_JSON = "../dataset/vrsbench/VRSBench_EVAL_Cap.json"
VRS_VQA_JSON = "../dataset/vrsbench/VRSBench_EVAL_vqa.json"

# Source 2: SkySense Data
SKY_IMG_CAP_DIR = "../dataset/skysense/images_caption/"
SKY_IMG_VQA_DIR = "../dataset/skysense/images_vqa/"
SKY_CAP_CSV = "../dataset/skysense/skysense_caption.csv"
SKY_VQA_CSV = "../dataset/skysense/skysense_vqa.csv"

# Inference Parameters
CAPTION_PROMPT = "Describe this image in detail."
BATCH_SIZE = 10
SAMPLES_PER_SOURCE = 500  # Total = 1000 per task (500 VRS + 500 Sky)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def batched(iterable, n):
    """Yields successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk: return
        yield chunk

# -----------------------------------------------------------------------------
# Data Loading Logic
# -----------------------------------------------------------------------------

def load_vrs_caption_samples(count):
    """
    Loads random captioning samples from VRSBench JSON.
    """
    print(f"[INFO] Loading {count} Caption samples from VRSBench...")
    samples = []
    try:
        with open(VRS_CAP_JSON, 'r') as f:
            data = json.load(f)
        
        random.seed(42)
        random.shuffle(data)
        
        for item in data:
            if len(samples) >= count: break
            
            img_path = os.path.join(VRS_VAL_IMG, item['image_id'])
            if os.path.exists(img_path):
                samples.append({
                    'image_id': item['image_id'],
                    'image_path': img_path,
                    'ground_truth': item['ground_truth'],
                    'dataset': 'VRSBench'
                })
    except Exception as e:
        print(f"[ERROR] Loading VRS Caption data: {e}")
    return samples

def load_sky_caption_samples(count):
    """
    Loads random captioning samples from SkySense CSV.
    """
    print(f"[INFO] Loading {count} Caption samples from SkySense...")
    samples = []
    try:
        df = pd.read_csv(SKY_CAP_CSV)
        df.columns = df.columns.str.strip()
        
        if len(df) > count: 
            df = df.sample(n=count, random_state=42)
            
        for _, row in df.iterrows():
            img_name = str(row['image'])
            img_path = os.path.join(SKY_IMG_CAP_DIR, img_name)
            
            if os.path.exists(img_path):
                samples.append({
                    'image_id': img_name,
                    'image_path': img_path,
                    'ground_truth': str(row['caption']),
                    'dataset': 'SkySense'
                })
    except Exception as e:
        print(f"[ERROR] Loading SkySense Caption data: {e}")
    return samples

def load_vrs_vqa_samples(count):
    """
    Loads random VQA samples from VRSBench JSON.
    """
    print(f"[INFO] Loading {count} VQA samples from VRSBench...")
    samples = []
    try:
        with open(VRS_VQA_JSON, 'r') as f:
            data = json.load(f)
            
        random.seed(42)
        random.shuffle(data)
        
        for item in data:
            if len(samples) >= count: break
            
            img_path = os.path.join(VRS_VAL_IMG, item['image_id'])
            if os.path.exists(img_path):
                samples.append({
                    'image_id': item['image_id'],
                    'image_path': img_path,
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'dataset': 'VRSBench'
                })
    except Exception as e:
        print(f"[ERROR] Loading VRS VQA data: {e}")
    return samples

def load_sky_vqa_samples(count):
    """
    Loads random VQA samples from SkySense CSV.
    Handles potential column name variations (query/question, response/answer).
    """
    print(f"[INFO] Loading {count} VQA samples from SkySense...")
    samples = []
    try:
        df = pd.read_csv(SKY_VQA_CSV)
        df.columns = df.columns.str.strip()
        
        # Resolve column names dynamically
        img_col = 'image' if 'image' in df.columns else df.columns[0]
        q_col = 'query' if 'query' in df.columns else df.columns[1]
        a_col = 'response' if 'response' in df.columns else df.columns[2]

        if len(df) > count: 
            df = df.sample(n=count, random_state=42)
            
        for _, row in df.iterrows():
            img_name = str(row[img_col])
            img_path = os.path.join(SKY_IMG_VQA_DIR, img_name)
            
            if os.path.exists(img_path):
                samples.append({
                    'image_id': img_name,
                    'image_path': img_path,
                    'question': str(row[q_col]),
                    'ground_truth': str(row[a_col]),
                    'dataset': 'SkySense'
                })
    except Exception as e:
        print(f"[ERROR] Loading SkySense VQA data: {e}")
    return samples

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

def load_model():
    """
    Loads the Base Qwen2.5-VL model and attaches the Hybrid LoRA adapter.
    """
    print(f"[INFO] Loading Base Model: {BASE_MODEL_ID}...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL_ID, 
            use_fast=True, 
            trust_remote_code=True
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, 
            quantization_config=quant_config, 
            device_map="auto", 
            trust_remote_code=True
        )
        
        # Load Hybrid Adapter
        if os.path.exists(ADAPTER_PATH):
            print(f"[INFO] Attaching Hybrid Adapter: {ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        else:
            print(f"[WARN] Adapter not found at {ADAPTER_PATH}. Proceeding with Base Model.")

        model.eval()
        return model, processor
        
    except Exception as e:
        print(f"[FATAL] Model load failed: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Inference Task Runners
# -----------------------------------------------------------------------------

def run_captioning(model, processor, data):
    """
    Runs batched caption generation on the combined dataset.
    """
    print(f"\n[INFO] Running Captioning on {len(data)} samples (Batched)...")
    results = []
    batches = list(batched(data, BATCH_SIZE))
    
    for batch in tqdm(batches, desc="Caption Inference"):
        batch_prompts = []
        batch_images = []
        
        # Prepare Batch
        for item in batch:
            try:
                pil_img = Image.open(item['image_path']).convert("RGB")
                
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": pil_img}, 
                            {"type": "text", "text": CAPTION_PROMPT}
                        ]
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                batch_prompts.append(text)
                batch_images.append(pil_img)
            except: 
                continue
        
        if not batch_prompts: continue
        
        try:
            # Inference
            inputs = processor(
                text=batch_prompts, 
                images=batch_images, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=200)
                
            decoded = []
            for i in range(len(ids)):
                out = ids[i][len(inputs['input_ids'][i]):]
                decoded.append(
                    processor.decode(out, skip_special_tokens=True).strip()
                )
            
            # Map results
            for item, ans in zip(batch, decoded):
                results.append({
                    "image_id": item['image_id'],
                    "dataset": item['dataset'],
                    "ground_truth": item['ground_truth'],
                    "model_answer": ans
                })
                
        except Exception as e:
            print(f"[ERROR] Caption Batch Failed: {e}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CAP), exist_ok=True)
    with open(OUTPUT_CAP, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[INFO] Caption results saved to: {OUTPUT_CAP}")

def run_vqa(model, processor, data):
    """
    Runs VQA inference sequentially (One-by-One).
    """
    print(f"\n[INFO] Running VQA on {len(data)} samples (One-by-One)...")
    results = []
    
    for item in tqdm(data, desc="VQA Inference"):
        try:
            pil_img = Image.open(item['image_path']).convert("RGB")
            
            prompt = f"{item['question']} Answer with a single word or number."
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": pil_img}, 
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            inputs = processor(
                text=[text], 
                images=image_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=10)
                
            out = ids[0][len(inputs['input_ids'][0]):]
            ans = processor.decode(out, skip_special_tokens=True).strip()
            
            results.append({
                "image_id": item['image_id'],
                "dataset": item['dataset'],
                "question": item['question'],
                "ground_truth": item['ground_truth'],
                "model_answer": ans
            })
            
        except Exception as e:
            print(f"[WARN] VQA Error for {item['image_id']}: {e}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_VQA), exist_ok=True)
    with open(OUTPUT_VQA, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[INFO] VQA results saved to: {OUTPUT_VQA}")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("[INFO] Starting Hybrid Inference Pipeline...")
    
    # 1. Load Data
    print("[INFO] Aggregating evaluation data...")
    
    # Captioning Data
    vrs_cap = load_vrs_caption_samples(SAMPLES_PER_SOURCE)
    sky_cap = load_sky_caption_samples(SAMPLES_PER_SOURCE)
    full_cap_data = vrs_cap + sky_cap
    random.shuffle(full_cap_data)
    
    # VQA Data
    vrs_vqa = load_vrs_vqa_samples(SAMPLES_PER_SOURCE)
    sky_vqa = load_sky_vqa_samples(SAMPLES_PER_SOURCE)
    full_vqa_data = vrs_vqa + sky_vqa
    random.shuffle(full_vqa_data)
    
    if not full_cap_data and not full_vqa_data:
        print("[FATAL] No data loaded. Please check dataset paths.")
        sys.exit(1)
        
    print(f"[INFO] Data Ready: {len(full_cap_data)} Caption samples, {len(full_vqa_data)} VQA samples.")
    
    # 2. Load Model
    model, processor = load_model()
    
    # 3. Run Inference
    if full_cap_data:
        run_captioning(model, processor, full_cap_data)
    else:
        print("[INFO] No Caption data found. Skipping.")
        
    if full_vqa_data:
        run_vqa(model, processor, full_vqa_data)
    else:
        print("[INFO] No VQA data found. Skipping.")
        
    print("\n[INFO] All Hybrid Inference tasks complete.")