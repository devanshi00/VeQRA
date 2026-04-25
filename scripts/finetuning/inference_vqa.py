"""
File: inference_vqa.py

Description:
Runs inference using a VQA fine-tuned Qwen2.5-VL model.
Evaluates on both VQA (Target Task) and Captioning (for capability retention checks).

Notes:
VQA inference is run sequentially to ensure precise generation for short answers,
while Captioning is batched for throughput.
"""

import json
import os
import torch
import itertools
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
# Path to the VQA-specific LoRA adapter
ADAPTER_PATH = "./qwen_vrsbench_vqa_lora/final_vqa_checkpoint"

# Output Files
OUTPUT_VQA = "./results/vqa_finetuned.json"
OUTPUT_CAP = "./results/caption_vqa_finetuned.json"

# Dataset Paths
VRS_VAL_IMG = "../dataset/vrsbench/Images_val/"
VRS_VQA_JSON = "../dataset/vrsbench/VRSBench_EVAL_vqa.json"
VRS_CAP_JSON = "../dataset/vrsbench/VRSBench_EVAL_Cap.json"

# Inference Settings
BATCH_SIZE = 10
SAMPLE_LIMIT = 1000 

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

def load_vqa_data():
    """Loads VQA evaluation data, limited to SAMPLE_LIMIT."""
    print(f"[INFO] Loading VQA Evaluation Set: {os.path.basename(VRS_VQA_JSON)}")
    try:
        with open(VRS_VQA_JSON, 'r') as f:
            data = json.load(f)
        
        if len(data) > SAMPLE_LIMIT:
            print(f"[INFO] Limiting VQA samples to {SAMPLE_LIMIT} (Original: {len(data)})")
            data = data[:SAMPLE_LIMIT]
            
        return data
    except Exception as e:
        print(f"[ERROR] Loading VQA JSON: {e}")
        return []

def load_caption_data():
    """Loads Caption evaluation data, limited to SAMPLE_LIMIT."""
    print(f"[INFO] Loading Caption Evaluation Set: {os.path.basename(VRS_CAP_JSON)}")
    try:
        with open(VRS_CAP_JSON, 'r') as f:
            data = json.load(f)
            
        if len(data) > SAMPLE_LIMIT:
            print(f"[INFO] Limiting Caption samples to {SAMPLE_LIMIT} (Original: {len(data)})")
            data = data[:SAMPLE_LIMIT]
            
        return data
    except Exception as e:
        print(f"[ERROR] Loading Caption JSON: {e}")
        return []

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

def load_model():
    """
    Loads Qwen2.5-VL with 4-bit quantization and attaches the VQA LoRA adapter.
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
        
        if os.path.exists(ADAPTER_PATH):
            print(f"[INFO] Attaching VQA Adapter: {ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        else:
            print(f"[WARN] Adapter not found at {ADAPTER_PATH}. Running Base Model.")

        model.eval()
        return model, processor
        
    except Exception as e:
        print(f"[FATAL] Model load failed: {e}")
        return None, None

# -----------------------------------------------------------------------------
# Inference Task Runners
# -----------------------------------------------------------------------------

def run_vqa(model, processor, data):
    """
    Runs VQA inference sequentially (One-by-One).
    """
    print(f"\n[INFO] Running VQA on {len(data)} samples (Sequential)...")
    results = []
    
    for item in tqdm(data, desc="VQA Inference"):
        img_path = os.path.join(VRS_VAL_IMG, item['image_id'])
        try:
            image = Image.open(img_path).convert("RGB")
            
            prompt = f"{item['question']} Answer concisely in a single word or number."
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image}, 
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
                # Short generation for concise VQA
                ids = model.generate(**inputs, max_new_tokens=10)
                
            out = ids[0][len(inputs['input_ids'][0]):]
            ans = processor.decode(out, skip_special_tokens=True).strip()
            
            # Merge original item data with model answer
            results.append({
                **item, 
                "model_answer": ans
            })
            
        except Exception as e:
            print(f"[WARN] VQA Error on {item['image_id']}: {e}")

    # Save Results
    os.makedirs(os.path.dirname(OUTPUT_VQA), exist_ok=True)
    with open(OUTPUT_VQA, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[INFO] VQA results saved to: {OUTPUT_VQA}")

def run_captioning(model, processor, data):
    """
    Runs Captioning inference in batches.
    """
    print(f"\n[INFO] Running Captioning on {len(data)} samples (Batched)...")
    results = []
    batches = list(batched(data, BATCH_SIZE))
    
    for batch in tqdm(batches, desc="Caption Inference"):
        batch_prompts = []
        batch_images = []
        
        # Prepare Batch
        for item in batch:
            img_path = os.path.join(VRS_VAL_IMG, item['image_id'])
            try:
                pil_img = Image.open(img_path).convert("RGB")
                
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": pil_img}, 
                            {"type": "text", "text": "Describe this image in detail."}
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
                    "ground_truth": item.get('ground_truth') or item.get('caption'),
                    "dataset": "VRSBench",
                    "model_answer": ans
                })
                
        except Exception as e:
            print(f"[ERROR] Caption Batch Failed: {e}")

    # Save Results
    os.makedirs(os.path.dirname(OUTPUT_CAP), exist_ok=True)
    with open(OUTPUT_CAP, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[INFO] Caption results saved to: {OUTPUT_CAP}")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load Model
    model, processor = load_model()
    
    if model:
        # 2. Run VQA (Target Task)
        vqa_data = load_vqa_data()
        if vqa_data:
            run_vqa(model, processor, vqa_data)
        else:
            print("[WARN] No VQA data found.")
            
        # Clear VRAM before next heavy task
        torch.cuda.empty_cache()
        print("[INFO] GPU memory cleared.")
        
        # 3. Run Captioning (Secondary Task)
        cap_data = load_caption_data()
        if cap_data:
            run_captioning(model, processor, cap_data)
        else:
            print("[WARN] No Caption data found.")
            
        print("\n[INFO] All inference tasks complete.")