"""
File: inference_caption.py

Description:
Runs inference on a series of Qwen2.5-VL checkpoints (Base + LoRA Adapters).
Performs both Visual Question Answering (VQA) and Image Captioning tasks.

Notes:
Handles Qwen-specific image processing via 'qwen_vl_utils'.
Includes memory management to clear GPU cache between checkpoint evaluations.
"""

import json
import os
import torch
import itertools
from collections import defaultdict
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

# Directory containing fine-tuned checkpoints
CHECKPOINT_DIR = "./qwen_vrsbench_caption_lora"
RESULTS_DIR = "./results"

# Datasets
VQA_EVAL_JSON = "../dataset/vrsbench/VRSBench_EVAL_vqa.json"
CAP_EVAL_JSON = "../dataset/vrsbench/VRSBench_EVAL_Cap.json"
IMAGE_DIR = "../dataset/vrsbench/Images_val/"

# Inference Settings
BATCH_SIZE = 10
IMAGE_LIMIT = 1000 

# -----------------------------------------------------------------------------
# Checkpoint Selection Logic
# -----------------------------------------------------------------------------
# 1. Start with the Base Model (None represents no adapter)
CHECKPOINTS: list = [None]

# 2. Add intermediate checkpoints (Steps 200 to 1200)
# Adjust the range parameters (start, stop, step) based on your training run
for i in range(200, 1267, 200):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-{i}")
    if os.path.exists(ckpt_path):
        CHECKPOINTS.append(ckpt_path)

# 3. Add the final checkpoint
final_ckpt = os.path.join(CHECKPOINT_DIR, "final_checkpoint")
if os.path.exists(final_ckpt):
    CHECKPOINTS.append(final_ckpt)

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


def get_image_files(image_base_path):
    """Retrieves a limited list of images for inference."""
    try:
        all_files = os.listdir(image_base_path)
        return all_files[:IMAGE_LIMIT]
    except Exception as e:
        print(f"[ERROR] Reading image dir: {e}")
        return []


def load_data(json_path):
    """Loads evaluation JSON and maps queries to Image IDs."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        mapping = defaultdict(list)
        for item in data:
            mapping[item['image_id']].append(item)
        return mapping
    except Exception as e:
        print(f"[ERROR] Loading {json_path}: {e}")
        return {}

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

def load_model(adapter_path):
    """
    Loads the Qwen2.5-VL model with 4-bit quantization.
    If 'adapter_path' is provided, loads the LoRA adapter via PeftModel.
    """
    adapter_name = adapter_path if adapter_path else 'Base Model'
    print(f"\n[INFO] Loading Model (Adapter: {adapter_name})...")
    
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
        
        if adapter_path:
            print(f"[INFO] Attaching LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            
        model.eval()
        return model, processor
        
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return None, None

# -----------------------------------------------------------------------------
# Inference Task Runners
# -----------------------------------------------------------------------------

def run_vqa(model, processor, images_list, query_map, output_file):
    """
    Runs VQA inference sequentially (One-by-One) to ensure stability.
    Uses 'process_vision_info' for correct Qwen-VL input formatting.
    """
    print(f"[INFO] Running VQA Inference -> {os.path.basename(output_file)}")
    results = []
    
    for img_file in tqdm(images_list, desc="VQA"):
        if img_file not in query_map: 
            continue
        
        img_path = os.path.join(IMAGE_DIR, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except: 
            continue
            
        for query in query_map[img_file]:
            try:
                # Construct Prompt
                prompt = f"{query['question']} Answer with a single word or number."
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": image}, 
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Preprocess Inputs
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
                
                # Generate Answer
                with torch.no_grad():
                    # Short max_new_tokens for VQA concise answers
                    ids = model.generate(**inputs, max_new_tokens=10) 
                    
                # Robust Decoding (Trimming input tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, ids)
                ]
                answer = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0].strip()
                
                # Store Result (preserving ground_truth)
                result_entry = query.copy() 
                result_entry["model_answer"] = answer
                results.append(result_entry)

            except Exception as e:
                print(f"[WARN] VQA Error for {query.get('question_id')}: {e}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def run_captioning(model, processor, images_list, query_map, output_file):
    """
    Runs Captioning inference in batches for efficiency.
    """
    print(f"[INFO] Running Captioning Inference -> {os.path.basename(output_file)}")
    results = []
    batches = list(batched(images_list, BATCH_SIZE))
    
    for batch in tqdm(batches, desc="Captioning"):
        batch_prompts = []
        batch_images = []
        batch_queries = []
        
        # Prepare Batch
        for img_file in batch:
            if img_file not in query_map: continue
            
            img_path = os.path.join(IMAGE_DIR, img_file)
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except: continue
                
            for query in query_map[img_file]:
                # Standard Prompt used during fine-tuning
                prompt = "Describe this image in detail." 
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
                
                batch_prompts.append(text)
                batch_images.append(pil_img)
                batch_queries.append(query)
        
        if not batch_prompts: continue
        
        try:
            # Batched Inference
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
                
            # Map results back to queries
            for q, ans in zip(batch_queries, decoded):
                result_entry = q.copy()
                result_entry["model_answer"] = ans
                results.append(result_entry)

        except Exception as e:
            print(f"[WARN] Caption Batch Error: {e}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("[INFO] Loading datasets...")
    vqa_data = load_data(VQA_EVAL_JSON)
    cap_data = load_data(CAP_EVAL_JSON)
    images = get_image_files(IMAGE_DIR)
    
    print(f"[INFO] Found {len(CHECKPOINTS)} checkpoints to evaluate.")
    
    for ckpt in CHECKPOINTS:
        # Determine output filenames
        if ckpt is None:
            name = "base_model"
        else:
            name = os.path.basename(ckpt) # e.g., 'checkpoint-200'
            
        vqa_out = os.path.join(RESULTS_DIR, f"vqa_{name}.json")
        cap_out = os.path.join(RESULTS_DIR, f"caption_{name}.json")
        
        # Check if already processed
        if os.path.exists(vqa_out) and os.path.exists(cap_out):
            print(f"[INFO] Skipping {name} (results already exist)")
            continue
            
        # Load Model
        model, processor = load_model(ckpt)
        if not model: continue
        
        # Run Inference Tasks
        if not os.path.exists(vqa_out):
            run_vqa(model, processor, images, vqa_data, vqa_out)
        else:
            print(f"[INFO] VQA results for {name} exist. Skipping VQA.")
            
        if not os.path.exists(cap_out):
            run_captioning(model, processor, images, cap_data, cap_out)
        else:
            print(f"[INFO] Caption results for {name} exist. Skipping Captioning.")
            
        # Cleanup Memory (Essential for loop)
        del model
        del processor
        torch.cuda.empty_cache()
        print("[INFO] GPU memory cleared.\n")