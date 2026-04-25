"""
File: skysense_test.py

Description:
Specific inference script for the SkySense dataset (Captioning & VQA).
Loads a fine-tuned Qwen2.5-VL model, runs inference on CSV-based datasets,
and computes BERT-BLEU scores on-the-fly using a secondary BERT model.

Notes: Requires 'bert-base-uncased' for metrics and Qwen2.5-VL for generation.
"""

import json
import os
import sys
import random
import itertools
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModel
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info 

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
SKYSENSE_ROOT = "../dataset/skysense/"
CAPTION_CSV = os.path.join(SKYSENSE_ROOT, "skysense_caption.csv")
VQA_CSV = os.path.join(SKYSENSE_ROOT, "skysense_vqa.csv")
CAPTION_IMG_DIR = os.path.join(SKYSENSE_ROOT, "images_caption/")
VQA_IMG_DIR = os.path.join(SKYSENSE_ROOT, "images_vqa/")

# Model Configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Path to the hybrid adapter (VRSBench + SkySense)
CHECKPOINT_PATH = "./qwen_hybrid_caption_lora/final_hybrid_checkpoint"
INSTRUCTION_PROMPT = "Describe this image in detail."

# Inference Settings
OUTPUT_DIR = "skysense_results_hybrid"
BATCH_SIZE = 10
SAMPLE_LIMIT = 1000 

# -----------------------------------------------------------------------------
# Metric Initialization (BERT-BLEU)
# -----------------------------------------------------------------------------
print("[INFO] Initializing BERT model for metric calculation...")
try:
    BERT_NAME = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    bert_model = AutoModel.from_pretrained(BERT_NAME)
    bert_model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
except Exception as e:
    print(f"[FATAL] Failed to load BERT: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_bert_embedding(text):
    """
    Generates mean-pooled BERT embeddings for metric computation.
    """
    if not text or not text.strip(): 
        return np.zeros(768)
        
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def calculate_bert_bleu(reference, candidate, N=4):
    """
    Calculates BERT-BLEU score.
    Combines N-gram precision with Semantic Similarity (Cosine of BERT embeddings).
    """
    ref = str(reference or "").strip().lower().split()
    cand = str(candidate or "").strip().lower().split()
    
    if not ref or not cand: return 0.0
    
    # Adaptive N: Prevent empty n-grams if sentences are short
    effective_n = min(N, len(ref), len(cand))
    if effective_n < 1: return 0.0

    def get_ngrams(tokens, n):
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    precisions = []
    for n in range(1, effective_n + 1):
        cand_ngrams = get_ngrams(cand, n)
        ref_ngrams = get_ngrams(ref, n)
        
        if not cand_ngrams or not ref_ngrams:
            precisions.append(1e-8)
            continue
        
        # Compute embeddings and similarity matrix
        c_emb = np.array([get_bert_embedding(x) for x in cand_ngrams])
        r_emb = np.array([get_bert_embedding(x) for x in ref_ngrams])
        sim = cosine_similarity(c_emb, r_emb)
        
        # Max similarity (Precision)
        precisions.append(max(sim.max(axis=1).mean(), 1e-8))
        
    # Geometric Mean
    log_sum = sum(np.log(p) for p in precisions)
    return round(float(np.exp(log_sum / effective_n)), 4)

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

def load_skysense_caption_data():
    """
    Loads SkySense captioning data from CSV.
    Handles random sampling if dataset exceeds SAMPLE_LIMIT.
    """
    print(f"[INFO] Loading Caption Data: {os.path.basename(CAPTION_CSV)}")
    df = pd.read_csv(CAPTION_CSV)
    df.columns = df.columns.str.strip()
    
    data = []
    # Dynamic column name resolution
    img_col = 'image' if 'image' in df.columns else df.columns[0]
    cap_col = 'caption' if 'caption' in df.columns else df.columns[1]

    for _, row in df.iterrows():
        data.append({
            'image_id': str(row[img_col]),
            'ground_truth': str(row[cap_col]),
            'image_path': os.path.join(CAPTION_IMG_DIR, str(row[img_col]))
        })
        
    # Apply limit
    if len(data) > SAMPLE_LIMIT:
        random.seed(42)
        data = random.sample(data, SAMPLE_LIMIT)
        
    print(f"[INFO] Loaded {len(data)} caption samples.")
    return data

def load_skysense_vqa_data():
    """
    Loads SkySense VQA data from CSV.
    Groups questions by image to optimize loading during inference.
    """
    print(f"[INFO] Loading VQA Data: {os.path.basename(VQA_CSV)}")
    df = pd.read_csv(VQA_CSV)
    df.columns = df.columns.str.strip()
    
    grouped_data = defaultdict(list)
    
    # Dynamic column name resolution
    img_col = 'image' if 'image' in df.columns else df.columns[0]
    q_col = 'query' if 'query' in df.columns else df.columns[1]
    ans_col = 'response' if 'response' in df.columns else df.columns[2]

    # Sample based on Unique Images to keep context together
    unique_images = df[img_col].unique()
    if len(unique_images) > SAMPLE_LIMIT:
        random.seed(42)
        selected_images = set(random.sample(list(unique_images), SAMPLE_LIMIT))
    else:
        selected_images = set(unique_images)

    count = 0
    for _, row in df.iterrows():
        img_id = str(row[img_col])
        if img_id not in selected_images: continue
        
        grouped_data[img_id].append({
            'image_id': img_id,
            'question': str(row[q_col]),
            'ground_truth': str(row[ans_col]),
            'image_path': os.path.join(VQA_IMG_DIR, img_id)
        })
        count += 1
        
    print(f"[INFO] Loaded {count} VQA questions across {len(grouped_data)} images.")
    return grouped_data

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

def load_qwen_model():
    """
    Loads Qwen2.5-VL with 4-bit quantization and attaches the LoRA adapter.
    """
    print(f"[INFO] Loading Model (Adapter: {os.path.basename(CHECKPOINT_PATH)})...")
    
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
        
        if os.path.exists(CHECKPOINT_PATH):
            model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
            print("[INFO] Adapter attached successfully.")
        else:
            print(f"[WARN] Checkpoint not found at {CHECKPOINT_PATH}. Using Base Model.")
            
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
    Runs Captioning inference in batches.
    Computes BERT-BLEU scores immediately after generation.
    """
    print("[INFO] Starting Captioning Task (Batched)...")
    results = []
    scores = []
    
    batches = list(batched(data, BATCH_SIZE))
    
    for batch in tqdm(batches, desc="Captioning"):
        batch_prompts = []
        batch_images = []
        valid_items = []
        
        # Prepare Batch
        for item in batch:
            try:
                img = Image.open(item['image_path']).convert("RGB")
                
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": img}, 
                            {"type": "text", "text": INSTRUCTION_PROMPT}
                        ]
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                batch_prompts.append(text)
                batch_images.append(img)
                valid_items.append(item)
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
                
            # Decode and Score
            for i, item in enumerate(valid_items):
                out_ids = ids[i][len(inputs['input_ids'][i]):]
                pred = processor.decode(out_ids, skip_special_tokens=True).strip()
                
                # Compute Metric
                score = calculate_bert_bleu(item['ground_truth'], pred)
                scores.append(score)
                
                results.append({
                    **item,
                    "model_answer": pred,
                    "bert_bleu": score
                })
        except Exception as e:
            print(f"[WARN] Batch Error: {e}")
            
    avg = sum(scores)/len(scores) if scores else 0
    print(f"   -> Captioning Mean BERT-BLEU: {avg:.4f}")
    return results, avg

def run_vqa(model, processor, grouped_data):
    """
    Runs VQA inference sequentially (One-by-One).
    Computes BERT-BLEU scores immediately after generation.
    """
    print("[INFO] Starting VQA Task (Sequential)...")
    results = []
    scores = []
    
    # Iterate over images (keys of the group dict)
    all_img_ids = list(grouped_data.keys())
    
    for img_id in tqdm(all_img_ids, desc="VQA"):
        items = grouped_data[img_id]
        if not items: continue
        
        # Load image once per group
        try:
            image = Image.open(items[0]['image_path']).convert("RGB")
        except: 
            continue
            
        for item in items:
            try:
                prompt = f"{item['question']} Answer with a single word or number."
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": image}, 
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Process inputs (requires qwen_vl_utils for VQA specific formatting)
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
                
                # Generate
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=10)
                    
                out_ids = ids[0][len(inputs['input_ids'][0]):]
                pred = processor.decode(out_ids, skip_special_tokens=True).strip()
                
                # Compute Metric
                score = calculate_bert_bleu(item['ground_truth'], pred)
                scores.append(score)
                
                results.append({
                    **item,
                    "model_answer": pred,
                    "bert_bleu": score
                })
            except Exception as e:
                print(f"[WARN] VQA Error: {e}")

    avg = sum(scores)/len(scores) if scores else 0
    print(f"   -> VQA Mean BERT-BLEU: {avg:.4f}")
    return results, avg

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    cap_data = load_skysense_caption_data()
    vqa_data = load_skysense_vqa_data()
    
    # 2. Load Model
    model, processor = load_qwen_model()
    
    # 3. Run Tasks
    cap_results, cap_score = run_captioning(model, processor, cap_data)
    vqa_results, vqa_score = run_vqa(model, processor, vqa_data)
    
    # 4. Save Results
    print(f"[INFO] Saving results to {OUTPUT_DIR}...")
    
    with open(os.path.join(OUTPUT_DIR, "caption_results.json"), 'w') as f:
        json.dump(
            {"summary": {"mean_score": cap_score}, "results": cap_results}, 
            f, indent=2
        )
        
    with open(os.path.join(OUTPUT_DIR, "vqa_results.json"), 'w') as f:
        json.dump(
            {"summary": {"mean_score": vqa_score}, "results": vqa_results}, 
            f, indent=2
        )
        
    print("[INFO] SkySense Testing Complete.")