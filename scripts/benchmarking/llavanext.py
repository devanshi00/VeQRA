"""
File: llavanext.py

Description:
Runs inference using the LLaVA-NeXT (v1.6 Mistral) model for Visual Question Answering (VQA)
and Image Captioning. Handles 4-bit quantization and processes tasks sequentially or in batches.

Notes: Requires 'llava-hf/llava-v1.6-mistral-7b-hf'.
"""

import json
import os
import sys
import itertools
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
IMAGE_BASE_PATH = "../dataset/vrsbench/Images_val/"
VQA_EVAL_PATH = "../dataset/vrsbench/VRSBench_EVAL_vqa.json"
CAP_EVAL_PATH = "../dataset/vrsbench/VRSBench_EVAL_Cap.json"

VQA_OUTPUT_DIR = "./results/vqa/"
CAP_OUTPUT_DIR = "./results/caption/"

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def batched(iterable, n):
    """
    Splits an iterable into tuples of length n.
    Used for creating image batches for efficient inference.
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def load_model():
    """
    Initializes the LLaVA-NeXT model with 4-bit quantization (NF4).
    
    Returns:
        tuple: (model, processor) ready for inference.
    """
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    print(f"[INFO] Loading Model: {model_id} (4-bit)...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        
        # Ensure padding token is correctly set for generation
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
            
        print("[INFO] LLaVA-NeXT Model and processor loaded successfully.")
        return model, processor

    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        sys.exit(1)

def load_and_map_queries(eval_json_path):
    """
    Reads the evaluation JSON and maps questions to their corresponding image IDs.
    
    Args:
        eval_json_path (str): Path to the JSON file containing questions.
        
    Returns:
        dict: A dictionary mapping image_id to a list of query objects.
    """
    print(f"[INFO] Loading evaluation data: {os.path.basename(eval_json_path)}")
    try:
        with open(eval_json_path, 'r') as f:
            all_queries = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Eval JSON file not found at {eval_json_path}")
        raise

    queries_by_image = defaultdict(list)
    for query in all_queries:
        queries_by_image[query['image_id']].append(query)
        
    print(f"[INFO] Mapped {len(all_queries)} questions across {len(queries_by_image)} unique images.")
    return queries_by_image

def get_image_files(image_base_path):
    """
    Retrieves the list of image files from the directory.
    Trims the list to the first 1000 images for benchmarking.
    """
    print(f"[INFO] Scanning image directory: {image_base_path}")
    try:
        all_image_files = os.listdir(image_base_path)
    except FileNotFoundError:
        print(f"[ERROR] Image directory not found at {image_base_path}")
        raise

    subset_files = all_image_files[:1000]
    print(f"[INFO] Processing subset of {len(subset_files)} images.")
    return subset_files

# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------

def run_vqa_inference(model, processor):
    """
    Executes VQA inference sequentially (One-by-One).
    Uses strict token slicing to extract only the generated answer.
    """
    print("[INFO] Starting VQA Inference Task (LLaVA-NeXT)...")
    
    output_file = os.path.join(VQA_OUTPUT_DIR, "llavanext.json")
    os.makedirs(VQA_OUTPUT_DIR, exist_ok=True)
    
    all_results = []

    try:
        queries_by_image = load_and_map_queries(VQA_EVAL_PATH)
        all_image_files = get_image_files(IMAGE_BASE_PATH)
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return
        
    # Process images sequentially
    for image_file in tqdm(all_image_files, desc="VQA Inference"):
        if image_file not in queries_by_image:
            continue

        image_path = os.path.join(IMAGE_BASE_PATH, image_file)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            continue

        queries_for_this_image = queries_by_image[image_file]

        for query in queries_for_this_image:
            try:
                # Format prompt for concise VQA output
                question_text = f"{query['question']} Answer with a single word or number."
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"}, 
                            {"type": "text", "text": question_text}
                        ]
                    },
                ]

                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                
                inputs = processor(
                    text=[prompt], 
                    images=[image], 
                    return_tensors="pt", 
                    padding=True
                ).to(model.device)

                with torch.inference_mode():
                    generate_ids = model.generate(**inputs, max_new_tokens=5)

                # Extract only new tokens (slice off the input prompt)
                output_ids = generate_ids[0][len(inputs['input_ids'][0]):]
                model_answer = processor.decode(
                    output_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                ).strip()

                all_results.append({
                    "image_id": query["image_id"], 
                    "question": query["question"],
                    "question_id": query["question_id"], 
                    "dataset": query["dataset"],
                    "type": query["type"], 
                    "model_answer": model_answer,
                    "ground_truth": query["ground_truth"]
                })

            except Exception as e:
                print(f"[ERROR] Processing Q_ID {query['question_id']}: {e}")

    print(f"[INFO] VQA complete. Saving {len(all_results)} results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

def run_captioning_inference(model, processor):
    """
    Executes Captioning inference in batched mode (Batch Size = 10).
    """
    print("[INFO] Starting Captioning Inference Task (LLaVA-NeXT)...")
    
    output_file = os.path.join(CAP_OUTPUT_DIR, "llavanext.json")
    os.makedirs(CAP_OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    batch_size = 10

    try:
        queries_by_image = load_and_map_queries(CAP_EVAL_PATH)
        all_image_files = get_image_files(IMAGE_BASE_PATH)
        image_batches = list(batched(all_image_files, batch_size))
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return

    # Process batches
    for image_file_batch in tqdm(image_batches, desc="Caption Inference"):
        batch_prompts = []
        batch_images = []
        batch_queries = []

        # Prepare batch data
        for image_file in image_file_batch:
            if image_file not in queries_by_image:
                continue
            
            image_path = os.path.join(IMAGE_BASE_PATH, image_file)
            try:
                pil_image = Image.open(image_path)
            except Exception as e:
                print(f"[ERROR] Failed to load image {image_path}: {e}")
                continue
            
            for query in queries_by_image[image_file]:
                question_text = query['question']
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"}, 
                            {"type": "text", "text": question_text}
                        ]
                    }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                
                batch_prompts.append(prompt)
                batch_images.append(pil_image)
                batch_queries.append(query)

        if not batch_prompts:
            continue

        try:
            # Batch Inference
            inputs = processor(
                text=batch_prompts, 
                images=batch_images, 
                return_tensors="pt", 
                padding=True
            ).to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=200)

            # Decode results (slicing inputs from outputs for each item in batch)
            decoded_answers = []
            for i in range(len(generated_ids)):
                output_ids = generated_ids[i][len(inputs['input_ids'][i]):]
                answer = processor.decode(
                    output_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                ).strip()
                decoded_answers.append(answer)

            # Map answers back to queries
            for query, answer in zip(batch_queries, decoded_answers):
                all_results.append({
                    "image_id": query["image_id"], 
                    "question": query["question"],
                    "question_id": query["question_id"], 
                    "dataset": query["dataset"],
                    "type": query["type"], 
                    "model_answer": answer,
                    "ground_truth": query["ground_truth"]
                })

        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")

    print(f"[INFO] Captioning complete. Saving {len(all_results)} results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

def main():
    model, processor = load_model()
    if model and processor:
        run_vqa_inference(model, processor)
        run_captioning_inference(model, processor)
        print("\n[INFO] All inference tasks completed.")
    else:
        print("[FATAL] Model failed to initialize. Exiting.")

if __name__ == "__main__":
    main()