"""
File: convert_detection.py

Description:
Generates synthetic training data (Captions and VQA pairs) from MMRS-1M detection samples.
It samples images from SAR and IR modalities, prompts GPT-4o with image/bbox context, 
and saves the resulting natural language data.

Notes: Requires a valid OpenAI API Key.
"""

import json
import os
import random
import base64
import re
import sys
from tqdm import tqdm
from openai import OpenAI

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE" 

DET_JSON_DIR = "mmrs-1m/data/json/detection/"
IMAGE_ROOT = "mmrs-1m/"
OUTPUT_DIR = "mmrs-1m/gpt_generated_data/"

# Generation Limits
SAMPLES_PER_MODALITY = 500

# Dataset Modality Mapping
SAR_DATASETS = ['HRSID', 'SARV2'] 
IR_DATASETS = ['infrared', 'IR_']

# -----------------------------------------------------------------------------
# API Client Initialization
# -----------------------------------------------------------------------------
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
    # Attempt to load from environment variable if not hardcoded
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OpenAI API Key is missing. Please set it in the script or environment.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_modality(filename):
    """
    Determines if a dataset belongs to SAR or IR based on filename keywords.
    """
    fname = filename.lower()
    if any(k.lower() in fname for k in SAR_DATASETS): return 'sar'
    if any(k.lower() in fname for k in IR_DATASETS): return 'ir'
    return 'unknown'

def get_dataset_name(json_path):
    """
    Extracts a clean dataset name from the file path.
    """
    parent = os.path.basename(os.path.dirname(json_path))
    if parent == "detection":
        return os.path.splitext(os.path.basename(json_path))[0]
    return parent

def encode_image(image_path):
    """
    Reads an image file and encodes it to a Base64 string for API transmission.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_incremental(data, filepath):
    """
    Overwrites the specified JSON file with the current data list.
    Used to save progress after every generation step.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save {filepath}: {e}")

# -----------------------------------------------------------------------------
# Parsing Logic
# -----------------------------------------------------------------------------

def parse_mmrs_conversation(value):
    """
    Parses the internal string format of MMRS-1M annotations.
    Extracts category names and bounding box coordinates.
    """
    objects = []
    # Regex to capture content inside <p>tags</p> and [brackets]
    pattern = r"<p>(.*?)</p>\[(.*?)\]"
    matches = re.findall(pattern, value)
    
    for obj_text, box_text in matches:
        # Clean up category names (remove leading numbers like "13 ships")
        cat_clean = re.sub(r'^(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+', '', obj_text, flags=re.IGNORECASE).strip()
        
        # Parse coordinate groups
        box_groups = box_text.split(';')
        for bg in box_groups:
            if not bg.strip(): continue
            try:
                coords = [float(x) for x in bg.split(',')]
                if len(coords) == 4:
                    objects.append({'category': cat_clean, 'bbox': coords})
            except: 
                pass
    return objects

# -----------------------------------------------------------------------------
# GPT-4o Interaction Logic
# -----------------------------------------------------------------------------

def generate_synthetic_data(image_path, objects, modality):
    """
    Constructs a prompt with object context and sends the image to GPT-4o.
    Expects a JSON response containing a caption and Q&A pairs.
    """
    # 1. Summarize object context for the model (Ground Truth Injection)
    obj_summary = []
    for obj in objects:
        bbox = [round(b, 3) for b in obj['bbox']]
        obj_summary.append(f"- {obj['category']} at normalized bbox {bbox}")
    
    # Truncate if context is too long to avoid token limits
    if len(obj_summary) > 60:
        obj_summary = obj_summary[:60]
        obj_summary.append("... (and more objects)")
    
    context_str = "\n".join(obj_summary)
    
    # 2. Construct the System Prompt
    prompt = f"""
    You are an expert in satellite remote sensing imagery analysis ({modality.upper()} modality).
    
    I will provide an image and a list of ground-truth detected objects.
    Your task is to generate training data.
    
    Ground Truth Objects in this image:
    {context_str}
    
    Output a valid JSON object with exactly two keys:
    
    1. "caption": A detailed, natural language description of the image. 
       - STRICT RULE: Do NOT mention "bounding boxes", "coordinates", "bbox", or "normalized values".
       - Focus on the scene content, object spatial relationships (e.g. "clustered in the center"), and density.
       - Do not add information beyond what is mentioned in the provided data. Keep your language simple and do not describe anything beyond the objects specified.
    
    2. "qa_pairs": A list of exactly 3 Question-Answer pairs matching these types:
       - Pair 1 (Numeric): Question about counting objects. Answer must be a single number (integer). Type label: "object quantity".
       - Pair 2 (Binary): Existence question ("Is there..."). Answer must be "Yes" or "No". Type label: "object existence".
       - Pair 3 (Semantic): Question about position, category, or layout. Answer must be short (1-3 words). Type label: "object category" or "object position".
    
    The output must be raw JSON only.
    """
    
    try:
        base64_image = encode_image(image_path)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                        }
                    ]
                }
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        # Parse Response
        content = response.choices[0].message.content
        # Strip code block formatting if present
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
        
    except Exception as e:
        print(f"[ERROR] GPT Generation failed: {e}")
        return None

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Starting Synthetic Data Generation Pipeline...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Indexing: Gather valid images and objects from JSONs
    candidates = {"sar": [], "ir": []}
    
    print("[INFO] Scanning detection datasets...")
    json_files = []
    for root, dirs, files in os.walk(DET_JSON_DIR):
        for file in files:
            if file.endswith(".json") and 'total' not in file:
                json_files.append(os.path.join(root, file))
                
    for json_path in tqdm(json_files, desc="Indexing Metadata"):
        filename = os.path.basename(json_path)
        dataset_name = get_dataset_name(json_path)
        
        # Check modality
        modality = get_modality(filename) 
        if modality == 'unknown': 
            modality = get_modality(os.path.basename(os.path.dirname(json_path)))
        if modality == 'unknown': 
            continue
            
        try:
            with open(json_path, 'r') as f: 
                data = json.load(f)
        except: 
            continue
            
        for item in data:
            # Resolve Image Path
            rel_path = item.get('image') or item.get('image_path')
            if not rel_path: continue
            
            full_path = os.path.join(IMAGE_ROOT, rel_path)
            if not os.path.exists(full_path):
                # Try fallback path structure
                alt = os.path.join(IMAGE_ROOT, rel_path.replace("data/", "", 1))
                if os.path.exists(alt): 
                    full_path = alt
                else: 
                    continue
            
            # Extract Objects
            objects = []
            conversations = item.get('conversations', [])
            
            # Strategy A: Parse from GPT conversation history
            for turn in conversations:
                if turn['from'] == 'gpt':
                    objects.extend(parse_mmrs_conversation(turn['value']))
            
            # Strategy B: Parse from standard annotations list if Strategy A failed
            if not objects:
                anns = item.get('annotations', [])
                if not anns and 'bbox' in item: anns = [item]
                for ann in anns:
                    cat = ann.get('category_name') or str(ann.get('category_id', ''))
                    bbox = ann.get('bbox')
                    if cat and bbox: 
                        objects.append({'category': cat, 'bbox': bbox})
            
            if objects:
                candidates[modality].append({
                    "image_path": full_path,
                    "objects": objects,
                    "dataset_name": dataset_name,
                    "source_file": filename
                })

    # 2. Generation Loop: Sample images and query GPT
    data_store = {
        "sar": {"caption": [], "vqa": []},
        "ir": {"caption": [], "vqa": []}
    }
    
    # Initialize unique question IDs
    q_id_counters = {"sar": 1000000, "ir": 2000000}

    for mod in ['sar', 'ir']:
        pool = candidates[mod]
        print(f"\n[INFO] Found {len(pool)} candidates for {mod.upper()}.")
        
        # Random Sampling
        if len(pool) > SAMPLES_PER_MODALITY:
            selected = random.sample(pool, SAMPLES_PER_MODALITY)
        else:
            selected = pool
            
        print(f"[INFO] Generating data for {len(selected)} {mod.upper()} images...")
        
        # Define output file paths
        cap_path = os.path.join(OUTPUT_DIR, f"{mod}_gpt_caption.json")
        vqa_path = os.path.join(OUTPUT_DIR, f"{mod}_gpt_vqa.json")
        
        for item in tqdm(selected, desc=f"Generating {mod.upper()}"):
            # Call GPT-4o
            generated = generate_synthetic_data(item['image_path'], item['objects'], mod)
            
            if generated:
                # Process Caption Data
                cap_entry = {
                    "image_id": os.path.basename(item['image_path']),
                    "image_path": item['image_path'],
                    "ground_truth": generated.get('caption', ""),
                    "question": "Describe the image in detail",
                    "dataset": item['dataset_name'],
                    "question_id": q_id_counters[mod],
                    "type": "caption"
                }
                data_store[mod]["caption"].append(cap_entry)
                q_id_counters[mod] += 1
                
                # Process VQA Data (3 pairs per image)
                for qa in generated.get('qa_pairs', []):
                    vqa_entry = {
                        "image_id": os.path.basename(item['image_path']),
                        "image_path": item['image_path'],
                        "question": qa['question'],
                        "ground_truth": str(qa['answer']), 
                        "dataset": item['dataset_name'],
                        "question_id": q_id_counters[mod],
                        "type": qa.get('type', 'mixed')
                    }
                    data_store[mod]["vqa"].append(vqa_entry)
                    q_id_counters[mod] += 1
                
                # Save incrementally to prevent data loss on crash
                save_incremental(data_store[mod]["caption"], cap_path)
                save_incremental(data_store[mod]["vqa"], vqa_path)
            
        print(f"[INFO] Completed generation for {mod.upper()}.")

if __name__ == "__main__":
    main()