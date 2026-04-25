"""
File: convert_grounding.py

Description:
Processes MMRS-1M detection JSONs to generate a standardized grounding dataset.
It parses embedded text annotations, converts 4-point bounding boxes to 
normalized 8-point polygon coordinates, and segregates data by modality (SAR/IR).

Notes: Handles specific MMRS-1M string formatting (e.g., "<p>class</p>[coords]").
"""

import json
import os
import re
from tqdm import tqdm
from collections import defaultdict

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DET_JSON_DIR = "./mmrs-1m/data/json/detection/"
IMAGE_ROOT = "./mmrs-1m/"
OUTPUT_DIR = "./mmrs-1m/grounding_data/"

# Heuristics for dataset mapping
SAR_DATASETS = ['HRSID', 'SARV2'] 
IR_DATASETS = ['infrared', 'IR_']

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_modality(filename):
    """
    Determines if a dataset file belongs to SAR or IR based on filename keywords.
    """
    fname = filename.lower()
    if any(k.lower() in fname for k in SAR_DATASETS): return 'sar'
    if any(k.lower() in fname for k in IR_DATASETS): return 'ir'
    return 'unknown'

def convert_4pt_to_8pt(bbox):
    """
    Converts a standard 4-point bbox [x1, y1, x2, y2] into a normalized 
    8-point polygon representation [x1,y1, x2,y1, x2,y2, x1,y2].
    
    The order is Clockwise starting from Top-Left:
    TL -> TR -> BR -> BL
    
    Returns:
        list[float]: 8 coordinates clamped between 0.0 and 1.0.
    """
    x1, y1, x2, y2 = bbox
    
    # Define 8 points (Clockwise from Top-Left for a horizontal box)
    points = [x1, y1, x2, y1, x2, y2, x1, y2]
    
    # Ensure all points are floats and clamped strictly between 0.0 and 1.0
    normalized_8pt = [max(0.0, min(1.0, float(p))) for p in points]
        
    return normalized_8pt

def parse_mmrs_and_convert(value):
    """
    Parses the specific MMRS-1M string format.
    Input Format: "There are <p>13 ships</p>[0.75,0.69,0.76,0.71;...]..."
    
    Returns:
        list[dict]: A list of objects containing cleaned category names and 8-point boxes.
    """
    objects = []
    # Pattern matches: <p>Category</p>[box_string]
    pattern = r"<p>(.*?)</p>\[(.*?)\]"
    matches = re.findall(pattern, value)
    
    for obj_text, box_text in matches:
        # Clean category name (remove counts like "13 ships" -> "ships")
        cat_clean = re.sub(r'^(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+', '', obj_text, flags=re.IGNORECASE).strip()
        
        # Singularize simple cases
        if cat_clean.endswith('s') and len(cat_clean) > 1: 
            cat_clean = cat_clean[:-1]
        
        # Parse coordinate groups separated by semicolons
        box_groups = box_text.split(';')
        for bg in box_groups:
            if not bg.strip(): continue
            try:
                coords = [float(x) for x in bg.split(',')]
                if len(coords) == 4:
                    # Convert standard box to 8-point polygon format
                    pts_8 = convert_4pt_to_8pt(coords)
                    objects.append({'category': cat_clean, 'bbox_8pt': pts_8})
            except: 
                pass
                
    return objects

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Starting Grounding Data Conversion...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_store = {"sar": [], "ir": []}
    
    # 1. Recursive scan for JSON files
    print("[INFO] Scanning for detection JSON files...")
    json_files = []
    for root, _, files in os.walk(DET_JSON_DIR):
        for file in files:
            if file.endswith(".json") and 'total' not in file:
                json_files.append(os.path.join(root, file))
                
    print(f"[INFO] Discovered {len(json_files)} detection dataset files.")
    
    # 2. Process each dataset file
    for json_path in json_files:
        filename = os.path.basename(json_path)
        parent = os.path.basename(os.path.dirname(json_path))
        
        # Determine modality from filename or parent folder
        modality = get_modality(filename) 
        if modality == 'unknown': modality = get_modality(parent)
        if modality == 'unknown': continue
            
        print(f"[INFO] Processing: {filename} [{modality.upper()}]")
        
        try:
            with open(json_path, 'r') as f: 
                raw_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {filename}: {e}")
            continue
            
        # Iterate through samples in the JSON
        for item in tqdm(raw_data, leave=False, desc=f"Parsing {filename}"):
            # Resolve Image Path
            rel_path = item.get('image') or item.get('image_path')
            if not rel_path: continue
            
            # Construct absolute path and verify existence
            full_path = os.path.join(IMAGE_ROOT, rel_path)
            if not os.path.exists(full_path):
                # Try alternative path resolution (stripping 'data/' prefix)
                alt = os.path.join(IMAGE_ROOT, rel_path.replace("data/", "", 1))
                if os.path.exists(alt): 
                    full_path = alt
                else: 
                    continue
            
            # Extract and convert objects from conversation history
            objects = []
            conversations = item.get('conversations', [])
            for turn in conversations:
                if turn['from'] == 'gpt':
                    objects.extend(parse_mmrs_and_convert(turn['value']))
            
            if not objects: continue
            
            # Group boxes by category to form the prompt/answer pair
            cat_map = defaultdict(list)
            for obj in objects:
                cat_map[obj['category']].append(obj['bbox_8pt'])
            
            # Create a data sample for each category found in the image
            for cat, boxes in cat_map.items():
                # Format: List of 8-point coordinates
                box_strings = [str([round(x, 3) for x in b]) for b in boxes]
                answer_str = "; ".join(box_strings)
                
                data_store[modality].append({
                    "image": full_path,
                    "prompt": f"Detect all {cat}s in the image.",
                    "answer": answer_str,
                    "dataset": filename
                })

    # 3. Save processed data to disk
    for mod in ['sar', 'ir']:
        out_file = os.path.join(OUTPUT_DIR, f"{mod}_grounding_8pt.json")
        with open(out_file, 'w') as f:
            json.dump(data_store[mod], f, indent=2)
        print(f"[INFO] {mod.upper()} Generation Complete: Saved {len(data_store[mod])} samples to {out_file}")

    print("[INFO] All conversion tasks finished.")

if __name__ == "__main__":
    main()