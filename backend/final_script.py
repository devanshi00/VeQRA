from collections import defaultdict
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import PeftModel
from io import BytesIO
from qwen_vl_utils import process_vision_info
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
import requests
import re
import logging
import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from io import BytesIO
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA

# SETUP FastAPI
app = FastAPI()

#  SETUP LOGGING (FILE + CONSOLE)
LOG_FILE = "./pipeline_debug.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("===== PIPELINE STARTED =====")

DOTA_CLASSES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer ball field",
    14: "swimming pool"
}
DIOR_CLASSES = {
    0: "airplane",
    1: "airport",
    2: "baseballfield",
    3: "basketballcourt",
    4: "bridge",
    5: "chimney",
    6: "dam",
    7: "Expressway-Service-area",
    8: "Expressway-toll-station",
    9: "golffield",
    10: "groundtrackfield",
    11: "harbor",
    12: "overpass",
    13: "ship",
    14: "stadium",
    15: "storagetank",
    16: "tenniscourt",
    17: "trainstation",
    18: "vehicle",
    19: "windmill"
}


# CONFIG
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CAPTION_ADAPTER_PATH = "./adapters/caption_adapter"
VQA_ADAPTER_PATH     = "./adapters/vqa_adapter"
# Optional: if images are local instead of URL
IMAGE_DIR = "./uploads/"  
# To store grounded outputs
SAV_DIR= "./results/" 

# MODEL LOADING

def load_multi_adapter_model():
    """
    Load Qwen2.5-VL with caption + VQA adapters.
    4-bit quantization with bitsandbytes.
    """
    logger.info("Loading base model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        use_fast=True,
        trust_remote_code=True
    )
    logger.info("Processor loaded.")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("Base model loaded in 4-bit.")

    logger.info(f"Loading caption adapter: {CAPTION_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(
        base_model,
        CAPTION_ADAPTER_PATH,
        adapter_name="caption"
    )

    logger.info(f"Loading VQA adapter: {VQA_ADAPTER_PATH}")
    model.load_adapter(
        VQA_ADAPTER_PATH,
        adapter_name="vqa"
    )

    model.set_adapter("caption")
    model.eval()
    logger.info("Model + adapters loaded successfully.")

    return model, processor


# IMAGE LOADING
def local_variance(img, window=7):
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    return np.mean(mean_sq - mean**2)

def detect_image_type(img_path, sar_thresh=2500):
    img = np.array(Image.open(img_path))

    # CASE 1 — Single-band grayscale
    if img.ndim == 2:
        lv = local_variance(img)
        return "SAR" if lv > sar_thresh else "IR"

    # CASE 2 — RGBA → RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if img.shape[2] != 3:
        return f"Unsupported channels: {img.shape[2]}"

    gray = img.mean(axis=2)
    lv = local_variance(gray)

    # CASE 3 — SAR disguised as RGB (channels identical)
    if np.all(img[:,:,0] == img[:,:,1]) and np.all(img[:,:,1] == img[:,:,2]):
        return "SAR" if lv > sar_thresh else "RGB"

    # PCA
    X = img.reshape(-1, 3).astype(np.float32)
    pca = PCA(n_components=3).fit(X)
    pc1, pc2, pc3 = pca.explained_variance_ratio_

    r = X[:,0]
    g = X[:,1]
    b = X[:,2]

    # STRICT IR conditions
    mean_abs_rg = np.mean(np.abs(r - g))
    mean_abs_rb = np.mean(np.abs(r - b))
    mean_abs_gb = np.mean(np.abs(g - b))

    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)

    ir_condition = (
        pc1 > 0.999 and pc2 < 0.001 and
        mean_abs_rg < 1 and mean_abs_rb < 1 and mean_abs_gb < 1 and
        abs(var_r - var_g) < 1e-3 and abs(var_r - var_b) < 1e-3
    )

    if ir_condition:
        return "IR"

    # SAR via strong texture
    if lv > sar_thresh:
        return "SAR"

    # Default → RGB
    return "RGB"

def load_image(image_id, image_url=None):
    logger.debug(f"Loading image: {image_id}")

    local_path = os.path.join(IMAGE_DIR, image_id)

    if os.path.exists(local_path):
        logger.debug(f"Loaded from local: {local_path}")
        return Image.open(local_path).convert("RGB")

    if image_url:
        logger.debug(f"Downloading image from {image_url}")
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    logger.error(f"IMAGE NOT FOUND: {image_id}")
    raise FileNotFoundError(f"Could not load {image_id}")



# YOLO HELPERS
def extract_pred_components(pred):
    """
    Returns polys (Nx4x2) and class_ids (N,) from a YOLO OBB prediction.
    Fast and avoids re-parsing inside postprocess.
    """
    obb = pred[0].obb
    if obb is None or obb.cls is None or len(obb.cls) == 0:
        return None, None

    polys = obb.xyxyxyxy.cpu().numpy()
    cls_ids = obb.cls.cpu().numpy()
    return polys, cls_ids

def merge_dota_dior_preds(pred_dota, pred_dior, dota_classes, dior_classes):
    """
    Merges DOTA and DIOR predictions with class-based filtering.
    """

    ALLOWED_DIOR = {
        "airport",
        "chimney",
        "dam",
        "golffield",
        "overpass",
        "stadium",
        "Expressway-Service-area",
        "Expressway-toll-station",
        "trainstation",
        "windmill",
        "ship"   
    }

    polys_dota, cls_dota = extract_pred_components(pred_dota)
    polys_dior, cls_dior = extract_pred_components(pred_dior)

    if polys_dota is None:
        polys_dota = np.zeros((0,4,2))
        cls_dota = np.zeros((0,))
    if polys_dior is None:
        polys_dior = np.zeros((0,4,2))
        cls_dior = np.zeros((0,))

    keep_idx = [
        i for i, cid in enumerate(cls_dota)
        if dota_classes.get(int(cid), "") != "ship"
    ]
    polys_dota = polys_dota[keep_idx] if len(keep_idx) else np.zeros((0,4,2))
    cls_dota   = cls_dota[keep_idx]   if len(keep_idx) else np.zeros((0,))

    # Track DOTA sources (all remaining are from DOTA)
    dota_source = ["DOTA"] * len(cls_dota)

    # Classes remaining after removing ship
    dota_present_classes = {
        dota_classes.get(int(cid), "unknown")
        for cid in cls_dota
    }
    filtered_dior_polys = []
    filtered_dior_cls = []
    filtered_dior_source = []  # NEW

    for poly, cid in zip(polys_dior, cls_dior):
        cname = dior_classes.get(int(cid), "unknown")

        if cname == "ship":
            # Always take DIOR ship
            filtered_dior_polys.append(poly)
            filtered_dior_cls.append(cid)
            filtered_dior_source.append("DIOR")
            continue

        if cname in ALLOWED_DIOR and cname not in dota_present_classes:
            filtered_dior_polys.append(poly)
            filtered_dior_cls.append(cid)
            filtered_dior_source.append("DIOR")

    # Convert to arrays
    if len(filtered_dior_polys) > 0:
        filtered_dior_polys = np.stack(filtered_dior_polys)
        filtered_dior_cls = np.array(filtered_dior_cls)
    else:
        filtered_dior_polys = np.zeros((0,4,2))
        filtered_dior_cls = np.zeros((0,))

    merged_polys = np.concatenate([polys_dota, filtered_dior_polys], axis=0)
    merged_cls   = np.concatenate([cls_dota,  filtered_dior_cls], axis=0)

    merged_source = np.array(dota_source + filtered_dior_source)

    logger.debug("Merged DOTA + DIOR count: %d", len(merged_polys))

    # Return also source array
    return merged_polys, merged_cls, merged_source

def postprocess_from_arrays(
    polys, cls_ids, source, width, height,
    dota_classes, dior_classes,
    include_region=False
):
    """
    Class mapping is now driven STRICTLY by source:
        - If source[idx] == "DOTA": use DOTA classes
        - If source[idx] == "DIOR": use DIOR classes
    """

    all_lines = []

    for idx, (poly, cid) in enumerate(zip(polys, cls_ids), start=1):
        cid = int(cid)
        if source[idx - 1] == "DOTA":
            class_name = dota_classes.get(cid, "unknown")
        else:
            class_name = dior_classes.get(cid, "unknown")

        # Flatten polygon (convert 4x2 → 8 numbers)
        pixel_coords = [float(v) for p in poly for v in p]
        xs = pixel_coords[0::2]
        ys = pixel_coords[1::2]

        # Normalized center
        x_center = (sum(xs) / 4) / width
        y_center = (sum(ys) / 4) / height

        if include_region:

            # Normalize 8 coords
            norm_coords = [
                (pixel_coords[j] / width) if j % 2 == 0 else (pixel_coords[j] / height)
                for j in range(len(pixel_coords))
            ]

            # Determine region of image
            x1 = width / 3
            x2 = 2 * width / 3
            y1 = height / 3
            y2 = 2 * height / 3

            px_center = sum(xs) / 4
            py_center = sum(ys) / 4

            col = "left" if px_center < x1 else ("center" if px_center < x2 else "right")
            row = "top"  if py_center < y1 else ("middle" if py_center < y2 else "bottom")

            region = "center" if (row == "middle" and col == "center") else f"{row}-{col}"

            coords_str = " ".join(f"{c:.6f}" for c in norm_coords)
            line = f"{class_name} {coords_str} {region}"
        else:
            line = f"{idx}. {class_name} (x={x_center:.6f}, y={y_center:.6f})"

        all_lines.append(line)

    return "\n".join(all_lines)

def draw_obb_boxes(
        parsed_items=None,
        save_path=None,
        img_bgr=None
):
    """
    Draw OBBs where coords are NORMALIZED (0-1).
    Converts them to pixel coordinates automatically.

    """
    if img_bgr is None:
        raise ValueError("img_bgr must be provided")

    img = img_bgr.copy()
    h, w = img.shape[:2]     # image height & width

    for item in parsed_items:
        class_name = item["class"]
        coords = item["coords"]      # normalized coords
        region = item["region"]

        # Convert normalized → pixel coordinates
        coords_np = np.array(coords, dtype=np.float32).reshape(4, 2)
        pts = np.zeros_like(coords_np)
        pts[:, 0] = coords_np[:, 0] * w    # x * width
        pts[:, 1] = coords_np[:, 1] * h    # y * height

        pts = pts.astype(int)

        # Draw polygon
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        # Draw label
        label = None
        cv2.putText(
            img,
            label,
            (pts[0][0], pts[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save if requested
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(save_path, img)

    return img


# VLM HELPERS
def parse_multiple_output_lines(output_text):
    """
    Parses multi-line VLM output.
    Handles both cases:
        (1) class + 8 coords + region
        (2) class + 8 coords
    Returns list of dicts: [{"class":..., "coords":[...], "region":...}, ...]
    """
    results = []

    for line in output_text.strip().split("\n"):
        parts = line.split()
        if len(parts) < 9:
            # Need at least class + 8 numbers
            continue

        # Extract numeric tokens (coords)
        numeric_tokens = []
        for p in parts:
            try:
                numeric_tokens.append(float(p))
            except ValueError:
                pass

        if len(numeric_tokens) < 8:
            # Not enough coords; skip this line
            continue

        coords = numeric_tokens[-8:]  # take last 8 numeric tokens

        # Detect region (only if last token is not numeric)
        last_token = parts[-1]
        try:
            float(last_token)
            region = None  
        except ValueError:
            region = last_token  

        #Extract class name
        
        tokens_remaining = []

        nums_to_remove = set(coords)

        for p in parts:
            # remove region token if exists
            if region is not None and p == region:
                continue
            # remove numeric coords
            try:
                val = float(p)
                if val in nums_to_remove:
                    nums_to_remove.remove(val)
                    continue
            except ValueError:
                pass
            tokens_remaining.append(p)

        class_name = " ".join(tokens_remaining).strip()

        results.append({
            "class": class_name,
            "coords": coords,
            "region": region
        })

    return results

def extract_count_from_response(response):
    """
    Extract numeric count from VLM response.
    """
    # 1. Look for explicit digits
    numbers = re.findall(r'\d+', response)
    if numbers:
        return int(numbers[0])
    
    # 2. Look for number words
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12
    }
    
    response_lower = response.lower()
    for word, num in word_to_num.items():
        if word in response_lower:
            return num
    
    return None

def obb_area(coords, height, width, pixel_resolution):
    """
    Compute real-world area of an oriented bounding box (OBB).

    coords: list/array of 8 numbers
            [x1,y1, x2,y2, x3,y3, x4,y4]  (pixel coordinates)
    pixel_resolution: size of one pixel (e.g., meters per pixel)

    Returns: area in (pixel_resolution^2) units (e.g., m^2)
    """
    pts = np.array(coords, dtype=float).reshape(4, 2)

    # Shoelace polygon area
    x = pts[:, 0]
    y = pts[:, 1]
    area_px = 0.5 * np.abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )

    # Convert to actual area
    real_area= area_px * (pixel_resolution ** 2) * height * width

    return round(float(real_area), 6)

# INFERENCE HELPERS
def pick_adapter(image_type):
    """
    image_type: 'SAR', 'IR', 'RGB'
    Returns the adapter name to be used before generation.
    """
    if image_type == "SAR":
        return "sar"
    elif image_type == "IR":
        return "ir"
    else:
        return "rgb"     # RGB → use each function's default adapter

def generate_with_adapter(model, processor, adapter_name, image,
                          system_prompt, user_prompt, max_new_tokens=128,
                          img_type=None):

    if img_type in ["SAR", "IR"]:
        auto_adapter = pick_adapter(img_type)  # sar or ir
        logger.debug(f"Auto-selecting adapter: {auto_adapter} for image type: {img_type}")
        model.set_adapter(auto_adapter)
    else:
        # keep existing behaviour
        if adapter_name is None:
            model.disable_adapter()
        else:
            model.set_adapter(adapter_name)
            logger.debug(f"Using adapter: {adapter_name}")

    messages = [
        {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt}
        ]
        },   
        {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": user_prompt}
        ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    out_ids = generated_ids[0][len(inputs.input_ids[0]):]
    answer = processor.decode(
        out_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()
    logger.debug(f"VLM output: {answer}")

    return answer


def answer_caption(model, processor, image, instruction: str, img_type) -> str:
    """
    Use caption adapter to generate a detailed caption.
    """
    system_prompt = "You are an expert vision-language model.\n"
    user_prompt = instruction

    caption = generate_with_adapter(
        model, processor,
        adapter_name="caption",
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=256,
        img_type=img_type
    )
    words = caption.split()
    if "from GoogleEarth" in " ".join(words[:7]):
        caption = caption.replace("from GoogleEarth", "").strip()
        # also remove extra double spaces if any
        caption = " ".join(caption.split())

    return caption

def answer_grounding(model, processor, image, instruction: str, merged_polys, merged_cls, merged_source, output_path, img_type):
    """
    Ultra-low-latency grounding with fallback:
    - If YOLO finds no boxes → return [] and save original image.
    - If VLM returns no valid boxes → return [] and save original image.
    """

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    yolo_output = postprocess_from_arrays(
        merged_polys,
        merged_cls,
        merged_source,
        width=w,
        height=h,
        dota_classes=DOTA_CLASSES,
        dior_classes=DIOR_CLASSES,
        include_region=True          # grounding requires region + 8coords
    )

    if yolo_output.strip() == "":
        save_path = os.path.join(SAV_DIR, output_path)
        cv2.imwrite(save_path, img_bgr)   # Save original image
        return []  # No grounding possible

    system_prompt = """You are an expert vision-language model.
                        Input:
                        1) A natural-language query about objects in an image.
                        2) YOLO detections, each as a single line of 8-coordinate rotated OBB.

                        Rules:
                        1. If the query mentions ANY positional/attribute cues (left/right/top/bottom,
                        color, shape, size, near, far): return EXACTLY ONE best match.
                        2. If no such cues exist: return ALL boxes.

                        Output ONLY the original YOLO lines.
                        """

    user_prompt = f"""YOLO bounding boxes:
                        {yolo_output}

                        User query: '{instruction}'.
                        Return ONLY the bounding-box line(s)."""

    out_line = generate_with_adapter(
        model,
        processor,
        adapter_name="vqa",
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=1024,
        img_type=img_type
    )


    parsed = parse_multiple_output_lines(out_line)
    logger.debug(f"Parsed Grounding list:\n{parsed}")

    if len(parsed) == 0:
        save_path = os.path.join(SAV_DIR, output_path)
        cv2.imwrite(save_path, img_bgr)   # Save original image
        return []  # No valid grounding results

    vis = draw_obb_boxes(
        parsed_items=parsed,
        img_bgr=img_bgr,
        save_path=os.path.join(SAV_DIR, output_path)
    )

    response_list = []
    for idx, det in enumerate(parsed, start=1):
        response_list.append({
            "object-id": str(idx),
            "obbox": det["coords"]
        })

    return response_list

def answer_vqa_binary(model, processor, image, instruction: str, img_type) -> str:
    """
    Use VQA adapter for yes/no questions.
    """
    system_prompt = "You are an expert vision-language model.\n"
    user_prompt = f"{instruction} Answer only 'Yes' or 'No'."
    ans = generate_with_adapter(
        model, processor,
        adapter_name="vqa",
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=8,
        img_type=img_type
    )
    # Small cleanup to make the value neat
    ans_clean = ans.strip().split()[0]
    if ans_clean.lower().startswith("y"):
        return "Yes"
    if ans_clean.lower().startswith("n"):
        return "No"
    return ans_clean

def answer_vqa_area(model, processor, image, instruction: str, merged_polys, merged_cls, merged_source, spatial_resolution, img_type) -> float:
    """
    Function for area calculation
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    yolo_output = postprocess_from_arrays(
        merged_polys,
        merged_cls,
        merged_source,
        width=w,
        height=h,
        dota_classes=DOTA_CLASSES,
        dior_classes=DIOR_CLASSES,
        include_region=True          # grounding requires region + 8coords
    )

    if yolo_output.strip() == "":
        return 0.0  # No grounding possible

    system_prompt = """You are an expert vision-language model.
                        Input:
                        1) A natural-language query about objects in an image.
                        2) YOLO detections, each as a single line of 8-coordinate rotated OBB.

                        Rules:
                        1. If the query mentions ANY positional/attribute cues (left/right/top/bottom,
                        color, shape, size, near, far): return EXACTLY ONE best match.
                       
                        2.  Output ONLY the original YOLO line which is EXACTLY ONE best match the object mentioned in the query.
                        """

    user_prompt = f"""YOLO bounding boxes:
                        {yolo_output}

                        User query: '{instruction}'.
                        Return ONLY the bounding-box line."""

    out_line = generate_with_adapter(
        model,
        processor,
        adapter_name="vqa",
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=512,
        img_type=img_type
    )
    parsed = parse_multiple_output_lines(out_line)
    logger.debug(f"Parsed Grounding list:\n{parsed}")

    if len(parsed) == 0:
        return 0.0 # No valid grounding results

    # Only one entry expected in parsed
    coords = parsed[0]["coords"]

    # Compute area
    area = obb_area(coords, h, w, spatial_resolution)

    return area

def answer_vqa_numeric(model, processor, image, instruction: str, merged_polys, merged_cls, merged_source, img_type):
    """
    Use VQA adapter for numeric questions.
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    # Postprocess: normalized center only
    yolo_output = postprocess_from_arrays(
        merged_polys,
        merged_cls,
        merged_source,
        width=w,
        height=h,
        dota_classes=DOTA_CLASSES,
        dior_classes=DIOR_CLASSES,
        include_region=False      # counting works on center coords only
    )
    logger.debug(f"YOLO POSTPROCESS OUTPUT NUMERIC:\n{yolo_output}")
    if yolo_output.strip() == "":
        logger.info("No detections → running direct VLM counting.")
        system_prompt ="""
        You are a visual reasoning assistant. Count objects in the image based on the user query.
        Rules:
        1. Identify all objects relevant to the query (use synonyms).
        2. For each object, verify visual attributes from its pixels (e.g., color).
        3. For color attributes, verify the dominant visible color (allow natural color variations).
        5. Only count objects satisfying all attributes AND spatial constraints.
        6. Output: the count (integer only)
        7. If none match, output 0.
        Be precise, do not guess, rely on visible evidence.      
        """
        user_prompt = f"Your answer:"
        return generate_with_adapter( model, processor, adapter_name="vqa", image=image, system_prompt=system_prompt, user_prompt=user_prompt, max_new_tokens=8, img_type=img_type)
    # Build VLM prompt
    system_prompt = f"""You are a filtering assistant. Your task is to count objects from a PROVIDED LIST ONLY.

                        COORDINATE SYSTEM:
                        - x ranges from 0.0 (left edge) to 1.0 (right edge)
                        - y ranges from 0.0 (top edge) to 1.0 (bottom edge)

                        DETECTED OBJECTS LIST (from YOLO):
                        {yolo_output}

                        USER QUESTION: "{instruction}"

                        CRITICAL INSTRUCTIONS:
                        1. You MUST ONLY count objects from the "DETECTED OBJECTS LIST" above.
                        2. DO NOT look at the image to find new objects or count visually by yourself.
                        3. The image is provided ONLY to help you verify attributes (like color, orientation) of objects already in the list.
                        4. Your process:
                        Step 1: Identify which CLASS(es) from the list are relevant to the question (e.g., if asked about "aeroplanes", find all "plane" or "aeroplane" entries)
                        Step 2: For each relevant object, use the image at its coordinates to verify any visual attributes mentioned (e.g., "yellow", "parked", "flying")
                        Step 3: Apply any spatial filters mentioned (e.g., "in top half" means y < 0.5)
                        Step 4: For queries specifying colour attribute, look at the image to verify the coorect answer from yolos detections.
                        Step 4: Count ONLY the objects from the list that match ALL criteria
                        5. If no objects in the list match the question, the answer is 0.
                        6. Respond with ONLY THE NUMBER - nothing else.
                        """
    user_prompt = f"Your answer:"
    ans = generate_with_adapter(
        model, processor,
        adapter_name="vqa",
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=8,
        img_type=img_type
    )
    logger.debug(f"Raw numeric VLM answer: {ans}")
    final_count = extract_count_from_response(ans)
    logger.debug(f"Extracted numeric: {final_count}")

    fc_str = str(final_count).strip()

    token = fc_str.split()[0].replace(",", "")
    try:
        return float(token)
    except:
        return final_count  # fallback: return raw int or original output

def answer_vqa_semantic(model, processor, image, instruction: str, img_type) -> str:
    """
    Use VQA adapter for short semantic attributes (colors, materials, etc.).
    """
    system_prompt = "You are an expert vision-language model.\n"
    user_prompt = f"{instruction} Answer concisely in 1-3 words."
    ans = generate_with_adapter(
        model, processor,
        adapter_name="vqa",
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=8,
        img_type=img_type
    )
    return ans.strip()

# MAIN PIPELINE

# Load model once at startup
logger.info("Loading model...")
model, processor = load_multi_adapter_model()
logger.info("Model Loaded!!")

# Load YOLO models once at startup
logger.info("Running YOLO inference...")
yolo_dota = YOLO("./yolo11x-obb.pt")
yolo_dior = YOLO("./dior_only_last.pt")

# API ENDPOINTS DEFINITIONS
class UploadRequest(BaseModel):
    image_id: str
    image_url: str
    
class CaptionRequest(BaseModel):
    image_id: str
    image_url: str
    cap_instr: str
    img_type: str
    
class GroundingRequest(BaseModel):
    image_id: str
    image_url: str
    grounding_instr: str
    merged_polys: List[Any]
    merged_cls: List[Any]
    merged_source: List[Any]
    img_type: str
    output_path: str
    
class BinaryRequest(BaseModel):
    image_id: str
    image_url: str
    img_type: str
    bin_instr: str

class NumericRequest(BaseModel):
    image_id: str
    image_url: str
    num_instr: str
    img_type: str
    merged_polys: List[Any]
    merged_cls: List[Any]
    merged_source: List[Any]
    spatial_resolution: float

class SemanticRequest(BaseModel):
    image_id: str
    image_url: str
    img_type: str
    sem_instr: str

# Utility to convert numpy arrays to lists
def to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

# API ROUTES

# YOLO UPLOAD + MERGE
@app.post('/upload')
async def upload(data: UploadRequest):
    image = load_image(data.image_id, image_url=data.image_url)
    img_type = detect_image_type(os.path.join(IMAGE_DIR, data.image_id))
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
    pred_dota = yolo_dota.predict(img_bgr, verbose=False)
    pred_dior = yolo_dior.predict(img_bgr, verbose=False)
    logger.info("YOLO inference done.")

    merged_polys, merged_cls, merged_source = merge_dota_dior_preds(
        pred_dota, pred_dior,
        dota_classes=DOTA_CLASSES,
        dior_classes=DIOR_CLASSES
    )  
    
    return {
        "img_type": img_type,
        "merged_polys": to_list(merged_polys),
        "merged_cls": to_list(merged_cls),
        "merged_source": to_list(merged_source)
    }

@app.post('/caption')
async def upload(data: CaptionRequest):
    image = load_image(data.image_id, image_url=data.image_url)
    return {"caption": answer_caption(model, processor, image, data.cap_instr, data.img_type)}
    
@app.post('/grounding')
async def upload(data: GroundingRequest):
    image = load_image(data.image_id, image_url=data.image_url)
    return {"ground": answer_grounding(model, processor, image, data.grounding_instr, data.merged_polys, data.merged_cls, data.merged_source, data.output_path, data.img_type)}
    
@app.post('/binary')
async def upload(data: BinaryRequest):
    image = load_image(data.image_id, image_url=data.image_url)
    return {"answer": answer_vqa_binary(model, processor, image, data.bin_instr, data.img_type)}
    
@app.post('/numeric_evaluate')
async def upload(data: NumericRequest):
    image = load_image(data.image_id, image_url=data.image_url)
    if "area" in data.num_instr.lower():
        num_resp = answer_vqa_area(
            model, processor, image,
            data.num_instr, data.merged_polys, data.merged_cls, data.merged_source,
            data.spatial_resolution,
            data.img_type
        )
    else:
        num_resp = answer_vqa_numeric(
            model, processor, image,
            data.num_instr, data.merged_polys, data.merged_cls, data.merged_source, data.img_type
        )
    return {"answer": num_resp}

@app.post('/numeric_chat')
async def upload(data: NumericRequest):
    image = load_image(data.image_id, image_url=data.image_url)
    num_resp = answer_vqa_numeric(model, processor, image, data.num_instr, data.merged_polys, data.merged_cls, data.merged_source, data.img_type)
    return {"answer": num_resp}
    
@app.post('/semantic')
async def upload(data: SemanticRequest):
    image = load_image(data.image_id, image_url=data.image_url)
    return {"answer": answer_vqa_semantic(model, processor, image, data.sem_instr, data.img_type)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
