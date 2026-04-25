"""
File: finetune_hybrid.py

Description:
Fine-tunes Qwen2.5-VL on a hybrid dataset consisting of VRSBench (JSON-based) 
and SkySense (CSV-based) for Image Captioning.

Notes:
Implements a balanced sampling strategy to ensure equal representation of both datasets 
during training. Uses QLoRA (4-bit) for memory efficiency.
"""

import json
import os
import sys
import random
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
PATH_CONFIG = {
    # VRSBench Paths
    "vrs_train_ann": "../dataset/vrsbench/Annotations_train/",
    "vrs_train_img": "../dataset/vrsbench/Images_train/",
    "vrs_val_ann": "../dataset/vrsbench/Annotations_val/",
    "vrs_val_img": "../dataset/vrsbench/Images_val/",
    
    # SkySense Paths
    "sky_csv": "../dataset/skysense/skysense_caption.csv",
    "sky_img_dir": "../dataset/skysense/images_caption/",
    
    # Output
    "output_dir": "./qwen_hybrid_caption_lora"
}

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTRUCTION_PROMPT = "Describe this image in detail."

# -----------------------------------------------------------------------------
# Data Loading Utilities
# -----------------------------------------------------------------------------

def load_vrsbench_data(ann_dir, img_dir):
    """
    Reads VRSBench individual JSON annotation files into a list.
    Verifies image existence before adding to the dataset.
    """
    print(f"[INFO] Loading VRSBench data from: {ann_dir}")
    data_items = []
    missing_count = 0
    
    try:
        files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        for f in files:
            with open(os.path.join(ann_dir, f), 'r') as json_file:
                item = json.load(json_file)
                img_path = os.path.join(img_dir, item['image'])
                
                # Check existence upfront to prevent runtime errors later
                if os.path.exists(img_path):
                    data_items.append({
                        'image_path': img_path,
                        'caption': item['caption'],
                        'source': 'vrsbench'
                    })
                else:
                    missing_count += 1
    except Exception as e:
        print(f"[ERROR] Loading VRSBench: {e}")
        return []
        
    if missing_count > 0:
        print(f"[WARN] Skipped {missing_count} missing images in VRSBench.")
        
    return data_items

def load_skysense_data(csv_path, img_dir):
    """
    Reads SkySense data from a CSV file.
    Expects columns: 'image' and 'caption'.
    """
    print(f"[INFO] Loading SkySense data from: {csv_path}")
    data_items = []
    missing_count = 0
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        if 'image' not in df.columns or 'caption' not in df.columns:
            print(f"[FATAL] SkySense CSV missing required columns. Found: {df.columns}")
            sys.exit(1)

        for _, row in df.iterrows():
            img_name = str(row['image'])
            # Handle potential absolute paths or prefixes in CSV
            if img_name.startswith("/"): 
                img_name = os.path.basename(img_name)
                
            img_path = os.path.join(img_dir, img_name)
            
            # Verify existence
            if os.path.exists(img_path):
                data_items.append({
                    'image_path': img_path,
                    'caption': str(row['caption']),
                    'source': 'skysense'
                })
            else:
                missing_count += 1

    except Exception as e:
        print(f"[ERROR] Loading SkySense: {e}")
        return []
        
    if missing_count > 0:
        print(f"[WARN] Skipped {missing_count} missing images in SkySense.")
        
    return data_items

def prepare_datasets():
    """
    Orchestrates the data loading and balancing process.
    1. Loads both datasets.
    2. Splits SkySense into Train/Val.
    3. Balances the training set (undersamples the larger dataset).
    4. Creates a fixed validation set of 1000 images (500 from each).
    """
    # 1. Load Raw Data
    vrs_train_all = load_vrsbench_data(PATH_CONFIG['vrs_train_ann'], PATH_CONFIG['vrs_train_img'])
    vrs_val_all = load_vrsbench_data(PATH_CONFIG['vrs_val_ann'], PATH_CONFIG['vrs_val_img'])
    
    sky_all = load_skysense_data(PATH_CONFIG['sky_csv'], PATH_CONFIG['sky_img_dir'])
    
    if len(sky_all) == 0 or len(vrs_train_all) == 0:
        print("[FATAL] One of the datasets is empty. Please check paths and data integrity.")
        sys.exit(1)
    
    # 2. Split SkySense (90/10 split)
    sky_train, sky_val = train_test_split(sky_all, test_size=0.1, random_state=42)
    
    print(f"\n[INFO] Data Counts (Valid):")
    print(f"   - VRSBench: Train={len(vrs_train_all)}, Val={len(vrs_val_all)}")
    print(f"   - SkySense: Train={len(sky_train)}, Val={len(sky_val)}")
    
    # 3. Create Balanced Training Set
    # We take the size of the smaller dataset to balance the training distribution
    min_train_len = min(len(vrs_train_all), len(sky_train))
    
    random.seed(42)
    vrs_train_balanced = random.sample(vrs_train_all, min_train_len)
    sky_train_balanced = random.sample(sky_train, min_train_len)
    
    final_train = vrs_train_balanced + sky_train_balanced
    random.shuffle(final_train)
    
    print(f"\n[INFO] Balanced Training Set Created:")
    print(f"   - VRSBench samples: {len(vrs_train_balanced)}")
    print(f"   - SkySense samples: {len(sky_train_balanced)}")
    print(f"   - Total Training:   {len(final_train)}")
    
    # 4. Create Balanced Validation Set (Max 1000 items)
    vrs_val_count = min(500, len(vrs_val_all))
    sky_val_count = min(500, len(sky_val))
    
    vrs_val_subset = random.sample(vrs_val_all, vrs_val_count)
    sky_val_subset = random.sample(sky_val, sky_val_count)
    
    final_val = vrs_val_subset + sky_val_subset
    random.shuffle(final_val)
    
    print(f"[INFO] Validation Set Created: {len(final_val)} samples.")
    
    return final_train, final_val

# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------

class HybridDataset(torch.utils.data.Dataset):
    """
    Standard PyTorch Dataset for the hybrid data list.
    Includes a fallback mechanism to handle corrupt images during training.
    """
    def __init__(self, data_list, processor):
        self.data = data_list
        self.processor = processor
        self.eos_token = processor.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        caption = item['caption']
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback Mechanism:
            # If an image is corrupt, log a warning and pick a random replacement 
            # to prevent the batch collator from crashing.
            print(f"[WARN] Corrupt image found: {image_path} ({e}). Picking random replacement.")
            new_idx = random.randint(0, len(self.data)-1)
            return self.__getitem__(new_idx)

        return {
            "image": image,
            "caption": caption + self.eos_token
        }

# -----------------------------------------------------------------------------
# Data Collator
# -----------------------------------------------------------------------------

class CustomDataCollator:
    """
    Handles tokenization, padding, and label masking for Qwen2.5-VL.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: return {}

        images = [f["image"] for f in features]
        captions = [f["caption"] for f in features]
        
        prompt_texts = []
        full_texts = []
        
        for cap in captions:
            # Construct User Prompt
            msg_prompt = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION_PROMPT}]}
            ]
            # Construct Full Conversation (User + Assistant)
            msg_full = msg_prompt + [
                {"role": "assistant", "content": [{"type": "text", "text": cap}]}
            ]
            
            prompt_texts.append(self.processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(msg_full, tokenize=False))

        # Tokenize full conversation (Input IDs)
        full_inputs = self.processor(
            text=full_texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        
        # Tokenize prompt only (to get lengths for masking)
        prompt_inputs = self.processor(
            text=prompt_texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        
        # Create Labels: Mask prompt tokens with -100
        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
        
        batch = full_inputs
        batch["labels"] = labels
        return batch

# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Starting Hybrid Fine-Tuning Pipeline...")
    
    # 1. Prepare Data
    train_data_list, val_data_list = prepare_datasets()
    
    # 2. Load Model & Processor
    print("\n[INFO] Loading Model and Processor...")
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

    # 3. Apply LoRA
    print("\n[INFO] Applying LoRA Adapters...")
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Initialize Datasets
    print("\n[INFO] initializing Datasets and Collator...")
    train_dataset = HybridDataset(train_data_list, processor)
    val_dataset = HybridDataset(val_data_list, processor)
    data_collator = CustomDataCollator(processor)

    # 5. Training Configuration
    print("\n[INFO] Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=PATH_CONFIG["output_dir"],
        num_train_epochs=1,
        per_device_train_batch_size=1, # Kept at 1 to minimize VRAM usage
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, # Effective batch size = 16
        learning_rate=1e-4,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False, 
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # 6. Start Training
    print("[INFO] Starting Training...")
    trainer.train()
    
    # 7. Save Final Model
    final_save = os.path.join(PATH_CONFIG["output_dir"], "final_hybrid_checkpoint")
    trainer.save_model(final_save)
    print(f"[INFO] Hybrid fine-tuning complete. Model saved to: {final_save}")

if __name__ == "__main__":
    main()