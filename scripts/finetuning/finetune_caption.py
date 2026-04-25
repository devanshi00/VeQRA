"""
File: finetune_caption.py

Description:
Fine-tunes the Qwen2.5-VL model on the VRSBench dataset for Image Captioning
using Low-Rank Adaptation (LoRA) and 4-bit quantization.

Notes:
Implementation uses a custom data collator to handle chat templates and label masking
to ensure the model only calculates loss on the assistant's response.
"""

import json
import os
import sys
import torch
from PIL import Image
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
    "train_ann_dir": "../dataset/vrsbench/Annotations_train/",
    "train_img_dir": "../dataset/vrsbench/Images_train/",
    "val_ann_dir": "../dataset/vrsbench/Annotations_val/",
    "val_img_dir": "../dataset/vrsbench/Images_val/",
    "output_dir": "../arnav/fine_tuning/qwen_vrsbench_caption_lora"
}

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTRUCTION_PROMPT = "Describe this image in detail."

# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------

class CaptionDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for loading raw images and caption annotations.
    Tokenization is deferred to the Data Collator to handle padding dynamically.
    """
    def __init__(self, annotation_dir, image_dir, processor):
        print(f"[INFO] Scanning annotation directory: {annotation_dir}")
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.eos_token = processor.tokenizer.eos_token
        
        try:
            self.annotations = [
                f for f in os.listdir(annotation_dir) 
                if f.endswith('.json')
            ]
            if not self.annotations:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"[FATAL] No .json files found in {annotation_dir}")
            sys.exit(1)
        
        print(f"[INFO] Found {len(self.annotations)} annotation files.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieves image and caption. Returns None if data loading fails,
        which must be handled by the collator.
        """
        json_filename = self.annotations[idx]
        annotation_path = os.path.join(self.annotation_dir, json_filename)
        
        try:
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            image_file = annotation['image']
            caption = annotation['caption']
            
            image_path = os.path.join(self.image_dir, image_file)
            image = Image.open(image_path).convert("RGB")

        except Exception as e:
            print(f"[WARN] Error loading data for {json_filename}. Error: {e}. Skipping.")
            return None 
            
        return {
            "image": image,
            "caption": caption + self.eos_token
        }

# -----------------------------------------------------------------------------
# Data Collator Implementation
# -----------------------------------------------------------------------------

class CustomDataCollator:
    """
    Handles tokenization, padding, and label masking for the batch.
    Ensures that loss is not calculated on the user instruction tokens.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Filter out failed loads (None) from __getitem__
        features = [f for f in features if f is not None]
        if not features:
            return {}

        images = [f["image"] for f in features]
        captions = [f["caption"] for f in features]
        
        prompt_texts = []
        full_texts = []
        
        # Construct Chat Templates
        for cap in captions:
            # User Prompt (Instruction + Image)
            msg_prompt = [
                {"role": "user", "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": INSTRUCTION_PROMPT}
                ]}
            ]
            # Full Conversation (Prompt + Assistant Response)
            msg_full = msg_prompt + [
                {"role": "assistant", "content": [{"type": "text", "text": cap}]}
            ]
            
            # Apply template: 'prompt_texts' is just the instruction
            # 'full_texts' includes the target caption
            prompt_texts.append(self.processor.apply_chat_template(
                msg_prompt, tokenize=False, add_generation_prompt=True
            ))
            full_texts.append(self.processor.apply_chat_template(
                msg_full, tokenize=False
            ))

        # Tokenize Full Conversation (Inputs + Targets)
        full_inputs = self.processor(
            text=full_texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )

        # Tokenize Prompt Only (to determine where to mask labels)
        prompt_inputs = self.processor(
            text=prompt_texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        
        # Prepare Labels: Mask out the prompt tokens
        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        
        for i in range(len(labels)):
            # Set labels to -100 (ignored by loss function) for the prompt duration
            labels[i, :prompt_lengths[i]] = -100
        
        batch = full_inputs
        batch["labels"] = labels
        
        return batch

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Loading Model and Processor...")
    
    # 4-bit Quantization Configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True,
        trust_remote_code=True
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    print("[INFO] Configuring LoRA Adapters...")
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Initializing Datasets...")
    
    train_dataset = CaptionDataset(
        annotation_dir=PATH_CONFIG["train_ann_dir"],
        image_dir=PATH_CONFIG["train_img_dir"],
        processor=processor
    )
    
    val_dataset = CaptionDataset(
        annotation_dir=PATH_CONFIG["val_ann_dir"],
        image_dir=PATH_CONFIG["val_img_dir"],
        processor=processor
    )
    
    data_collator = CustomDataCollator(processor)

    print("[INFO] Setting Training Arguments...")
    
    training_args = TrainingArguments(
        output_dir=PATH_CONFIG["output_dir"],
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
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

    print("[INFO] Initializing Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    print("[INFO] Starting Fine-Tuning...")
    
    trainer.train()
    
    print("[INFO] Saving Final Model Checkpoint...")
    
    final_save_path = os.path.join(PATH_CONFIG["output_dir"], "final_checkpoint")
    trainer.save_model(final_save_path)
    print(f"[INFO] Fine-tuning complete. LoRA adapter saved to: {final_save_path}")

if __name__ == "__main__":
    main()