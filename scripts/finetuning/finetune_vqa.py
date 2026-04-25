"""
File: finetune_vqa.py

Description:
Fine-tunes the Qwen2.5-VL model on the VRSBench dataset for Visual Question Answering (VQA).
Uses Low-Rank Adaptation (LoRA) and restricts training to a fixed sample limit.

Notes:
Configured for a single-epoch run without intermediate evaluation or checkpoints 
to maximize throughput.
"""

import json
import os
import sys
import torch
import random
from PIL import Image
from tqdm import tqdm
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
    "output_dir": "./qwen_vrsbench_vqa_lora"
}

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SAMPLES = 16000

# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------

class VQADataset(torch.utils.data.Dataset):
    """
    Custom Dataset for loading VQA pairs from VRSBench JSON annotations.
    Limits the total number of samples to MAX_SAMPLES for efficient fine-tuning.
    """
    def __init__(self, annotation_dir, image_dir, processor, split_name="train"):
        print(f"[INFO] Scanning {split_name} VQA data in: {annotation_dir}")
        self.image_dir = image_dir
        self.eos_token = processor.tokenizer.eos_token
        self.samples = []
        
        try:
            json_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
        except FileNotFoundError:
            print(f"[FATAL] Annotation directory not found: {annotation_dir}")
            sys.exit(1)
        
        # Load samples until limit is reached
        for f in tqdm(json_files, desc=f"Loading {split_name}"):
            if len(self.samples) >= MAX_SAMPLES:
                break
                
            try:
                with open(os.path.join(annotation_dir, f), 'r') as jf:
                    data = json.load(jf)
                    img_file = data['image']
                    
                    # Verify image existence
                    if not os.path.exists(os.path.join(image_dir, img_file)):
                        continue
                        
                    for qa in data.get('qa_pairs', []):
                        if len(self.samples) >= MAX_SAMPLES: break
                        
                        self.samples.append({
                            'image_file': img_file,
                            'question': qa['question'],
                            'answer': qa['answer']
                        })
            except Exception:
                continue
                
        print(f"[INFO] Loaded {len(self.samples)} {split_name} pairs (Limited to {MAX_SAMPLES}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_dir, item['image_file'])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback: pick a random sample if the current image fails
            # This prevents the collator from receiving None and crashing batch processing
            print(f"[WARN] Failed to load image {image_path}: {e}. Picking replacement.")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        return {
            "image": image,
            "question": item['question'],
            "answer": item['answer'] + self.eos_token
        }

# -----------------------------------------------------------------------------
# Data Collator
# -----------------------------------------------------------------------------

class CustomDataCollator:
    """
    Handles tokenization and masking.
    Ensures that the model only learns to generate the answer, not the question.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: return {}

        images = [f["image"] for f in features]
        questions = [f["question"] for f in features]
        answers = [f["answer"] for f in features]
        
        prompt_texts = []
        full_texts = []
        
        for q, a in zip(questions, answers):
            # Enforce concise answering in the system/user instruction
            instruction = f"{q} Answer concisely in a single word or number."
            
            # User Prompt
            msg_prompt = [
                {"role": "user", "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": instruction}
                ]}
            ]
            
            # Full Conversation (Prompt + Answer)
            msg_full = msg_prompt + [
                {"role": "assistant", "content": [{"type": "text", "text": a}]}
            ]
            
            prompt_texts.append(self.processor.apply_chat_template(
                msg_prompt, tokenize=False, add_generation_prompt=True
            ))
            full_texts.append(self.processor.apply_chat_template(
                msg_full, tokenize=False
            ))

        # Tokenize Full Inputs
        full_inputs = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True
        )

        # Tokenize Prompt Only (for masking)
        prompt_inputs = self.processor(
            text=prompt_texts, images=images, return_tensors="pt", padding=True
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
# Main Training Loop
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Loading Model and Processor...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4"
    )
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, use_fast=True, trust_remote_code=True
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=quant_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    print("[INFO] Applying LoRA Adapters...")
    
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
    
    print("[INFO] Preparing Dataset...")
    train_dataset = VQADataset(
        PATH_CONFIG["train_ann_dir"], 
        PATH_CONFIG["train_img_dir"], 
        processor, 
        split_name="train"
    )
    
    print("[INFO] Configuring Trainer...")
    
    training_args = TrainingArguments(
        output_dir=PATH_CONFIG["output_dir"],
        num_train_epochs=1,
        per_device_train_batch_size=1, # Keep small for VRAM efficiency
        gradient_accumulation_steps=16, # Effective batch size = 16
        learning_rate=1e-4,
        warmup_steps=10,
        logging_steps=10,
        
        # Optimization: Disable evaluation and intermediate saves for speed
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        
        fp16=False, 
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CustomDataCollator(processor)
    )

    print("[INFO] Starting Training...")
    trainer.train()
    
    final_save = os.path.join(PATH_CONFIG["output_dir"], "final_vqa_checkpoint")
    print(f"[INFO] Saving Final Model to: {final_save}")
    trainer.save_model(final_save)
    print("[INFO] VQA Fine-tuning complete.")

if __name__ == "__main__":
    main()