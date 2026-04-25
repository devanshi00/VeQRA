"""
File: evaluate.py

Description:
Evaluates fine-tuned VQA and Captioning models using optimized metrics.
Implements batched BERT-BLEU (Semantic) with length penalties, 
Relative Error Decay (Numeric), and Exact Match (Binary).

Notes:
Uses batch processing for BERT embeddings to improve evaluation speed on GPUs.
"""

import json
import os
import re
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
RESULTS_DIR = "./results"
METRICS_DIR = "./metrics"

os.makedirs(METRICS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
print("[INFO] Initializing BERT model for batched semantic evaluation...")

try:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] BERT model loaded successfully on: {device}")

except Exception as e:
    print(f"[ERROR] Failed to load BERT model: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Core Metric Logic: Batched BERT-BLEU
# -----------------------------------------------------------------------------

def get_bert_embedding_batch(texts, batch_size=32):
    """
    Generates BERT embeddings for a list of texts using batch processing.
    Uses mean pooling of the last hidden state.
    
    Args:
        texts (list): List of input strings.
        batch_size (int): Batch size for inference.
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Handle empty texts safely
            processed_texts = [text if text and text.strip() else " " for text in batch_texts]

            # Tokenize batch
            inputs = tokenizer(
                processed_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference & Pooling
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

    if not all_embeddings:
        return np.array([])
        
    return np.vstack(all_embeddings)

def get_ngrams(tokens, n):
    """
    Generates all n-grams from a list of tokens.
    """
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def compute_semantic_recall_batch(reference_ngrams, candidate_ngrams, bert_batch_size=32):
    """
    Computes Semantic Recall: For every n-gram in the reference, finds the 
    best matching n-gram in the candidate via cosine similarity.
    """
    if not reference_ngrams or not candidate_ngrams:
        return 0.0

    # Batch embedding generation
    reference_embeddings = get_bert_embedding_batch(reference_ngrams, bert_batch_size)
    candidate_embeddings = get_bert_embedding_batch(candidate_ngrams, bert_batch_size)

    # Compute similarity matrix
    similarities = cosine_similarity(reference_embeddings, candidate_embeddings)

    # Max similarity for each reference n-gram (Recall perspective)
    max_similarities = similarities.max(axis=1)

    # Average recall across all reference n-grams
    return max_similarities.mean()

def calculate_bert_bleu(reference, candidate, N=4, alpha=0.5, epsilon=1e-8, bert_batch_size=32):
    """
    Calculates BERT-BLEU with Length Penalty.
    Formula: Score = Length_Penalty * Max_Recall
    Length Penalty = exp(-alpha * |len_cand - len_ref| / len_ref)
    """
    reference = str(reference or "").strip()
    candidate = str(candidate or "").strip()

    if not reference or not candidate:
        return 0.0

    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not ref_tokens or not cand_tokens:
        return 0.0

    # Adaptive N: Ensure N does not exceed sentence length
    effective_n = min(N, len(ref_tokens), len(cand_tokens))
    if effective_n < 1: 
        return 0.0
    
    # Calculate Length Penalty
    LR = len(ref_tokens)
    LC = len(cand_tokens)
    length_penalty = np.exp(-alpha * abs(LC - LR) / LR)

    recalls = []

    # Compute recall for each n-gram order
    for n in range(1, effective_n + 1):
        reference_ngrams = get_ngrams(ref_tokens, n)
        candidate_ngrams = get_ngrams(cand_tokens, n)

        if not reference_ngrams:
            recalls.append(epsilon)
            continue

        pn = compute_semantic_recall_batch(reference_ngrams, candidate_ngrams, bert_batch_size)
        recalls.append(pn + epsilon)

    # Use max recall across n-gram sizes scaled by length penalty
    max_recall = np.max(recalls)
    bert_bleu = length_penalty * max_recall

    return round(float(bert_bleu), 4)

# -----------------------------------------------------------------------------
# Core Metric Logic: VQA Specifics
# -----------------------------------------------------------------------------

def is_binary_answer(text):
    """Checks for binary/boolean answer types."""
    text = str(text or "").strip().lower()
    text = text.replace(".", "").replace(",", "").replace("!", "")
    binary_values = {
        'yes', 'no', 'true', 'false', 'correct', 'incorrect',
        'present', 'absent', 'visible', 'not visible',
        'exists', 'does not exist'
    }
    return text in binary_values

def is_numeric_answer(text):
    """Checks for numeric answer types."""
    text = str(text or "").strip()
    if re.match(r'^-?\d+\.?\d*$', text):
        return True
    if re.match(r'^-?\d+\.?\d*\s*\w*$', text):
        return True
    return False

def calculate_binary_score(ground_truth, predicted):
    """Exact match scoring."""
    gt = str(ground_truth or "").strip().lower().replace(".", "").replace(",", "")
    pred = str(predicted or "").strip().lower().replace(".", "").replace(",", "")
    return 1.0 if gt == pred else 0.0

def extract_number(text):
    """Extracts the first number found in text."""
    text = str(text or "").strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except:
            return None
    return None

def calculate_numeric_score(ground_truth, predicted, alpha=23):
    """
    Calculates numeric score based on Relative Error.
    Formula: exp( -alpha * (|pred - gt| / gt) )
    """
    gt_num = extract_number(ground_truth)
    pred_num = extract_number(predicted)

    if gt_num is None or pred_num is None:
        return 0.0

    gt_num = abs(gt_num)
    pred_num = abs(pred_num)
    
    # Handle zero-division edge case
    if gt_num == 0:
        return 1.0 if pred_num == 0 else 0.0

    # Relative difference calculation
    diff = abs(pred_num - gt_num)
    diff = diff / gt_num
    score = np.exp(-diff * alpha)

    return round(float(score), 4)

def detect_answer_type(ground_truth):
    """Classifies answer type for metric routing."""
    gt = str(ground_truth or "").strip()
    if not gt: return 'semantic'
    if is_binary_answer(gt): return 'binary'
    if is_numeric_answer(gt): return 'numeric'
    return 'semantic'

# -----------------------------------------------------------------------------
# File Processing Runners
# -----------------------------------------------------------------------------

def evaluate_vqa_file(input_path, output_path):
    """
    Evaluates a VQA file. Supports both standard JSON lists and JSONL formats.
    """
    print(f"[INFO] Evaluating VQA: {os.path.basename(input_path)}")
    try:
        with open(input_path, 'r') as f:
            try:
                # Try loading as standard JSON array
                results = json.load(f)
            except json.JSONDecodeError:
                # Fallback to JSONL (JSON Lines)
                f.seek(0)
                results = [json.loads(line) for line in f if line.strip()]
                
    except Exception as e:
        print(f"[ERROR] Reading {input_path}: {e}")
        return
    
    stats = {
        "binary": {"count": 0, "sum": 0.0},
        "numeric": {"count": 0, "sum": 0.0},
        "bert_bleu": {"count": 0, "sum": 0.0}
    }
    
    results_with_scores = []

    for item in tqdm(results, desc="Scoring VQA"):
        # Normalize key names across different dataset versions
        gt = item.get('ground_truth') or item.get('gt_caption') or item.get('reference')
        pred = item.get('model_answer') or item.get('prediction') or item.get('candidate')
        
        ans_type = detect_answer_type(gt)
        
        score = 0.0
        metric_used = ""
        
        if ans_type == 'binary':
            score = calculate_binary_score(gt, pred)
            stats['binary']['count'] += 1
            stats['binary']['sum'] += score
            metric_used = "binary"
        elif ans_type == 'numeric':
            score = calculate_numeric_score(gt, pred)
            stats['numeric']['count'] += 1
            stats['numeric']['sum'] += score
            metric_used = "numeric"
        else:  # semantic
            score = calculate_bert_bleu(gt, pred, N=4, alpha=0.5, bert_batch_size=32)
            stats['bert_bleu']['count'] += 1
            stats['bert_bleu']['sum'] += score
            metric_used = "bert_bleu"
            
        item['score'] = score
        item['answer_type'] = ans_type
        item['metric_used'] = metric_used
        results_with_scores.append(item)

    # Calculate category averages
    avg_scores = {}
    for key, val in stats.items():
        avg_scores[key] = val['sum'] / val['count'] if val['count'] > 0 else 0.0

    # Calculate Overall Weighted Average
    total_count = len(results_with_scores)
    overall_avg = 0.0
    
    if total_count > 0:
        weight_binary = 0.10
        weight_numeric = 0.20
        weight_bert_bleu = 0.20 
        total_weight = weight_binary + weight_numeric + weight_bert_bleu
        
        if total_weight > 0:
            norm_bin = weight_binary / total_weight
            norm_num = weight_numeric / total_weight
            norm_sem = weight_bert_bleu / total_weight
            
            overall_avg = (avg_scores['binary'] * norm_bin) + \
                          (avg_scores['numeric'] * norm_num) + \
                          (avg_scores['bert_bleu'] * norm_sem)

    # Prepare and Save Output
    output_data = {
        "summary": {
            "overall_weighted_score": round(overall_avg, 4),
            "total_samples": total_count,
            "metrics_breakdown": {
                "binary": {
                    "average": round(avg_scores['binary'], 4), 
                    "count": stats['binary']['count']
                },
                "numeric": {
                    "average": round(avg_scores['numeric'], 4), 
                    "count": stats['numeric']['count']
                },
                "semantic": {
                    "average": round(avg_scores['bert_bleu'], 4), 
                    "count": stats['bert_bleu']['count']
                }
            }
        },
        "results": results_with_scores
    }
    
    with open(output_path, 'w') as f: 
        json.dump(output_data, f, indent=2)
        
    print(f"   -> Overall: {overall_avg:.4f}")
    print(f"   -> Bin: {avg_scores['binary']:.4f} | "
          f"Num: {avg_scores['numeric']:.4f} | "
          f"Sem: {avg_scores['bert_bleu']:.4f}")

def evaluate_caption_file(input_path, output_path):
    """
    Evaluates a Captioning file using BERT-BLEU. Supports JSON and JSONL.
    """
    print(f"[INFO] Evaluating Caption: {os.path.basename(input_path)}")
    try:
        with open(input_path, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                results = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"[ERROR] Reading {input_path}: {e}")
        return

    scores = []
    results_with_scores = []
    
    for item in tqdm(results, desc="Scoring Caption"):
        gt = item.get('ground_truth') or item.get('gt_caption') or item.get('reference')
        pred = item.get('model_answer') or item.get('prediction') or item.get('candidate')
        
        score = calculate_bert_bleu(gt, pred, N=4, alpha=0.5, bert_batch_size=32)
        scores.append(score)
        
        item['bert_bleu'] = score
        results_with_scores.append(item)
        
    avg_score = sum(scores) / len(scores) if scores else 0
    
    output_data = {
        "summary": {
            "mean_bert_bleu": avg_score,
            "total_samples": len(scores)
        },
        "results": results_with_scores
    }
    
    with open(output_path, 'w') as f: 
        json.dump(output_data, f, indent=2)
        
    print(f"   -> Mean BERT-BLEU: {avg_score:.4f}")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        print(f"[FATAL] Results directory not found at {RESULTS_DIR}")
        sys.exit(1)

    try:
        files = os.listdir(RESULTS_DIR)
    except OSError as e:
        print(f"[FATAL] Unable to list files in results directory: {e}")
        sys.exit(1)
    
    # 1. Process VQA Files
    vqa_files = [f for f in files if f.startswith("vqa_") and f.endswith(".json")]
    if vqa_files:
        print(f"\n[INFO] Found {len(vqa_files)} VQA files.")
        for f in sorted(vqa_files):
            in_path = os.path.join(RESULTS_DIR, f)
            out_path = os.path.join(METRICS_DIR, f.replace(".json", "_metrics.json"))
            
            if not os.path.exists(out_path):
                evaluate_vqa_file(in_path, out_path)
            else:
                print(f"[INFO] Skipping {f} (Metrics file already exists)")
    else:
        print("\n[INFO] No VQA files found.")

    # 2. Process Caption Files
    cap_files = [f for f in files if f.startswith("caption_") and f.endswith(".json")]
    if cap_files:
        print(f"\n[INFO] Found {len(cap_files)} Caption files.")
        for f in sorted(cap_files):
            in_path = os.path.join(RESULTS_DIR, f)
            out_path = os.path.join(METRICS_DIR, f.replace(".json", "_metrics.json"))
            
            if not os.path.exists(out_path):
                evaluate_caption_file(in_path, out_path)
            else:
                print(f"[INFO] Skipping {f} (Metrics file already exists)")
    else:
        print("\n[INFO] No Caption files found.")
            
    print("\n[INFO] All evaluations complete.")