"""
File: evaluate.py

Description:
Evaluates VQA and Captioning model outputs against ground truth data.
Computes Binary, Numeric, and Semantic (BERT-BLEU) scores based on answer type.

Notes: Requires 'bert-base-uncased' and a CUDA-capable device is recommended for performance.
"""

import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
VQA_RESULTS_PATH = "./results/vqa/"
CAP_RESULTS_PATH = "./results/caption/"
VQA_METRICS_PATH = "./metrics/vqa/"
CAP_METRICS_PATH = "./metrics/caption/"

os.makedirs(VQA_METRICS_PATH, exist_ok=True)
os.makedirs(CAP_METRICS_PATH, exist_ok=True)

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
# We initialize the BERT model globally to avoid reloading it for every metric call.
print("[INFO] Loading BERT model for semantic evaluation...")

try:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Model loaded successfully on: {device}")

except Exception as e:
    print(f"[ERROR] Failed to load BERT model. Exception: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Core Evaluation Utilities
# -----------------------------------------------------------------------------

def get_bert_embedding(text):
    """
    Generates the mean-pooled BERT embedding for a given input text.
    
    Args:
        text (str): The input string to embed.
        
    Returns:
        np.array: A 768-dimensional numpy array representing the text.
    """
    if not text or not text.strip():
        return np.zeros(768)

    # Tokenize and run inference
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over the sequence dimension
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def calculate_bert_bleu(reference, candidate, N=4):
    """
    Calculates an Adaptive BERT-BLEU score suitable for short VQA answers.
    It compares n-gram embeddings of the candidate against the reference.

    Args:
        reference (str): The ground truth text.
        candidate (str): The model's predicted text.
        N (int): Maximum n-gram size to consider.

    Returns:
        float: The calculated similarity score.
    """
    ref = str(reference or "").strip().lower().split()
    cand = str(candidate or "").strip().lower().split()

    # Edge case: Empty strings
    if not ref or not cand:
        return 0.0

    # Adapt N if the sentences are shorter than the default N
    effective_n = min(N, len(ref), len(cand))
    if effective_n < 1:
        return 0.0

    def get_ngrams(tokens, n):
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    precisions = []

    # Compute similarity for each n-gram size up to effective_n
    for n in range(1, effective_n + 1):
        cand_ngrams = get_ngrams(cand, n)
        ref_ngrams = get_ngrams(ref, n)

        if not cand_ngrams or not ref_ngrams:
            precisions.append(1e-8)
            continue
        
        # Batch embedding generation for n-grams
        c_emb = np.array([get_bert_embedding(x) for x in cand_ngrams])
        r_emb = np.array([get_bert_embedding(x) for x in ref_ngrams])
        
        # Compute cosine similarity matrix
        sim = cosine_similarity(c_emb, r_emb)
        
        # Take the maximum similarity for each candidate n-gram (precision-like)
        precisions.append(max(sim.max(axis=1).mean(), 1e-8))

    # Geometric mean of precisions (standard BLEU formulation)
    log_sum = sum(np.log(p) for p in precisions)
    bert_bleu = np.exp(log_sum / effective_n)

    return round(float(bert_bleu), 4)

# -----------------------------------------------------------------------------
# Answer Type Detection & Scoring
# -----------------------------------------------------------------------------

def is_binary_answer(text):
    """Check if the answer belongs to a closed set of binary/boolean options."""
    text = str(text or "").strip().lower().replace(".", "").replace(",", "").replace("!", "")
    binary_values = {
        'yes', 'no', 'true', 'false', 'correct', 'incorrect',
        'present', 'absent', 'visible', 'not visible'
    }
    return text in binary_values

def is_numeric_answer(text):
    """Check if the answer is primarily numeric using regex."""
    text = str(text or "").strip()
    # Matches simple numbers (e.g., "-5", "3.14")
    if re.match(r'^-?\d+\.?\d*$', text):
        return True
    # Matches numbers followed by units (e.g., "5 kg")
    if re.match(r'^-?\d+\.?\d*\s*\w*$', text):
        return True
    return False

def calculate_binary_score(gt, pred):
    """Exact match scoring for binary answers."""
    gt = str(gt).strip().lower().replace(".", "").replace(",", "")
    pred = str(pred).strip().lower().replace(".", "").replace(",", "")
    return 1.0 if gt == pred else 0.0

def extract_number(text):
    """Helper to parse the first float value found in text."""
    numbers = re.findall(r'-?\d+\.?\d*', str(text))
    return float(numbers[0]) if numbers else None

def calculate_numeric_score(gt, pred):
    """Scores numeric answers based on proximity (exponential decay of difference)."""
    gt_num = extract_number(gt)
    pred_num = extract_number(pred)
    
    if gt_num is None or pred_num is None:
        return 0.0
    
    return round(float(np.exp(-abs(pred_num - gt_num))), 4)

def detect_answer_type(ground_truth):
    """
    Categorizes the ground truth into 'binary', 'numeric', or 'semantic'.
    This determines which metric is applied during evaluation.
    """
    gt = str(ground_truth or "").strip()
    
    if not gt:
        return 'semantic'
    if is_binary_answer(gt):
        return 'binary'
    if is_numeric_answer(gt):
        return 'numeric'
    
    return 'semantic'

def evaluate_vqa_result(result):
    """
    Routes a single VQA result to the correct scoring function based on its type.
    """
    gt = result.get("ground_truth")
    pred = result.get("model_answer")
    ans_type = detect_answer_type(gt)
    
    scores = {}
    
    if ans_type == 'binary':
        scores["binary_score"] = calculate_binary_score(gt, pred)
        scores["metric_used"] = "binary"
    elif ans_type == 'numeric':
        scores["numeric_score"] = calculate_numeric_score(gt, pred)
        scores["metric_used"] = "numeric"
    else: 
        scores["bert_bleu_score"] = calculate_bert_bleu(gt, pred)
        scores["metric_used"] = "bert_bleu"
        
    return scores

# -----------------------------------------------------------------------------
# Batch Execution Logic
# -----------------------------------------------------------------------------

def run_full_vqa_evaluation(input_file, output_file):
    """
    Processes a full VQA result file, applies weighted scoring, and saves metrics.
    Weights: Binary (0.1), Numeric (0.2), Semantic (0.2).
    """
    print(f"[INFO] Evaluating VQA Dataset: {os.path.basename(input_file)}")
    
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)

        stats = {
            "binary": {"count": 0, "sum": 0.0},
            "numeric": {"count": 0, "sum": 0.0},
            "bert_bleu": {"count": 0, "sum": 0.0}
        }

        results_with_scores = []
        
        # Iterate and score each item
        for item in tqdm(results, desc="Scoring VQA"):
            scores = evaluate_vqa_result(item)
            item.update(scores)
            results_with_scores.append(item)
            
            # Aggregate stats
            metric = scores["metric_used"]
            if metric == "binary":
                stats['binary']['count'] += 1
                stats['binary']['sum'] += scores['binary_score']
            elif metric == "numeric":
                stats['numeric']['count'] += 1
                stats['numeric']['sum'] += scores['numeric_score']
            else:
                stats['bert_bleu']['count'] += 1
                stats['bert_bleu']['sum'] += scores['bert_bleu_score']

        # Calculate averages per category
        avg_scores = {}
        for key, val in stats.items():
            avg_scores[key] = val['sum'] / val['count'] if val['count'] > 0 else 0.0

        # Calculate overall weighted score
        total = len(results)
        overall_avg = 0.0
        
        if total > 0:
            w_bin, w_num, w_sem = 0.1, 0.2, 0.2
            total_w = w_bin + w_num + w_sem
            
            if total_w > 0:
                norm_bin = w_bin / total_w
                norm_num = w_num / total_w
                norm_sem = w_sem / total_w
                
                overall_avg = (avg_scores['binary'] * norm_bin) + \
                              (avg_scores['numeric'] * norm_num) + \
                              (avg_scores['bert_bleu'] * norm_sem)

        # Prepare output structure
        output_data = {
            "summary": {
                "overall_weighted_score": round(overall_avg, 4),
                "total_samples": total,
                "breakdown": {
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

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(
            f"   -> Overall: {overall_avg:.4f} "
            f"[Bin: {avg_scores['binary']:.2f}, "
            f"Num: {avg_scores['numeric']:.2f}, "
            f"Sem: {avg_scores['bert_bleu']:.2f}]"
        )

    except Exception as e:
        print(f"[ERROR] processing VQA file: {e}")

def run_bert_bleu_evaluation(input_file, output_file):
    """
    Processes a captioning result file. Uses BERT-BLEU for all entries.
    """
    print(f"[INFO] Evaluating Caption Dataset: {os.path.basename(input_file)}")
    
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)

        scores = []
        for item in tqdm(results, desc="Scoring Caption"):
            score = calculate_bert_bleu(
                item.get('ground_truth'),
                item.get('model_answer')
            )
            scores.append(score)
            item['bert_bleu'] = score

        avg = sum(scores)/len(scores) if scores else 0

        output_data = {
            "summary": {
                "mean_bert_bleu": avg,
                "total_samples": len(scores)
            },
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"   -> Mean BERT-BLEU: {avg:.4f}")

    except Exception as e:
        print(f"[ERROR] processing Caption file: {e}")

def main():
    model_names = ["llavanext", "qwen", "rsllava"]

    # Process VQA Files
    for model in model_names:
        in_f = os.path.join(VQA_RESULTS_PATH, f"{model}.json")
        out_f = os.path.join(VQA_METRICS_PATH, f"{model}_metrics.json")
        
        if os.path.exists(in_f):
            run_full_vqa_evaluation(in_f, out_f)
        else:
            print(f"[WARN] VQA results for {model} not found. Skipping.")

    # Process Caption Files
    for model in model_names:
        in_f = os.path.join(CAP_RESULTS_PATH, f"{model}.json")
        out_f = os.path.join(CAP_METRICS_PATH, f"{model}_metrics.json")
        
        if os.path.exists(in_f):
            run_bert_bleu_evaluation(in_f, out_f)
        else:
            print(f"[WARN] Caption results for {model} not found. Skipping.")
            
    print("\n[INFO] Evaluation pipeline completed.")

if __name__ == "__main__":
    main()