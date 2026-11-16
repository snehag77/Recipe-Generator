# -*- coding: utf-8 -*-
"""RecipeGen AI — Comprehensive BERT vs Rule-Based Ingredient NER Evaluation.

Evaluates the fine-tuned BERT token classifier against the rule-based parser
on the held-out test split (data/ner_dataset/ner_test.jsonl, 500 sentences, 250 recipes).

Computes:
1. Entity-level Precision, Recall, and F1 (via seqeval and span matching).
2. Token-level classification accuracy.
3. Comparative side-by-side performance with utils.ner rule-based baseline.
4. Detailed inspection of sample predictions.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import ner as rule_ner

DATA_DIR = PROJECT_ROOT / "data" / "ner_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "ingredient_ner"
RESULTS_FILE = DATA_DIR / "evaluation_results.json"

LABELS = ["O", "B-INGREDIENT", "I-INGREDIENT"]
ID2LABEL = {0: "O", 1: "B-INGREDIENT", 2: "I-INGREDIENT"}


def extract_entities_from_bio(tokens: List[str], tags: List[str]) -> List[str]:
    """Convert a sequence of tokens and BIO tags into extracted entity strings."""
    entities = []
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "B-INGREDIENT":
            if current_tokens:
                entities.append(" ".join(current_tokens))
                current_tokens = []
            current_tokens.append(token)
        elif tag == "I-INGREDIENT":
            if current_tokens:
                current_tokens.append(token)
            else:
                # Malformed sequence: treat isolated I- as B-
                current_tokens.append(token)
        else:
            if current_tokens:
                entities.append(" ".join(current_tokens))
                current_tokens = []

    if current_tokens:
        entities.append(" ".join(current_tokens))

    # Clean punctuation tokens and lower-case
    cleaned = []
    for ent in entities:
        c = re.sub(r"\s+([^\w\s])", r"\1", ent).strip()
        c = re.sub(r"([^\w\s])\s+", r"\1", c).strip()
        if len(c) >= 2:
            cleaned.append(c.lower())
    return sorted(list(set(cleaned)))


def compute_entity_set_metrics(
    pred_entities_list: List[List[str]],
    true_entities_list: List[List[str]],
) -> Dict[str, float]:
    """Compute micro-averaged Precision, Recall, F1 over predicted vs true entity sets."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, trues in zip(pred_entities_list, true_entities_list):
        p_set = set(e.lower() for e in preds)
        t_set = set(e.lower() for e in trues)

        tp = len(p_set & t_set)
        fp = len(p_set - t_set)
        fn = len(t_set - p_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-8)

    return {
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def predict_sentence_bert(
    text: str,
    tokenizer,
    model,
    device,
) -> Tuple[List[str], List[str], List[str]]:
    """Predict BIO tags for a raw input string using fine-tuned BERT."""
    # Word-level tokenization matching dataset prep
    words = [m.group(0) for m in re.finditer(r"\w+|[^\w\s]", text)]
    if not words:
        return [], [], []

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

    word_ids = encoding.word_ids(batch_index=0)
    word_tags = []
    prev_word_id = None

    for p_id, w_id in zip(preds, word_ids):
        if w_id is not None and w_id != prev_word_id:
            word_tags.append(ID2LABEL.get(p_id, "O"))
            prev_word_id = w_id

    # Pad or truncate if WordPiece mismatch occurred
    if len(word_tags) < len(words):
        word_tags.extend(["O"] * (len(words) - len(word_tags)))
    else:
        word_tags = word_tags[: len(words)]

    extracted = extract_entities_from_bio(words, word_tags)
    return words, word_tags, extracted


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ingredient NER on Test Split")
    parser.add_argument("--test-file", type=Path, default=DATA_DIR / "ner_test.jsonl")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    print("======================================================================")
    print("RecipeGen AI: Independent Test Evaluation & Benchmark")
    print("======================================================================")

    if not args.test_file.is_file():
        raise FileNotFoundError(f"Missing test split: {args.test_file}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Missing trained model directory: {args.model_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model and tokenizer from: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Load test split
    test_records = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    print(f"Evaluating on {len(test_records)} held-out test sentences...")

    bert_predicted_entities = []
    rule_predicted_entities = []
    ground_truth_entities = []

    seqeval_true = []
    seqeval_pred = []

    total_tokens = 0
    correct_tokens = 0

    t0 = time.time()

    for rec in test_records:
        raw_text = rec["text"]
        true_tokens = rec["tokens"]
        true_tags = rec["ner_tags"]
        true_ents = rec.get("extracted_entities", [])

        # 1. BERT Inference
        pred_words, pred_tags, pred_ents = predict_sentence_bert(raw_text, tokenizer, model, device)

        # 2. Rule-Based Inference
        rule_extracted = rule_ner.extract_ingredients(raw_text)

        bert_predicted_entities.append(pred_ents)
        rule_predicted_entities.append(rule_extracted)
        ground_truth_entities.append(true_ents)

        # Seqeval token comparison (align lengths)
        min_len = min(len(true_tags), len(pred_tags))
        seqeval_true.append(true_tags[:min_len])
        seqeval_pred.append(pred_tags[:min_len])

        for t_t, p_t in zip(true_tags[:min_len], pred_tags[:min_len]):
            total_tokens += 1
            if t_t == p_t:
                correct_tokens += 1

    eval_duration = time.time() - t0

    # Metrics
    bert_seqeval_f1 = f1_score(seqeval_true, seqeval_pred) * 100.0
    bert_seqeval_p = precision_score(seqeval_true, seqeval_pred) * 100.0
    bert_seqeval_r = recall_score(seqeval_true, seqeval_pred) * 100.0
    token_acc = (correct_tokens / max(total_tokens, 1)) * 100.0

    bert_set_metrics = compute_entity_set_metrics(bert_predicted_entities, ground_truth_entities)
    rule_set_metrics = compute_entity_set_metrics(rule_predicted_entities, ground_truth_entities)

    print("\n----------------------------------------------------------------------")
    print("📊 1. BERT Token-Level & BIO Sequence Metrics (seqeval):")
    print(f"   - Token Classification Accuracy: {token_acc:.2f}%")
    print(f"   - Entity Precision (seqeval):   {bert_seqeval_p:.2f}%")
    print(f"   - Entity Recall (seqeval):      {bert_seqeval_r:.2f}%")
    print(f"   - Entity F1-Score (seqeval):    {bert_seqeval_f1:.2f}%")

    print("\n📊 2. Entity-Span Set Match: BERT vs Rule-Based Parser (Same Test Split):")
    print("   Metric                BERT Model        Rule-Based Parser    Delta (BERT vs Rule)")
    print("   -------------------------------------------------------------------------------")
    print(f"   Precision:            {bert_set_metrics['precision']:.2f}%           {rule_set_metrics['precision']:.2f}%             {bert_set_metrics['precision'] - rule_set_metrics['precision']:+.2f}%")
    print(f"   Recall:               {bert_set_metrics['recall']:.2f}%           {rule_set_metrics['recall']:.2f}%             {bert_set_metrics['recall'] - rule_set_metrics['recall']:+.2f}%")
    print(f"   Entity F1-Score:      {bert_set_metrics['f1']:.2f}%           {rule_set_metrics['f1']:.2f}%             {bert_set_metrics['f1'] - rule_set_metrics['f1']:+.2f}%")
    print(f"   True Positives (TP):  {bert_set_metrics['tp']}             {rule_set_metrics['tp']}")
    print(f"   False Positives (FP): {bert_set_metrics['fp']}             {rule_set_metrics['fp']}")
    print(f"   False Negatives (FN): {bert_set_metrics['fn']}             {rule_set_metrics['fn']}")

    # Pick 5 qualitative comparison examples
    examples = []
    print("\n----------------------------------------------------------------------")
    print("🔍 3. Qualitative Side-by-Side Comparison Examples:")
    print("----------------------------------------------------------------------")

    for i in [3, 12, 27, 45, 68]:
        rec = test_records[i]
        text = rec["text"]
        true_ents = rec["extracted_entities"]
        b_ents = bert_predicted_entities[i]
        r_ents = rule_predicted_entities[i]

        ex_dict = {
            "index": i,
            "recipe_id": rec["recipe_id"],
            "input_text": text,
            "expected_entities": true_ents,
            "bert_predictions": b_ents,
            "rule_predictions": r_ents,
        }
        examples.append(ex_dict)

        print(f"\n[Example {len(examples)}] Input: \"{text}\"")
        print(f"   - Expected Ground-Truth: {true_ents}")
        print(f"   - BERT Predictions:      {b_ents}")
        print(f"   - Rule-Based Extraction: {r_ents}")

    # Save results to file
    summary = {
        "num_test_sentences": len(test_records),
        "eval_duration_seconds": eval_duration,
        "token_accuracy": token_acc,
        "bert_seqeval": {
            "precision": bert_seqeval_p,
            "recall": bert_seqeval_r,
            "f1": bert_seqeval_f1,
        },
        "bert_entity_metrics": bert_set_metrics,
        "rule_entity_metrics": rule_set_metrics,
        "comparison_examples": examples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Complete evaluation telemetry saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
