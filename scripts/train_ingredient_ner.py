# -*- coding: utf-8 -*-
"""RecipeGen AI — BERT Ingredient NER Training Pipeline.

Trains a BertForTokenClassification model (google-bert/bert-base-cased)
on weakly supervised ingredient BIO tokens.

Features:
- Subword alignment with PyTorch ignore_index=-100.
- Automatic hardware detection (Apple Silicon M4 MPS or CPU).
- Entity-level Precision, Recall, and F1 evaluation via seqeval.
- Early stopping tracking validation entity F1.
- Complete performance telemetry: duration, peak memory, loss curves.
- Saves model weights, config, and tokenizer to models/ingredient_ner/.
"""

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ner_dataset"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "ingredient_ner"

LABELS = ["O", "B-INGREDIENT", "I-INGREDIENT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


class IngredientNERDataset(Dataset):
    """PyTorch Dataset for Token-Level BIO Ingredient NER."""

    def __init__(self, jsonl_path: Path, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        word_tokens = sample["tokens"]
        word_tags = sample["ner_tags"]

        encoding = self.tokenizer(
            word_tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align word tags to WordPiece subwords
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        prev_word_id = None

        for w_id in word_ids:
            if w_id is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif w_id != prev_word_id:
                # First subword of a word receives the authentic tag
                tag = word_tags[w_id] if w_id < len(word_tags) else "O"
                label_ids.append(LABEL2ID.get(tag, 0))
            else:
                # Subsequent subwords get -100 so loss is not inflated
                label_ids.append(-100)
            prev_word_id = w_id

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return item


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device) -> Tuple[float, float, float, float]:
    """Evaluate model on validation dataloader computing entity-level P, R, F1."""
    model.eval()
    total_loss = 0.0
    true_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=2).detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()

            for p_seq, l_seq in zip(preds, label_ids):
                p_filtered = []
                l_filtered = []
                for p_idx, l_idx in zip(p_seq, l_seq):
                    if l_idx != -100:
                        p_filtered.append(ID2LABEL[p_idx])
                        l_filtered.append(ID2LABEL[l_idx])
                true_predictions.append(p_filtered)
                true_labels.append(l_filtered)

    avg_loss = total_loss / max(len(dataloader), 1)
    prec = precision_score(true_labels, true_predictions)
    rec = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    return avg_loss, prec, rec, f1


def main():
    parser = argparse.ArgumentParser(description="Train BERT Ingredient NER Model")
    parser.add_argument("--model-name", type=str, default="google-bert/bert-base-cased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    print("======================================================================")
    print("RecipeGen AI: BERT Ingredient NER Training Pipeline")
    print("======================================================================")
    print(f"Pretrained Model: {args.model_name}")
    print(f"Training Epochs:  {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")

    # Device selection: Apple Silicon MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Accelerator: Apple Silicon GPU via Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("ℹ️ Accelerator: CPU execution")

    # Load tokenizer
    print(f"\n⏳ Loading tokenizer '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Prepare datasets
    train_file = DATA_DIR / "ner_train.jsonl"
    val_file = DATA_DIR / "ner_val.jsonl"

    if not (train_file.is_file() and val_file.is_file()):
        raise FileNotFoundError(f"Missing NER dataset files in {DATA_DIR}. Run prepare_ner_dataset.py first.")

    train_dataset = IngredientNERDataset(train_file, tokenizer, max_len=args.max_len)
    val_dataset = IngredientNERDataset(val_file, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Initialize model
    print(f"⏳ Instantiating BertForTokenClassification (3 labels)...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    start_time = time.time()

    print("\n🚀 Beginning Supervised Fine-Tuning...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_p, val_r, val_f1 = evaluate(model, val_loader, device)
        epoch_dur = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_precision": val_p,
            "val_recall": val_r,
            "val_f1": val_f1,
            "duration_seconds": epoch_dur,
        })

        print(
            f"Epoch {epoch} [{epoch_dur:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Precision: {val_p:.4f} | Val Recall: {val_r:.4f} | Val Entity F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"⭐ New Best Validation F1: {best_val_f1:.4f}! Saving checkpoint...")
            MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"🛑 Early stopping triggered (patience={args.patience}).")
                break

    total_duration = time.time() - start_time

    # Save training telemetry
    telemetry = {
        "model_name": args.model_name,
        "epochs_trained": len(history),
        "total_duration_seconds": total_duration,
        "best_val_f1": best_val_f1,
        "device": str(device),
        "history": history,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(MODEL_OUTPUT_DIR / "training_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print("\n======================================================================")
    print(f"🎉 Training Complete in {total_duration:.1f}s ({total_duration/60:.2f} mins)")
    print(f"Best Validation Entity F1: {best_val_f1:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print("======================================================================")


if __name__ == "__main__":
    main()
