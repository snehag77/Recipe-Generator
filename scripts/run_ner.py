# -*- coding: utf-8 -*-
"""RecipeGen AI — Standalone BERT Ingredient NER Runner.

Executes token classification using the fine-tuned BERT model under venv_pt.
Returns JSON-formatted extraction results for seamless IPC integration.
"""

import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "ingredient_ner"
ID2LABEL = {0: "O", 1: "B-INGREDIENT", 2: "I-INGREDIENT"}

_CACHED_TOKENIZER = None
_CACHED_MODEL = None
_CACHED_DEVICE = None


def get_model_and_tokenizer():
    global _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE
    if _CACHED_MODEL is not None:
        return _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"BERT NER model not found at {MODEL_DIR}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    _CACHED_TOKENIZER = tokenizer
    _CACHED_MODEL = model
    _CACHED_DEVICE = device
    return tokenizer, model, device


def extract_entities_from_bio(tokens, tags):
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
                current_tokens.append(token)
        else:
            if current_tokens:
                entities.append(" ".join(current_tokens))
                current_tokens = []

    if current_tokens:
        entities.append(" ".join(current_tokens))

    cleaned = []
    seen = set()
    for ent in entities:
        c = re.sub(r"\s+([^\w\s])", r"\1", ent).strip()
        c = re.sub(r"([^\w\s])\s+", r"\1", c).strip()
        c_lower = c.lower()
        if len(c_lower) >= 2 and c_lower not in seen:
            seen.add(c_lower)
            cleaned.append(c)

    return cleaned


def predict_ingredients(text: str):
    import torch

    tokenizer, model, device = get_model_and_tokenizer()

    words = [m.group(0) for m in re.finditer(r"\w+|[^\w\s]", text)]
    if not words:
        return []

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

    if len(word_tags) < len(words):
        word_tags.extend(["O"] * (len(words) - len(word_tags)))
    else:
        word_tags = word_tags[: len(words)]

    return extract_entities_from_bio(words, word_tags)


def main():
    parser = argparse.ArgumentParser(description="Run BERT Ingredient NER")
    parser.add_argument("--text", type=str, default=None, help="Input sentence or recipe text")
    args = parser.parse_args()

    input_text = args.text
    if input_text is None:
        input_text = sys.stdin.read().strip()

    if not input_text:
        print(json.dumps({"status": "ok", "method": "bert", "ingredients": []}))
        return

    try:
        ingredients = predict_ingredients(input_text)
        print(json.dumps({
            "status": "ok",
            "method": "bert",
            "ingredients": ingredients,
        }))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "method": "bert",
            "error": str(e),
            "ingredients": [],
        }))


if __name__ == "__main__":
    main()
