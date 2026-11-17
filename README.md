# Recipe-Generator: RecipeGen AI (P45)

RecipeGen AI is an AI-powered culinary application that transforms a user-provided list of ingredients into a structured, tasty recipe, validates measurements, screens for common allergen risks, and compares base vs fine-tuned neural models.

Fine-tuned with **TensorFlow 2**, **Keras 3**, and **KerasHub** (`gpt2_base_en`) on the RecipeNLG corpus, and featuring **BERT Named Entity Recognition (NER)** for ingredient extraction.

---

## 🌟 Key Features

1. **Structured Recipe Generation (U1):** Explicit delimiter tokens (`<INGREDIENTS> ... <END_ING> ... <RECIPE>`) conditioning GPT-2 on ingredient inputs.
2. **GPT-2 Fine-Tuning (U2):** Fine-tuned decoder-only causal language model saved to `models/recipe_gpt2_full/`.
3. **BERT Ingredient NER (Phase 4):** Token-classification entity recognition (`google-bert/bert-base-cased`) extracting clean culinary entities.
4. **Safety & Allergen Warnings (U5a):** Rule-based screening for 9 major allergen families with false-positive neutralizations (e.g. `peanut butter` ≠ dairy).
5. **Quantity & Measurement Validation (U5b):** Full fraction parsing (`1 1/2 tbsp`, `½ tsp`), negative/zero bound checks, and culinary unit threshold audits.
6. **Model Comparison Suite:** Side-by-side evaluation of base vs fine-tuned models across reference benchmark ingredient sets.
7. **Training Dashboard:** Real-time visibility into loss curves, perplexity, and training configurations.

---

## 🚀 Setup & Launch Instructions

### 1. Requirements
- macOS (Apple Silicon M-Series or Intel) / Linux
- Python 3.12 installed (`python3.12 --version`)

### 2. Environment Activation
```bash
cd /Users/snehagupta/Downloads/Recipe_genai

# Activate the dedicated TensorFlow 2 / KerasHub environment
source venv_tf/bin/activate
```

### 3. Launch Streamlit Application
```bash
streamlit run app/main.py
```
Open **`http://localhost:8503`** (or configured port) in your browser.

---

## 🧪 Pipeline Scripts

| Script | Purpose |
| :--- | :--- |
| `scripts/train_gpt2.py` | Fine-tunes GPT-2 with AdamW and cosine decay on `data/processed/recipes_train.jsonl` |
| `scripts/train_ingredient_ner.py` | Fine-tunes BERT for token-level ingredient named entity recognition |
| `scripts/evaluate_ingredient_ner.py` | Benchmarks BERT NER against rule-based parser on 500 held-out sentences |
| `scripts/validate_recipe_dataset.py` | Quality audit ensuring delimiter adherence and zero cross-split leakage |
| `scripts/analyze_recipe_dataset.py` | Profiles character and token length distributions |
| `scripts/check_tensorflow.py` | Verifies TensorFlow 2, Keras 3 backend, and compute devices |
| `scripts/check_gpt2.py` | Verifies GPT-2 base model loading and autoregressive text generation |

---

## 📂 Project Architecture

```
Recipe_genai/
├── app/
│   ├── main.py                     # App entry point with lazy page loading
│   └── pages/
│       ├── 1_Recipe_Generator.py   # Real fine-tuned GPT-2 generation & safety checks
│       ├── 2_Ingredient_Analysis.py# BERT NER & rule-based extraction & allergen screening
│       ├── 3_Model_Comparison.py   # Base vs Fine-Tuned comparison on benchmark sets
│       └── 4_Model_Training.py     # Training loss progression & held-out test metrics
├── data/
│   ├── processed/                  # Cleaned 90:5:5 JSONL splits (train, val, test)
│   ├── ner_dataset/                # NER BIO dataset and evaluation benchmarks
│   └── outputs/                    # Saved metrics and training logs
├── models/
│   ├── ingredient_ner/             # Fine-tuned BERT NER checkpoint config & tokenizer
│   └── recipe_gpt2_full/           # Fine-tuned GPT-2 checkpoint config & history
├── scripts/                        # Training, dataset preparation, and validation scripts
├── utils/                          # Allergen, validation, prompt, ner, and model utilities
└── requirements.txt                # Production deployment dependencies
```

---

## 🔒 Deployment Notes

- The raw 2.29 GB dataset (`data/archive 2/`) and binary model checkpoints (`*.weights.h5`, `*.safetensors`) are strictly excluded via `.gitignore` to keep deployment builds lightweight and performant.
- All model checkpoints and cache paths load directly from local storage with graceful fallbacks.
