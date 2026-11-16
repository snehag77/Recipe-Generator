# RecipeGen AI (P45): End-to-End Fine-Tuned Recipe Generator

RecipeGen AI is an AI-powered culinary application that transforms a user-provided list of ingredients into a structured, tasty recipe, validates measurements, screens for common allergen risks, and compares base vs fine-tuned neural models.

Fine-tuned with **TensorFlow 2**, **Keras 3**, and **KerasHub** (`gpt2_base_en`) on the RecipeNLG corpus.

---

## 🌟 Key Features

1. **Structured Recipe Generation (U1):** Explicit delimiter tokens (`<INGREDIENTS> ... <END_ING> ... <RECIPE>`) conditioning GPT-2 on ingredient inputs.
2. **GPT-2 Fine-Tuning (U2):** Fine-tuned decoder-only causal language model saved to `models/gpt2_finetuned.weights.h5`.
3. **Safety & Allergen Warnings (U5a):** Rule-based screening for 9 major allergen families with false-positive neutralizations (e.g. `peanut butter` ≠ dairy).
4. **Quantity & Measurement Validation (U5b):** Full fraction parsing (`1 1/2 tbsp`, `½ tsp`), negative/zero bound checks, and culinary unit threshold audits.
5. **Model Comparison Suite:** Side-by-side evaluation of base vs fine-tuned models across 5 reference benchmark ingredient sets.
6. **Training Dashboard:** Real-time visibility into loss curves, perplexity, and training configurations.

---

## 🚀 Setup & Launch Instructions

### 1. Requirements
- macOS (Apple Silicon M-Series or Intel)
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
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Pipeline Scripts

| Script | Purpose |
| :--- | :--- |
| `scripts/train_gpt2.py` | Fine-tunes GPT-2 with AdamW and cosine decay on `data/processed/recipes_train.jsonl` |
| `scripts/validate_recipe_dataset.py` | Quality audit ensuring delimiter adherence and zero cross-split leakage |
| `scripts/analyze_recipe_dataset.py` | Profiles character and token length distributions |
| `scripts/check_tensorflow.py` | Verifies TensorFlow 2, Keras 3 backend, and Apple Silicon compute devices |
| `scripts/check_gpt2.py` | Verifies GPT-2 base model loading and autoregressive text generation |

---

## 📂 Project Architecture

```
Recipe_genai/
├── app/
│   ├── main.py                     # App entry point with lazy page loading
│   └── pages/
│       ├── 1_Recipe_Generator.py   # Real fine-tuned GPT-2 generation & safety checks
│       ├── 2_Ingredient_Analysis.py# Token-level ingredient breakdown & allergen screening
│       ├── 3_Model_Comparison.py   # Base vs Fine-Tuned comparison on 5 benchmark sets
│       └── 4_Model_Training.py     # Training loss progression & perplexity metrics
├── data/
│   ├── processed/                  # Cleaned 90:5:5 JSONL splits (train, val, test)
│   └── outputs/                    # Saved metrics and training logs
├── models/
│   └── gpt2_finetuned.weights.h5   # Fine-tuned GPT-2 checkpoint (TensorFlow 2 / KerasHub)
├── scripts/                        # Training, dataset preparation, and validation scripts
├── utils/                          # Allergen, validation, prompt, and model utilities
└── requirements.txt                # Production deployment dependencies
```

---

## 🔒 Deployment Notes

- The raw 2.29 GB dataset (`data/archive 2/`) is strictly excluded via `.gitignore` to keep deployment builds lightweight and performant.
- All model checkpoints and cache paths load directly from local storage with graceful fallbacks.
