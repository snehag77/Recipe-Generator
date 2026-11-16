# RecipeGen AI (P45) – Complete Project Status Audit

**Audit Date:** 2026-11-16 14:58 IST  
**Project Path:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Device:** MacBook Air (Apple Silicon arm64, Apple M4 / T8132, macOS 26.6.2, 16 GB Unified Memory)  
**Auditor:** Antigravity Senior AI/ML Engineer & Auditor  

---

## 1. Executive Summary & Completion Percentage

| Metric | Assessment |
| :--- | :--- |
| **Overall Project Completion** | **28%** |
| **Status Classification** | Stage 1 UI & Logic **STABILIZED**; Stage 2 ML Environment **INITIALIZED**; Model Training & Weights **NOT IMPLEMENTED / NOT LOADED** |
| **Readiness for GPT-2 Fine-Tuning** | **NOT READY YET** – Dataset is missing, and the base preset must be downloaded once in `venv_tf`. |

### Completion Percentage Breakdown (Weighted)
| Requirement / Layer | Weight | Actual Status | Points Earned |
| :--- | :---: | :--- | :---: |
| **U1 – Structured Prompt Design** | 15% | **COMPLETE** (Tags `<INGREDIENTS>...<END_ING>`, `<RECIPE>`, `<TITLE>`, etc.) | 15.0% |
| **U2 – GPT-2 Fine-Tuning & Comparison** | 25% | **PLACEHOLDER** (Demo template only; no fine-tuned weights, no dataset) | 2.5% |
| **U3 – BERT NER & Diffusion Image** | 20% | **PLACEHOLDER / NOT IMPLEMENTED** (Comma-split only; no BERT or diffusion) | 2.0% |
| **U4 – TensorFlow 2 Training Pipeline** | 15% | **PARTIALLY COMPLETE** (TF 2.21 + Keras 3 installed in `venv_tf`, no training script) | 4.0% |
| **U5 – Allergen & Quantity Validation** | 15% | **COMPLETE** (Fractions, bound checks, safe substitutions working) | 15.0% |
| **Application UI & Navigation** | 10% | **PARTIALLY COMPLETE** (4 pages load with `run()`; pages 3 & 4 are placeholders) | 5.0% |
| **Total Weighted Completion** | **100%** | | **28.5% (approx 28%)** |

---

## 2. Project File Structure & Inventory

```
/Users/snehagupta/Downloads/Recipe_genai
├── app/
│   ├── main.py                        [COMPLETE] Entry point, lazy loader, error boundary
│   └── pages/
│       ├── 1_Recipe_Generator.py      [COMPLETE (Demo)] Input, allergen, quantity checks, template output
│       ├── 2_Ingredient_Analysis.py   [COMPLETE (Demo)] Rule-based extraction & allergen screening
│       ├── 3_Model_Comparison.py      [PLACEHOLDER] Informational only; benchmark plan
│       ├── 4_Model_Training.py        [PLACEHOLDER] Informational only; offline training architecture
├── data/
│   └── outputs/                       [EMPTY] Missing metrics, curves, and comparison JSONs
├── models/                            [EMPTY] Missing weights (.h5) and NER config
├── scripts/
│   ├── check_tensorflow.py            [COMPLETE] TF 2 / Keras backend & device check
│   ├── check_gpt2.py                  [COMPLETE] KerasHub GPT-2 preset & generation test
│   ├── find_needed_files.py           [UTILITY] Script for locating files on Google Colab / Drive
│   ├── package_models.py              [UTILITY] Script for archiving Colab models to zip
│   └── copy_to_drive.py               [UTILITY] Script for backing up to Google Drive
├── utils/
│   ├── __init__.py                    [COMPLETE] Package marker
│   ├── allergen.py                    [COMPLETE] 9 allergen groups + false-positive neutralization
│   ├── gpt2.py                        [PLACEHOLDER] Stage 1 demo template; no neural model
│   ├── ner.py                         [PLACEHOLDER] Comma-separated split; no BERT model
│   ├── prompt.py                      [COMPLETE] Standardized U1 prompt format with delimiters
│   └── validation.py                  [COMPLETE] Fractions (½, 1 1/2), negative/zero/excess checks
├── venv/                              [STAGE 1 ENV] Python 3.14.7, Streamlit 1.67.0 (Intact)
├── venv_tf/                           [STAGE 2 ENV] Python 3.12.8, TF 2.21.0, Keras 3.15.1, KerasHub 0.33.0
├── requirements.txt                   [PARTIAL] Lists Streamlit and Pillow only
├── ENVIRONMENT_SETUP.md               [COMPLETE] Guide for venv_tf setup
└── README.md                          [OUTDATED] Claims full fine-tuning/diffusion exist
```

---

## 3. Application & UI Status

| Component | Status | Verification & Evidence |
| :--- | :---: | :--- |
| **`app/main.py`** | **COMPLETE** | Successfully parses sys.path, configures Streamlit layout, uses dynamic `load_page()`, catches page exceptions cleanly. |
| **Page 1 (`1_Recipe_Generator.py`)** | **PARTIALLY COMPLETE (Demo Mode)** | Defines `run()`. Correctly inputs ingredients, triggers allergen and quantity validations, formats structured prompt, displays demo recipe output. **Does not use a neural language model.** |
| **Page 2 (`2_Ingredient_Analysis.py`)** | **PARTIALLY COMPLETE (Demo Mode)** | Defines `run()`. Displays extracted tokens, quantity validations, and allergen screening. **Uses rule-based comma-splitting, not BERT NER.** |
| **Page 3 (`3_Model_Comparison.py`)** | **PLACEHOLDER** | Defines `run()`. Displays planned benchmark sets and metrics. Displays zero fabricated results. |
| **Page 4 (`4_Model_Training.py`)** | **PLACEHOLDER** | Defines `run()`. Explains Colab GPU training architecture and expected artifacts. No fake training buttons. |
| **Navigation & Stability** | **COMPLETE** | Verified: all 4 pages contain callable `run()` methods and do not crash on load. |

---

## 4. Utility Modules Audit

| Module | Status | Verification Findings |
| :--- | :---: | :--- |
| **`utils/prompt.py`** | **COMPLETE** | Generates `<INGREDIENTS>\n...\n<END_ING>\n<RECIPE>\n<TITLE>...\n<SERVINGS>...\n<PREP_TIME>\n<COOK_TIME>\n<QUANTITIES>\n<STEPS>\n<END_RECIPE>`. Fully compatible with planned GPT-2 training format. |
| **`utils/gpt2.py`** | **PLACEHOLDER** | Contains `generate_demo_recipe()`. Outputs a formatted template string using input ingredients. No Keras or neural weights involved. |
| **`utils/ner.py`** | **PLACEHOLDER** | Contains `extract_ingredients()`. Splits string by commas and strips whitespace. No BERT embeddings, tokenizers, or weights. |
| **`utils/allergen.py`** | **COMPLETE** | Covers Milk/Dairy, Peanuts, Tree Nuts, Wheat/Gluten, Soy, Eggs, Fish, Shellfish, Sesame. Tested and verified: safe substitutions (`peanut butter`, `coconut milk`, `cream of tartar`) do NOT cause false Milk/Dairy alerts. |
| **`utils/validation.py`** | **COMPLETE** | Tested and verified: parses Unicode fractions (`½`), mixed numbers (`1 1/2`), decimals (`0.75`), flags negatives (`-1 cup`), flags zero (`0 g`), flags unrealistic weights (`500 kg butter`). |

---

## 5. Machine Learning Environment Status

### Two Distinct Virtual Environments Detected:
1. **`venv/` (Stage 1 Demo Environment):**
   - **Python:** 3.14.7
   - **Packages:** `streamlit==1.67.0`, `pillow==13.0.0`
   - **TensorFlow Status:** Broken on Python 3.14 (unsupported platform binary).
   - **Role:** Dedicated solely to running the Stage 1 demo application.
2. **`venv_tf/` (Stage 2 ML Environment - Freshly Installed):**
   - **Python:** 3.12.8 (`/Library/Frameworks/Python.framework/Versions/3.12`)
   - **TensorFlow:** `2.21.0` (Native arm64) – **IMPORTS SUCCESSFULLY** ✅
   - **Keras:** `3.15.1` (Backend: `tensorflow`) – **IMPORTS SUCCESSFULLY** ✅
   - **KerasHub (KerasNLP):** `0.33.0` – **IMPORTS SUCCESSFULLY** ✅
   - **Tensor Calculation:** Verified 2x2 matrix multiplication (`tf.matmul`) produces correct result (sum = 134) ✅
   - **Physical Devices:** CPU (`/physical_device:CPU:0`).
   - **Metal GPU Status:** Not active (standard CPU mode). `tensorflow-metal` is not installed; CPU execution functions reliably.
   - **KerasHub GPT-2 Base Preset:** Library registered 5 presets (`gpt2_base_en` selected). Model weights not yet downloaded locally (download failed inside sandbox due to blocked network access to `api.kaggle.com`).

---

## 6. P45 Academic Requirements Status Matrix

| Requirement | Description | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **U1** | Structured ingredient-to-recipe prompt format | **COMPLETE** | `<INGREDIENTS> ... <END_ING>` delimiter format implemented and active in `utils/prompt.py`. |
| **U2** | Fine-tune GPT-2 using KerasNLP & compare base vs fine-tuned | **NOT IMPLEMENTED** | No fine-tuning scripts or weights exist in project. Model generation is currently a demo template. |
| **U3** | Identify ingredients via BERT NER & dish image via diffusion | **NOT IMPLEMENTED** | `utils/ner.py` is a simple string split. Diffusion model is not implemented. |
| **U4** | Model training in TensorFlow 2 | **NOT IMPLEMENTED** | TensorFlow 2.21 environment is installed, but no training script has run; `models/` is empty. |
| **U5** | Allergen detection & impossible quantity validation | **COMPLETE** | Both `utils/allergen.py` and `utils/validation.py` are robust, tested, and active. |
| **Deliverables** | Streamlit app + 5-set model comparison table | **PARTIALLY COMPLETE** | App runs; Model Comparison page is currently a placeholder without data. |

---

## 7. Test Results Summary

| Test | Target | Command / Check | Result |
| :--- | :--- | :--- | :---: |
| **T1: Python Syntax** | All 16 `.py` files | `python3 -m py_compile` | **PASS** (16/16) |
| **T2: Page Interface** | `app/pages/*.py` | Verify `hasattr(module, 'run')` | **PASS** (4/4) |
| **T3: Allergen Screening** | `utils/allergen.py` | Detect true allergens & ignore safe swaps | **PASS** |
| **T4: Quantity Validation** | `utils/validation.py` | Parse fractions, catch negatives, zeros, high bounds | **PASS** |
| **T5: Prompt Generation** | `utils/prompt.py` | Produce U1 delimiter blocks | **PASS** |
| **T6: TF & Keras Import** | `scripts/check_tensorflow.py` | Execute inside `venv_tf` (Python 3.12) | **PASS** |
| **T7: GPT-2 Base Inference** | `scripts/check_gpt2.py` | Load `gpt2_base_en` from Kaggle/KerasHub | **BLOCKED (Offline)** |
| **T8: App Launch** | `app/main.py` | Streamlit server startup | **PASS** (Terminal) |

---

## 8. Main Blockers & Critical Path

1. **Pretrained Weights Download (One-time network step):**  
   Inside the local terminal (with live internet), running `python scripts/check_gpt2.py` inside `venv_tf` must be completed once so KerasHub can cache the `gpt2_base_en` tokenizer and base weights (~500 MB).
2. **Missing Recipe Dataset:**  
   There is currently no recipe corpus (e.g., RecipeNLG subset or sample JSON/CSV) in `data/` to fine-tune on.
3. **No Trained Weights or Training Script:**  
   `models/` is completely empty. We need either:
   - A fine-tuning training script (`scripts/train_gpt2.py`) to run on a small dataset, OR
   - The `.weights.h5` files if trained on Google Colab.

---

## 9. Is the Project Ready for GPT-2 Fine-Tuning?

### ❌ **NOT YET.**
To be ready, two immediate prerequisites must be completed:
1. **Network Cache Check:** Run `python scripts/check_gpt2.py` inside `venv_tf` in your Mac terminal so KerasHub downloads the base GPT-2 weights.
2. **Dataset Decision:** Provide or generate the recipe training dataset (e.g. `data/recipes.txt` or a formatted JSON file with ~1,000–5,000 recipes).

---

## 10. Prioritized Next Steps

1. **Step 1 (User Action):** In your Mac terminal, activate `venv_tf` and test base GPT-2 loading:
   ```bash
   cd /Users/snehagupta/Downloads/Recipe_genai
   source venv_tf/bin/activate
   python scripts/check_gpt2.py
   ```
2. **Step 2:** Prepare a curated recipe training dataset (`data/recipes_train.json` or `.txt`) formatted with `<INGREDIENTS> ... <END_ING> ... <RECIPE>`.
3. **Step 3:** Create the GPT-2 fine-tuning pipeline script (`scripts/train_gpt2.py`) utilizing TensorFlow 2 and KerasHub `GPT2CausalLM`.
4. **Step 4:** Integrate base vs fine-tuned generation into `app/pages/1_Recipe_Generator.py` and `app/pages/3_Model_Comparison.py`.
