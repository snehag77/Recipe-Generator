# RecipeGen AI – Final Project Status Report

**Generated:** 2026-11-16 14:06 IST  
**Project location:** `/Users/snehagupta/Downloads/Recipe_genai`

---

## 1. Project Overview

| Field | Value |
|-------|-------|
| **Name** | RecipeGen AI (P45: Recipe Generator Fine-Tuned in Keras) |
| **Purpose** | Turn a free-text list of ingredients into a complete recipe using a fine-tuned GPT-2 model, tag ingredients with BERT NER, generate a dish image with diffusion, detect allergens, and flag impossible quantities. |
| **Input** | Free-text ingredient list (e.g. "chicken, rice, garlic, soy sauce") |
| **Output** | Formatted recipe (title, quantities, steps), allergen warnings, quantity flags, optional dish image |
| **Framework** | Streamlit (UI), TensorFlow/Keras (training), KerasHub/KerasNLP (model presets), PyTorch/Diffusers (image generation) |
| **Current stage** | Stage 1 (demo/placeholder) – core UI exists, ML models are NOT integrated |
| **Overall completion** | **~15–20%** (see justification in Section 14) |

---

## 2. Complete File Structure

### 2.1 Application Files

| File | Purpose | `run()` defined? | Status |
|------|---------|-------------------|--------|
| `app/main.py` | Entry point – lazy-loads pages, sidebar navigation, catches errors | N/A (entry point) | ✅ **Working** (updated to lazy-load) |
| `app/pages/1_Recipe_Generator.py` | Recipe generation UI: ingredient input → NER → prompt → GPT-2 → allergen/quantity | ✅ Yes | ⚠️ **Demo only** – uses static template, not a real model |
| `app/pages/2_Ingredient_Analysis.py` | Ingredient analysis placeholder | ❌ No (`load_page()` instead) | 🔴 **Broken** – `app/main.py` expects `run()` but file defines `load_page()` |
| `app/pages/3_Model_Comparison.py` | — | — | 🔴 **Missing** – file does not exist |
| `app/pages/4_Model_Training.py` | — | — | 🔴 **Missing** – file does not exist |

### 2.2 Utility Files

| File | Purpose | Key functions | Status |
|------|---------|--------------|--------|
| `utils/__init__.py` | Package marker | (empty) | ✅ OK |
| `utils/prompt.py` | U1: Build the prompt | `build_prompt(ingredients)` | ⚠️ **Demo only** – natural-language prompt, not the `<INGREDIENTS>…<END_ING>` tag format needed for the fine-tuned model |
| `utils/gpt2.py` | U2: GPT-2 generation | `generate_demo_recipe(prompt)` | 🔴 **Placeholder** – returns a static string; no model loading |
| `utils/ner.py` | U3: Ingredient NER | `extract_ingredients(text)` | 🔴 **Placeholder** – comma-split only; no BERT model |
| `utils/allergen.py` | U5a: Allergen detection | `detect_allergens(ingredients)` | ⚠️ **Partially complete** – basic keyword matching works but limited dictionary, no false-positive handling |
| `utils/validation.py` | U5b: Quantity validation | `parse_quantity(ing)`, `validate_quantities(ings)` | ⚠️ **Partially complete** – catches negatives and >1000, but no unit conversion, no category limits, no fraction parsing |

### 2.3 Scripts

| File | Purpose | Status |
|------|---------|--------|
| `scripts/check_tensorflow.py` | Verify TF import, versions, devices | ✅ Ready (but TF fails in current venv) |
| `scripts/check_gpt2.py` | Verify KerasNLP GPT-2 preset loading | ✅ Ready (but uses old `keras_nlp` API) |
| `scripts/find_needed_files.py` | Locate trained model files on Colab | ✅ Ready (Colab utility) |
| `scripts/package_models.py` | Zip model files for download | ✅ Ready (Colab utility) |
| `scripts/copy_to_drive.py` | Copy zip to Google Drive | ✅ Ready (Colab utility) |

### 2.4 Data & Models

| Directory | Contents | Status |
|-----------|----------|--------|
| `models/` | Should contain `.weights.h5` files and `ner_config.json` | 🔴 **Empty** |
| `data/outputs/` | Should contain metrics JSON, comparison JSON, training curves PNG | 🔴 **Empty** |

### 2.5 Configuration & Documentation

| File | Status |
|------|--------|
| `requirements.txt` | ✅ Exists – only `streamlit>=1.30` and `pillow>=10.2.0` (ML packages not listed) |
| `.streamlit/config.toml` | ✅ Exists – theme and server config present |
| `.gitignore` | 🔴 **Missing** |
| `README.md` | ⚠️ Exists but describes features that are NOT implemented (claims fine-tuning, NER, diffusion, training page) |
| `ENVIRONMENT_SETUP.md` | ⚠️ Outdated – recommends `tensorflow-macos` which is deprecated |
| `PROJECT_AUDIT_REPORT.md` | ✅ Exists (previous audit) |

### 2.6 Virtual Environments

| venv | Python | TF importable? | Keras importable? | Streamlit? | Notes |
|------|--------|----------------|-------------------|------------|-------|
| `venv/` | 3.14.7 | ❌ `PermissionError` on `~/.keras/keras.json` | ❌ Same TF crash | ✅ 1.67.0 | Has torch 2.15, transformers 5.23, diffusers 0.43, keras-hub 0.19, keras-nlp 0.19, but TF is broken |
| `.venv-ml/` | — | — | — | — | 🔴 Does not exist yet |

---

## 3. Technology Stack

| Layer | Package | Version in `venv/` | Status |
|-------|---------|-------------------|--------|
| **UI** | Streamlit | 1.67.0 | ✅ Works |
| **ML backend** | TensorFlow | (installed but crashes) | ❌ **Broken** – Python 3.14 + sandbox `PermissionError` on `~/.keras/keras.json` |
| **High-level API** | Keras | 3.15.1 | ❌ Depends on TF, same crash |
| **Model presets** | keras-hub | 0.19.0 | ❌ Depends on Keras |
| **Model presets (old)** | keras-nlp | 0.19.0 | ❌ Depends on Keras |
| **Tokenizers** | transformers | 5.23.0 | ✅ Works |
| **Image gen** | diffusers | 0.43.0 | ✅ Available (not tested for generation) |
| **GPU** | PyTorch (MPS) | 2.15.0 | ✅ `torch.backends.mps.is_available() = True` |
| **Data** | numpy | 2.5.4 | ✅ Works |
| **Data** | pandas | 3.1.1 | ✅ Works |

---

## 4. Environment Status

| Property | Value |
|----------|-------|
| **macOS** | 26.6.2 (Darwin 25.6.0) |
| **Architecture** | Apple Silicon (arm64, T8132 – MacBook Air M2/M3) |
| **RAM** | 16 GB |
| **Disk free** | 177 GB |
| **Python versions available** | 3.14.7 (main), 3.9.6 (system `/usr/bin/python3`) |
| **Python 3.11/3.12** | ❌ **Not installed** – no Homebrew, no pyenv detected |
| **TensorFlow** | ❌ Crashes on import (Python 3.14 incompatibility + permission error) |
| **Keras / KerasHub** | ❌ Depends on TF; crashes |
| **GPU/Metal** | PyTorch MPS available; TF Metal untested (TF won't import) |

> **Root cause of TF failure:** The `venv/` was created with Python 3.14.7. TensorFlow does not officially support Python 3.14. The installed TF wheel crashes when Keras tries to open `~/.keras/keras.json`. A new venv with Python 3.11 or 3.12 is required.

---

## 5. Requirement-Wise Implementation Status

### U1 – Ingredient-to-Recipe Prompt Design

| Aspect | Status | Evidence |
|--------|--------|----------|
| Prompt function | ⚠️ **Demo only** | `utils/prompt.py` builds a natural-language prompt (not the `<INGREDIENTS>…<END_ING>…<TITLE>` tag format the fine-tuned model expects) |
| Ingredient input UI | ✅ **Working** | `1_Recipe_Generator.py` has a text area and "Generate Recipe" button |
| Recipe output display | ⚠️ **Demo only** | Displays a static template string from `gpt2.generate_demo_recipe()` |

**Verdict: DEMO ONLY**

### U2 – GPT-2 Fine-Tuning

| Aspect | Status | Evidence |
|--------|--------|----------|
| Recipe dataset | 🔴 **Not present** | No `data/recipes.txt` or RecipeNLG data in the project |
| Dataset preprocessing | 🔴 **Not implemented** | No preprocessing scripts |
| KerasNLP/KerasHub GPT-2 loading | 🔴 **Not implemented** | `utils/gpt2.py` is a static placeholder |
| Fine-tuning script | 🔴 **Not implemented** | No `scripts/fine_tune_gpt2.py` exists (README claims it does) |
| Saved model checkpoint | 🔴 **Missing** | `models/` directory is empty |
| Base vs fine-tuned comparison | 🔴 **Not implemented** | No comparison page exists |
| 5 ingredient-set demos | 🔴 **Not implemented** | No `comparison_5sets.json` exists |

**Verdict: NOT IMPLEMENTED** (training was done on Colab but model files have not been copied to this project)

### U3 – BERT NER and Image Generation

| Aspect | Status | Evidence |
|--------|--------|----------|
| BERT NER model | 🔴 **Not implemented** | `utils/ner.py` is a comma-split placeholder |
| Ingredient extraction | 🔴 **Placeholder** | `extract_ingredients()` = `text.split(',')` |
| NER weights | 🔴 **Missing** | `models/ner_bert_final.weights.h5` absent |
| Diffusion model | 🔴 **Not implemented** | No `utils/image_gen.py` exists; `diffusers` is installed but unused |
| Dish image generation | 🔴 **Not implemented** | No image generation code in any page |

**Verdict: PLACEHOLDER / NOT IMPLEMENTED**

### U4 – TensorFlow 2 Training

| Aspect | Status | Evidence |
|--------|--------|----------|
| TensorFlow import | ❌ **Broken** | Crashes on `import tensorflow` (Python 3.14) |
| Training script | 🔴 **Not implemented** | No `scripts/train_tf_model.py` exists |
| Training curves | 🔴 **Missing** | No PNGs in `data/outputs/` |
| Saved training results | 🔴 **Missing** | No metrics JSON files |

**Verdict: NOT IMPLEMENTED** (training was done on Colab; results not present locally)

### U5 – Allergen Detection and Quantity Validation

| Aspect | Status | Evidence |
|--------|--------|----------|
| Allergen dictionary | ⚠️ **Partially complete** | `utils/allergen.py` has 9 categories with basic keywords; no false-positive handling (e.g. "coconut milk" triggers "Milk", "peanut butter" triggers "Milk" via "butter") |
| Allergen detection | ⚠️ **Working but limited** | `detect_allergens()` does simple `if kw in lower` matching |
| Quantity parsing | ⚠️ **Basic** | `parse_quantity()` handles simple `float + unit` but not fractions, mixed numbers, ranges, or parenthesised sizes |
| Impossible quantity detection | ⚠️ **Basic** | Only checks `qty <= 0` or `qty > 1000`; no category-specific limits, no time/temperature checks |
| Warning display | ✅ **Working** | Allergen and quantity warnings display in the Recipe Generator page |

**Verdict: PARTIALLY COMPLETE**

---

## 6. Streamlit Application Status

| Page | Can it load? | What it shows | Status |
|------|-------------|---------------|--------|
| **Home** | ✅ Yes | Title + description | ✅ Working |
| **Recipe Generator** | ✅ Yes (with current utils) | Text area, button, static demo recipe, allergen warnings, quantity warnings | ⚠️ Demo only |
| **Ingredient Analysis** | ❌ **Crashes** | `AttributeError: module 'ingredient_analysis' has no attribute 'run'` | 🔴 Broken |
| **Model Comparison** | ❌ **Crashes** | `FileNotFoundError: 3_Model_Comparison.py` | 🔴 Missing |
| **Model Training** | ❌ **Crashes** | `FileNotFoundError: 4_Model_Training.py` | 🔴 Missing |

> **Note:** The app can only be launched from your terminal (not from the sandbox) because Streamlit tries to write `~/.streamlit/credentials.toml`. The `app/main.py` lazy-loading mechanism means a missing page shows an error box instead of crashing the whole app – but page 2 still crashes because it lacks `run()`.

---

## 7. Machine Learning Implementation Status

| Component | Code exists? | Model loads? | Inference works? | Status |
|-----------|-------------|-------------|------------------|--------|
| GPT-2 (base) | ❌ Placeholder | ❌ | ❌ | Not implemented |
| GPT-2 (fine-tuned) | ❌ Placeholder | ❌ | ❌ | Not implemented |
| BERT NER | ❌ Placeholder | ❌ | ❌ | Not implemented |
| Diffusion (SD-Turbo) | ❌ No code | ❌ | ❌ | Not implemented |

---

## 8. Dataset Status

| Dataset | Expected location | Present? | Notes |
|---------|------------------|----------|-------|
| RecipeNLG subset | `data/` | ❌ No | Training was on Colab |
| `comparison_5sets.json` | `data/outputs/` | ❌ No | Needs to be exported from Colab |
| `base_metrics.json` | `data/outputs/` | ❌ No | " |
| `finetuned_metrics.json` | `data/outputs/` | ❌ No | " |
| `ner_metrics.json` | `data/outputs/` | ❌ No | " |
| `summary_metrics.csv` | `data/outputs/` | ❌ No | " |
| `gpt2_curves.png` | `data/outputs/` | ❌ No | " |
| `ner_curves.png` | `data/outputs/` | ❌ No | " |
| `image_grid.png` | `data/outputs/` | ❌ No | " |

---

## 9. Testing Results

### 9.1 Compilation Test

```
$ python3 -m py_compile (all .py files)
OK  app/main.py
OK  app/pages/1_Recipe_Generator.py
OK  app/pages/2_Ingredient_Analysis.py
OK  scripts/check_tensorflow.py
OK  scripts/check_gpt2.py
OK  scripts/copy_to_drive.py
OK  scripts/find_needed_files.py
OK  scripts/package_models.py
OK  utils/__init__.py
OK  utils/allergen.py
OK  utils/gpt2.py
OK  utils/ner.py
OK  utils/prompt.py
OK  utils/validation.py
```

**Result: 14/14 PASS** – all files are syntactically valid.

### 9.2 Import Tests

| Test | Result | Notes |
|------|--------|-------|
| `import streamlit` | ✅ PASS | v1.67.0 |
| `import torch` | ✅ PASS | v2.15.0, MPS available |
| `import transformers` | ✅ PASS | v5.23.0 |
| `import diffusers` | ✅ PASS | v0.43.0 |
| `import numpy` | ✅ PASS | v2.5.4 |
| `import pandas` | ✅ PASS | v3.1.1 |
| `import tensorflow` | ❌ FAIL | `PermissionError: ~/.keras/keras.json` (Python 3.14 incompatibility) |
| `import keras` | ❌ FAIL | Depends on TF, same crash |
| `import keras_hub` | ❌ FAIL | Depends on Keras |

### 9.3 Functional Tests (run manually)

| Test | Expected | Actual | Result |
|------|----------|--------|--------|
| Allergen: "peanut, milk, sesame" | Detects peanuts, milk, sesame | ✅ Detects all three | PASS |
| Allergen: "potato, onion, salt" | No allergens | ✅ No allergens | PASS |
| Validation: qty = -5 | Warning | ✅ "non-positive quantity" | PASS |
| Validation: qty = 5000 | Warning | ✅ "unrealistic quantity" | PASS |
| NER: "chicken, rice, garlic" | Extract ingredients | ✅ Returns `['chicken', 'rice', 'garlic']` | PASS (but comma-split only) |
| GPT-2 generation | Real recipe | ❌ Returns static template | FAIL (placeholder) |
| Page 2 loads | Shows ingredient analysis | ❌ `AttributeError: no run()` | FAIL |
| Page 3 loads | Shows model comparison | ❌ `FileNotFoundError` | FAIL |
| Page 4 loads | Shows training evidence | ❌ `FileNotFoundError` | FAIL |

### 9.4 App Launch Test

| Test | Result | Notes |
|------|--------|-------|
| `streamlit run app/main.py` from sandbox | ❌ FAIL | `PermissionError` on `~/.streamlit/credentials.toml` (sandbox limitation) |
| `streamlit run app/main.py` from your terminal | ⚠️ Likely works for Home + Page 1 | Requires you to press Enter at the email prompt, then open http://localhost:8501 |

---

## 10. Errors and Limitations

| # | Error | Cause | Fix |
|---|-------|-------|-----|
| 1 | `import tensorflow` → `PermissionError: ~/.keras/keras.json` | Python 3.14 + TF incompatibility; sandbox also blocks `~/.keras` access | Create new venv with Python 3.11 or 3.12 |
| 2 | Page 2 `AttributeError: no attribute 'run'` | `2_Ingredient_Analysis.py` defines `load_page()` not `run()` | Rename function to `run()` |
| 3 | Pages 3, 4 `FileNotFoundError` | Files don't exist | Create `3_Model_Comparison.py` and `4_Model_Training.py` |
| 4 | `models/` empty | Trained weights not copied from Colab | Export from Colab → copy to project |
| 5 | `data/outputs/` empty | Metrics/comparison files not copied | Same as above |
| 6 | `utils/gpt2.py` static demo | No real model loading code | Replace with KerasHub GPT-2 loading |
| 7 | `utils/ner.py` comma-split | No BERT NER model | Replace with BERT NER implementation |
| 8 | `utils/prompt.py` wrong format | Uses natural-language prompt, not tag format | Replace with `<INGREDIENTS>…<END_ING>…<TITLE>` format |
| 9 | Allergen false positives | "peanut butter" triggers "Milk" (via "butter") | Add safe-swap list |
| 10 | No `.gitignore` | Heavy directories could be committed | Create `.gitignore` |
| 11 | Only Python 3.14 installed | TF needs 3.11/3.12 | Install Python 3.11 or 3.12 via Homebrew |
| 12 | `README.md` claims features exist | Misleading for the actual state | Rewrite after implementation |

---

## 11. Missing Features (Prioritized)

| Priority | Feature | Files affected | Blocking? |
|----------|---------|---------------|-----------|
| 🔴 **CRITICAL** | Python 3.11/3.12 environment for TensorFlow | New `.venv-ml/` | Yes – blocks all ML |
| 🔴 **CRITICAL** | Copy trained model weights from Colab | `models/` | Yes – blocks all ML pages |
| 🔴 **CRITICAL** | Copy output artifacts from Colab | `data/outputs/` | Yes – blocks comparison & training pages |
| 🔴 **CRITICAL** | Fix page 2 (`run()` function) | `app/pages/2_Ingredient_Analysis.py` | Yes – page crashes |
| 🔴 **CRITICAL** | Create pages 3 and 4 | `app/pages/3_Model_Comparison.py`, `4_Model_Training.py` | Yes – pages missing |
| 🟠 **HIGH** | Replace `utils/gpt2.py` with real model loading | `utils/gpt2.py` | Yes – core feature |
| 🟠 **HIGH** | Replace `utils/ner.py` with BERT NER | `utils/ner.py` | Yes – core feature |
| 🟠 **HIGH** | Replace `utils/prompt.py` with tag format | `utils/prompt.py` | Yes – model expects it |
| 🟠 **HIGH** | Create `utils/config.py` | New file | Yes – paths and settings |
| 🟠 **HIGH** | Create `utils/app_state.py` | New file | Yes – cached model loaders |
| 🟡 **MEDIUM** | Upgrade `utils/allergen.py` (false-positive handling) | `utils/allergen.py` | No |
| 🟡 **MEDIUM** | Upgrade `utils/validation.py` (fractions, categories) | `utils/validation.py` | No |
| 🟡 **MEDIUM** | Create `utils/image_gen.py` + `scripts/generate_image.py` | New files | No (optional feature) |
| 🟡 **MEDIUM** | Create `tests/` | New directory | No |
| 🟢 **LOW** | Create `.gitignore` | New file | No |
| 🟢 **LOW** | Create `requirements-ml.txt` | New file | No |
| 🟢 **LOW** | Update `README.md` | Existing file | No |

---

## 12. Recommended Development Roadmap

### Stage 1: Foundation ✅ (Mostly done)

- **Objective:** Streamlit UI, prompt design, basic allergen/quantity validation.
- **Status:** ✅ Done (with caveats: page 2 broken, pages 3–4 missing, demo-only generation).
- **Remaining:** Fix page 2's `run()`, create placeholder pages 3–4 so the app doesn't crash.

### Stage 2: Environment + Model Files

- **Objective:** Working TF/Keras environment + trained model files present locally.
- **Actions:**
  1. Install Python 3.12 via Homebrew (`brew install python@3.12`).
  2. Create `.venv-ml` with Python 3.12.
  3. Install `tensorflow`, `keras>=3.5`, `keras-hub`, `transformers`, `torch`, `diffusers`, `streamlit`, etc.
  4. Verify `import tensorflow, keras, keras_hub` works.
  5. Export model files from Colab (using the provided scripts) and copy to `models/` and `data/outputs/`.
- **Files:** `requirements-ml.txt`, `.venv-ml/`, `.gitignore`
- **Testing:** `python scripts/check_tensorflow.py`, `ls -lh models/ data/outputs/`

### Stage 3: Port Real Code into Utils

- **Objective:** Replace all placeholder utils with the real implementations from `ANTIGRAVITY_ALL_CODE.md`.
- **Files to create:** `utils/config.py`, `utils/app_state.py`, `utils/image_gen.py`, `scripts/generate_image.py`, `scripts/smoke_test_models.py`
- **Files to replace:** `utils/prompt.py`, `utils/gpt2.py`, `utils/ner.py`, `utils/allergen.py`, `utils/validation.py`
- **Testing:** `python -m py_compile` on all files, `python -m pytest tests/`, `python scripts/smoke_test_models.py`

### Stage 4: Rewrite All Pages

- **Objective:** Pages use real models; show real output; handle missing files gracefully.
- **Files to replace:** `app/pages/1_Recipe_Generator.py`, `app/pages/2_Ingredient_Analysis.py`
- **Files to create:** `app/pages/3_Model_Comparison.py`, `app/pages/4_Model_Training.py`
- **Testing:** Launch app, test all 5 sidebar entries, verify real model output for the 5 test ingredient sets.

### Stage 5: Final Integration & Documentation

- **Objective:** Tests pass, documentation accurate, `FINAL_STATUS.md` written.
- **Files:** `tests/`, `README.md`, `ENVIRONMENT_SETUP.md`, `FINAL_STATUS.md`
- **Testing:** `pytest`, manual smoke test, review all pages.

---

## 13. Exact Commands to Run the Application

### Current state (demo only, pages 2–4 will show error boxes)

```bash
cd /Users/snehagupta/Downloads/Recipe_genai
source venv/bin/activate
streamlit run app/main.py
# Press Enter at the email prompt
# Open http://localhost:8501
```

### After full setup (future)

```bash
cd /Users/snehagupta/Downloads/Recipe_genai
source .venv-ml/bin/activate
streamlit run app/main.py
# Open http://localhost:8501
```

---

## 14. Overall Completion Estimate

| Requirement | Weight | Completion | Weighted |
|-------------|--------|------------|----------|
| U1 – Prompt design | 15% | 30% (demo prompt exists, wrong format) | 4.5% |
| U2 – GPT-2 fine-tuning | 25% | 5% (training done on Colab, but nothing local) | 1.25% |
| U3 – BERT NER + Image | 25% | 5% (comma-split placeholder, no image code) | 1.25% |
| U4 – TF2 training | 15% | 5% (TF installed but broken, no training script) | 0.75% |
| U5 – Allergens + quantities | 10% | 50% (basic keyword matching + simple validation) | 5% |
| UI/Integration | 10% | 30% (Home + Page 1 work, pages 2–4 broken/missing) | 3% |
| **Total** | **100%** | | **~16%** |

**Justification:** The Streamlit shell exists and the demo page works, but none of the five academic requirements are fully implemented locally. Training was done on Colab but the artifacts have not been transferred. The most critical blocker is the absence of model weights and a working TensorFlow environment.

---

## 15. Recommended Immediate Next Steps

1. **Fix the three broken pages** (page 2 needs `run()`, pages 3–4 need to be created) so the app at least launches without crashing. This is a 5-minute fix.
2. **Install Python 3.12** (`brew install python@3.12`) and create the `.venv-ml` environment.
3. **Export model files from Colab** using the provided scripts (`find_needed_files.py` → `package_models.py` → `copy_to_drive.py`) and copy them to `models/` and `data/outputs/`.
4. **Write all the real utility files** from `ANTIGRAVITY_ALL_CODE.md` (the complete code has already been prepared and is ready to be written file-by-file).

> ⚠️ **The complete replacement code for all files is already written in `ANTIGRAVITY_ALL_CODE.md`.** Once the model files are present and Python 3.12 is installed, the entire codebase can be updated in one pass. The code handles missing files gracefully (shows warnings instead of crashing). No training is needed – only inference with the weights trained on Colab.
