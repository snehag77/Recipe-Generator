# RecipeGen AI (P45) – Comprehensive End-to-End Application Audit

**Date of Audit:** 2026-11-16 16:19 IST  
**Auditor Roles:** Senior QA Engineer, Full-Stack Developer, AI/ML Engineer, UI/UX Reviewer, Security Auditor  
**Project Path:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Execution Environment:** macOS 26.6.2 (Darwin arm64, Apple M4 / T8132, 16 GB Unified RAM)  
**Active Python Runtimes:** `venv_tf` (Python 3.12.8), `venv` (Python 3.14.7)  

---

## 1. Executive Summary

A comprehensive, multi-perspective audit of the **RecipeGen AI** project was conducted. The application was audited across architecture, code execution, machine learning pipelines, dataset generation, UI/UX, security, and deployment readiness. **No files were modified or deleted during this audit.**

### Overall Readiness Score: **58 / 100**
* **Functionality & Pipeline Foundation:** 70 / 100 (Streamlit UI, data pipeline, TF 2 / KerasHub fine-tuning, allergen & quantity checks work).
* **AI/ML Architectural Compliance:** 50 / 100 (Fine-tuning is real, but BERT NER and Diffusion Image are missing; prompt delimiter mismatch exists).
* **Deployment & Production Readiness:** 45 / 100 (Missing locked dependencies, 1.4 GB model checkpoint, dual venv confusion, hardcoded `/tmp` paths).
* **UI/UX & Formatting:** 65 / 100 (Clear layout, but raw model token artifacts appear in UI; no deduplication on input chips).

---

## 2. Phase 1 – Application & Feature Inventory

| Component / File | Intended Role | Real Implementation Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| `app/main.py` | Main Streamlit orchestrator & navigation | **COMPLETE** | Lazy module loading, error boundaries via `try/except`, sidebar routing. |
| `app/pages/1_Recipe_Generator.py` | Core recipe generation interface | **PARTIALLY WORKING** | Connects to real GPT-2 model; triggers allergen & quantity checks. Displays raw model tags. |
| `app/pages/2_Ingredient_Analysis.py` | Ingredient entity breakdown & screening | **DEMO / MOCK** | **No BERT model**. Uses `str.split(',')`; allergen and quantity rules are real. |
| `app/pages/3_Model_Comparison.py` | Benchmark & Live Base vs FT comparison | **WORKING (Live)** | Successfully runs side-by-side inference with Base vs FT GPT-2. |
| `app/pages/4_Model_Training.py` | Training metrics & loss curves | **WORKING (Live)** | Reads `data/outputs/finetuned_metrics.json` and renders loss charts. |
| `utils/prompt.py` | U1 Prompt construction & delimiters | **DISCREPANT** | Builds `<INGREDIENTS> ... <TITLE> <SERVINGS> <PREP_TIME> <COOK_TIME> <QUANTITIES> <STEPS> <END_RECIPE>`. **Differs from training prompt format.** |
| `utils/gpt2.py` | GPT-2 model loading & generation | **WORKING** | Loads `models/gpt2_finetuned.weights.h5` (1.4 GB) and `gpt2_base_en` via KerasHub. |
| `utils/ner.py` | U3 BERT Named Entity Recognition | **NOT IMPLEMENTED** | Contains only `text.split(',')`. Zero BERT/Transformer NER code. |
| `utils/allergen.py` | U5a Allergen detection | **COMPLETE** | 9 categories + safe substitutions (peanut butter, coconut milk neutralized). |
| `utils/validation.py` | U5b Quantity & unit validation | **COMPLETE** | Handles fractions (`½`, `1 1/2`), bounds (<0, 0, >thresholds), and culinary units. |
| `scripts/prepare_recipe_dataset.py`| Data pipeline & reservoir sampling | **COMPLETE** | Sampled 5,000 recipes across 2.23M rows without bias; safe AST parsing. |
| `scripts/validate_recipe_dataset.py`| Split leakage & schema validator | **COMPLETE** | Proves zero train/val/test overlap and delimiter tag compliance. |
| `scripts/analyze_recipe_dataset.py` | Token distribution profiler | **COMPLETE** | Generates length percentiles and sequence recommendations. |
| `scripts/train_gpt2.py` | TensorFlow 2 / KerasHub fine-tuning | **COMPLETE** | Functional training script with AdamW, cosine decay, checkpointing, and perplexity. |

---

## 3. Phase 2 – Live Application Testing Results

### A. Recipe Generator Page (`1_Recipe_Generator.py`)
- **Neural Inference Authenticity:** Verified **Real Neural Model**. `gpt2.generate_recipe()` loads `models/gpt2_finetuned.weights.h5` and generates autoregressively.
- **Single Ingredient (`potato`):** Model successfully generates culinary continuations.
- **Empty Input:** Handled cleanly with an error message: `⚠️ Please provide at least one ingredient to generate a recipe.`.
- **Incompatible Ingredients (`banana, garlic, chocolate, mustard`):** Model continues generating, attempting to combine them into an experimental preparation.
- **Formatting Defect:** The output contains unclosed tag artifacts (e.g. `<IMPORTANT_SUGAR>`, `<MOISTURE>`, `<PRODUCT_NAME>`).

### B. Ingredient Analysis Page (`2_Ingredient_Analysis.py`)
- **Entity Extraction (U3):** **FAILED / MOCK.** When entering `"2 cups of diced potatoes and sweet onions"`, the system outputs `['2 cups of diced potatoes and sweet onions']`. It cannot segment food items from quantities without explicit commas.
- **Duplicate Handling:** Duplicates entered by the user (`chicken breast, garlic, chicken breast`) are not deduplicated on this page.
- **Allergen Screening:**
  - `peanut butter` -> Flags `Peanuts`, correctly avoids `Milk/Dairy` ✅
  - `cow milk, shrimp, soy sauce` -> Flags `Milk/Dairy`, `Shellfish`, `Soy`, `Wheat/Gluten` ✅
  - Hidden allergens (`casein`, `whey protein`, `edamame`, `tahini`) -> Correctly identified ✅
- **Quantity Validation:**
  - `-2 tbsp butter` -> Correctly flagged as negative ✅
  - `0 eggs` -> Correctly flagged as zero amount ✅
  - `1/2 cup milk`, `1 1/2 tsp salt`, `¾ cup sugar` -> Correctly parsed without errors ✅
  - `50,000 kg rice` -> Correctly flagged as unrealistic ✅

### C. Model Comparison Page (`3_Model_Comparison.py`)
- **Model Loading:** Both Base GPT-2 (`gpt2_base_en`) and Fine-Tuned GPT-2 load into memory via `utils/gpt2.py`.
- **Benchmark Sets:** All 5 required benchmark ingredient sets are present and can be queried.
- **Observation:** The Fine-Tuned model outputs structured ingredient lists (`- 2 tablespoons olive oil`, `- 1 teaspoon garlic powder`), whereas the Base model outputs unstructured conversational text (`<LEVEL_CITY>`, `<SUB_SIZE>100`, blog narratives).

### D. Model Training Page (`4_Model_Training.py`)
- **Metrics Display:** Successfully displays validation loss (`1.9420`), perplexity (`6.97`), sequence length (`256`), and loss progression curves from `data/outputs/finetuned_metrics.json`.
- **No Mock "Train" Buttons:** Read-only architecture prevents accidental browser freezing or heavy training on the main UI thread.

### E. Navigation and Layout
- All 4 pages load cleanly via the radio button menu.
- Streamlit layout is responsive on wide displays.
- Visual hierarchy is clear; warnings and flags use intuitive color coding (yellow for allergen caution, red for errors, green for success).

---

## 4. Phase 3 – AI/ML Implementation Audit

### 1. The Prompt Delimiter Mismatch (P0 Architectural Flaw)
There is a structural divergence between the training script and the inference utility:
- **How the dataset was built and trained (`scripts/prepare_recipe_dataset.py`):**
  ```text
  <INGREDIENTS>
  potato
  onion
  <END_ING>
  <RECIPE>
  <TITLE>
  Recipe Title
  </TITLE>
  <QUANTITIES>
  - ...
  </QUANTITIES>
  <STEPS>
  1. ...
  </STEPS>
  </RECIPE>
  ```
- **How `utils/prompt.py` constructs inference prompts:**
  ```text
  <INGREDIENTS>
  potato, onion
  <END_ING>
  <RECIPE>
  <TITLE>
  <SERVINGS> 4
  <PREP_TIME>
  <COOK_TIME>
  <QUANTITIES>
  <STEPS>
  <END_RECIPE>
  ```
- **Consequence:** `utils/prompt.py` sends closing tags `<END_RECIPE>` before the model has even generated the body, and injects `<SERVINGS>`, `<PREP_TIME>`, and `<COOK_TIME>` tags that were **never present in the RecipeNLG training corpus**. This confuses the model, causing tag hallucinations and unclosed XML tags.

### 2. Under-Fitting / Training Scale
- The current checkpoint (`models/gpt2_finetuned.weights.h5`) was trained on **20 samples for 1 epoch** (the smoke-test pilot).
- While the training code is fully functional, 20 samples is insufficient for the 124M-parameter GPT-2 model to master coherent recipe generation. It must be trained on the full 4,500 prepared training set (`data/processed/recipes_train.jsonl`).

### 3. Missing Academic Components (U3)
- **BERT NER:** The assignment requires BERT-based token classification to tag ingredients. `utils/ner.py` currently uses basic comma-splitting.
- **Diffusion Dish Image:** There is no image generation code or Stable Diffusion pipeline integrated into the UI.

---

## 5. Phase 4 – Dataset Pipeline Audit

- **Input File:** `data/archive 2/RecipeNLG_dataset.csv` (2.29 GB, 2,231,142 records).
- **Sampling Algorithm:** Reproducible Reservoir Sampling (Algorithm R, seed 42) completed across all 2.23M rows.
- **Deduplication:** SHA-256 fingerprinting on normalized title and ingredients dropped 148,876 duplicates.
- **Split Distribution (90:5:5):**
  - Train: 4,500 records (`data/processed/recipes_train.jsonl`)
  - Validation: 250 records (`data/processed/recipes_validation.jsonl`)
  - Test: 250 records (`data/processed/recipes_test.jsonl`)
- **Leakage Verification:** `scripts/validate_recipe_dataset.py` proved **0 cross-split duplicates**.
- **Dataset Audit Status:** **100% COMPLETE & PASS.**

---

## 6. Phase 5 – Security, Environment & Deployment Readiness

### 1. Dual Virtual Environment Conflict (P0 Deployment Issue)
- The project has two separate virtual environments:
  - `venv`: Created with Python 3.14. Has `streamlit`, but TensorFlow fails to import.
  - `venv_tf`: Created with Python 3.12. Has `tensorflow 2.21`, `keras 3.15`, `keras-hub 0.33`, and `streamlit 1.67`.
- **Risk:** If a user or automated deploy script runs `source venv/bin/activate` instead of `source venv_tf/bin/activate`, the app will crash with `ImportError: tensorflow`.

### 2. `requirements.txt` Incompleteness
- Current `requirements.txt`:
  ```
  streamlit>=1.30
  pillow>=10.2.0
  ```
- **Risk:** Completely misses `tensorflow`, `keras`, `keras-hub`, `numpy`, and `pandas`. Deploying to Streamlit Cloud or Docker will fail immediately.

### 3. Hardcoded `/tmp/.keras` Path
- In `utils/gpt2.py` and `scripts/check_tensorflow.py`, `os.environ["KERAS_HOME"] = "/tmp/.keras"` is hardcoded to bypass macOS sandbox file permissions. On standard Linux/Cloud servers, this can cause permission collisions if `/tmp` is shared.

### 4. Repository & Storage Footprint
- `data/archive 2/` is **2.1 GB**.
- `models/gpt2_finetuned.weights.h5` is **1.4 GB**.
- Git repositories (like GitHub) enforce a strict 100 MB per-file limit. Pushing `gpt2_finetuned.weights.h5` directly to GitHub will be rejected unless Git LFS is configured.

---

## 7. Phase 6 – UX & Product Quality Review

1. **Raw Tag Leakage in UI:** The generated recipe code block in `1_Recipe_Generator.py` prints the raw XML/HTML tags directly (`<TITLE>`, `<QUANTITIES>`, `</RECIPE>`) instead of parsing them into clean Markdown headers and bullet lists.
2. **Missing Token Streaming:** Generation takes 5–8 seconds on CPU. The UI freezes with a basic spinner instead of streaming tokens, making the app feel slow to first-time users.
3. **No Clear Labeling on Page 2:** Page 2 is named "Ingredient Analysis (BERT NER)", but displays a rule-based comma-split output, which could cause confusion during academic grading.
4. **Input Deduplication:** If a user types `"potato, onion, potato"`, the prompt builder should deduplicate the input list automatically.

---

## 8. Phase 7 – Test Matrix & Bug List

| Test ID | Component | Test Performed | Result | Severity | Evidence |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **TM-01** | Prompt | Check delimiter consistency with training data | **FAIL** | **P0** | Prompt uses `<SERVINGS>`, training data does not |
| **TM-02** | ML Model | Checkpoint fine-tuning scale | **PARTIAL** | **P1** | Checkpoint is functional but only trained on 20 samples |
| **TM-03** | NER | BERT token classification for ingredients | **NOT IMPLEMENTED** | **P1** | `utils/ner.py` is string splitting only |
| **TM-04** | Image Gen | Diffusion dish image generation | **NOT IMPLEMENTED** | **P2** | No diffusion pipeline present |
| **TM-05** | Packaging | `requirements.txt` dependencies | **FAIL** | **P0** | Missing TensorFlow, Keras, KerasHub |
| **TM-06** | Allergen | False positive neutralization | **PASS** | — | `peanut butter` did not flag dairy |
| **TM-07** | Allergen | True positive multi-detection | **PASS** | — | Detected dairy, soy, wheat, shellfish |
| **TM-08** | Validation | Fractional quantity parsing | **PASS** | — | Parsed `½`, `1 1/2`, `¾` accurately |
| **TM-09** | Validation | Negative and zero amounts | **PASS** | — | Flagged `-2 tbsp` and `0 eggs` |
| **TM-10** | Dataset | Zero cross-split duplicate leakage | **PASS** | — | 0 duplicates across 5,000 samples |
| **TM-11** | UI | Error handling on empty input | **PASS** | — | Clean warning banner displayed |
| **TM-12** | UI | Model comparison live execution | **PASS** | — | Runs both Base and FT models |

---

## 9. Answers to Critical Final Questions

### 1. Is the application actually functional?
**Yes.** The Streamlit application launches without errors, the navigation works across all 4 pages, input forms accept ingredients, allergen and quantity rules execute cleanly, and the GPT-2 model generates autoregressive recipe text.

### 2. Which features are real and which are placeholders?
- **Real:**
  - Streamlit UI shell and page navigation.
  - Allergen detection engine and safe-substitution rules.
  - Quantity parsing and measurement validation.
  - Dataset reservoir sampling, deduplication, and JSONL splitting (4,500 / 250 / 250).
  - TensorFlow 2 / KerasHub GPT-2 base model loading and causal generation.
  - Training metrics display on Page 4.
- **Placeholders / Incomplete:**
  - **BERT NER:** Currently just string splitting on commas.
  - **Diffusion Image Generation:** Completely missing.
  - **Model Training Scale:** The model checkpoint is real, but was only trained on a 20-sample pilot.

### 3. What is currently broken?
- **Prompt Format Discrepancy:** The prompt sent by `utils/prompt.py` does not match the prompt format used during training in `scripts/prepare_recipe_dataset.py`, causing tag confusion.
- **`requirements.txt`:** Missing all core ML packages.

### 4. What is missing?
- BERT Named Entity Recognition model (`bert_base_en_uncased`).
- Latent Diffusion dish visualization pipeline.
- Production-scale fine-tuning on the full 4,500 training dataset.
- Output post-processor to parse raw `<TAGS>` into styled Markdown.

### 5. What could fail during deployment?
- Installing dependencies via `pip install -r requirements.txt` will fail to install TensorFlow and KerasHub.
- Pushing to GitHub will fail because `models/gpt2_finetuned.weights.h5` is 1.4 GB (exceeds GitHub's 100 MB file limit).
- Cloud instances with less than 4 GB RAM will crash with Out-Of-Memory (OOM) when loading the 124M parameter GPT-2 model.

### 6. What must be fixed before deployment?
1. Update `requirements.txt` to include `tensorflow`, `keras>=3.5`, `keras-hub`, `numpy`, and `pandas`.
2. Configure Git LFS or external model storage for `models/gpt2_finetuned.weights.h5`.
3. Align `utils/prompt.py` to match the exact delimiter format of the training dataset.

### 7. What should be improved before academic project submission?
1. Fine-tune GPT-2 on the full 4,500 recipe dataset for 2–3 epochs to achieve high-quality generation.
2. Implement actual BERT NER in `utils/ner.py` using KerasHub (`BertBackbone`).
3. Parse the generated tags into formatted Markdown sections (Title, Ingredients, Steps) in the UI.

### 8. What is the recommended next development step?
**Align `utils/prompt.py` with the dataset delimiter structure, then launch the full fine-tuning run on `data/processed/recipes_train.jsonl` (4,500 recipes) using `scripts/train_gpt2.py`.**
