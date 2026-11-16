# RecipeGen AI — Stage 2K: Application Correction & UX Enhancement Report

**Date:** 2026-11-16  
**Auditor & Implementation Engineer:** Antigravity AI  
**Project Path:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Target Environment:** `venv_tf` (Python 3.12.8, TensorFlow 2.21.0, Keras 3.15.1, KerasHub 0.33.0, Streamlit 1.67.0)  
**Status:** Completed. All functional defects resolved, UI presentation elevated, Stage 2G test metrics dynamically integrated, and 100% test pass rate achieved.

---

## 1. Executive Summary

Stage 2K resolved all functional defects and UX shortcomings identified in the application audit, strictly preserving the trained model weights and dataset integrity:

1. **Fixed Ingredient Parsing (P0):** [`utils/ner.py`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/ner.py) was rewritten to support comma-separated, newline-separated, bullet-prefixed (`-`, `*`, `•`), numbered (`1.`), and mixed delimiter inputs. Case-insensitive deduplication and whitespace cleanup are now applied uniformly across Pages 1, 2, and 3.
2. **Dual-Tab Recipe Output View (P0):** [`app/pages/1_Recipe_Generator.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/1_Recipe_Generator.py) now provides two distinct tabs:
   - **Formatted Recipe:** Renders title in clean headers, quantities in itemized lists, and directions in numbered steps using [`utils/prompt.py`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/prompt.py). If tags are absent or output is partial, an advisory notice is shown and text is preserved.
   - **Raw Generated Text:** Full unmodified model output with delimiters preserved for technical inspection.
3. **Interactive Ingredient Management on Page 2:** [`app/pages/2_Ingredient_Analysis.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/2_Ingredient_Analysis.py) now provides interactive multiselect chip removal, dynamic re-screening of allergens and quantities, and one-click synchronization back to [`st.session_state["ingredients"]`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/1_Recipe_Generator.py). The outdated banner was updated to accurately state: *"Current Mode: Rule-Based Ingredient Extraction. BERT-based Ingredient NER remains a planned enhancement."*
4. **Safety Preflight & Progress Telemetry on Page 3:** [`app/pages/3_Model_Comparison.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/3_Model_Comparison.py) was updated to pass custom inputs through `ner.extract_ingredients()`, evaluate allergen and quantity safety checks before generation, and display two-step sequential inference spinners (`Step 1/2: Base Model`, `Step 2/2: Fine-Tuned Model`) with explicit Apple Silicon CPU duration notices.
5. **Stage 2G Held-Out Test Evaluation Section on Page 4:** [`app/pages/4_Model_Training.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/4_Model_Training.py) now dynamically loads and displays held-out evaluation benchmarks from [`data/outputs/stage2g_teacher_forced.json`](file:///Users/snehagupta/Downloads/Recipe_genai/data/outputs/stage2g_teacher_forced.json) and [`data/outputs/stage2g_generation_eval.json`](file:///Users/snehagupta/Downloads/Recipe_genai/data/outputs/stage2g_generation_eval.json) across **250 held-out test recipes**:
   - **Test Loss:** `0.8855` (Teacher-Forced, vs `1.3618` Base)
   - **Test Perplexity:** `2.42` (Teacher-Forced, vs `3.90` Base)
   - **Structural Tag Validity:** `90.4%` (vs `0.0%` Base)
   - **Ingredient Recall in Body:** `89.21%` (vs `4.11%` Base)
   - Replaced flat single-point line charts with a structured Cross-Split Generalization table comparing Training vs Validation vs Test.
6. **Polished App Shell & Home Page:** [`app/main.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/main.py) was enhanced with an architecture pipeline diagram, accurate technology stack disclosure, 4 quick navigation shortcut action cards, clean TensorFlow initialization (removed dead `import utils.config` attempt), and page-specific browser window titles (`RecipeGen AI — <Page>`).
7. **Zero Model or Dataset Weights Altered:** All 3 `.weights.h5` checkpoint files and all 3 `.jsonl` dataset splits remain 100% byte-for-byte identical to pre-audit states (MD5 checksums verified).

---

## 2. Inventory of Modified Files and Backups

| Modified File | Backup File | Summary of Changes |
| :--- | :--- | :--- |
| [`utils/ner.py`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/ner.py) | [`utils/ner.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/ner.py.bak) | Replaced comma-only split with regex delimiter splitting (`re.split(r'[\n,;]+')`), bullet stripping (`-*•`), whitespace trimming, and deduplication. |
| [`app/pages/1_Recipe_Generator.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/1_Recipe_Generator.py) | [`app/pages/1_Recipe_Generator.py.stage2k.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/1_Recipe_Generator.py.stage2k.bak) | Added dual-tab presentation (`Formatted Recipe` vs `Raw Generated Text`), tag parsing, and partial-generation safety fallback. |
| [`app/pages/2_Ingredient_Analysis.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/2_Ingredient_Analysis.py) | [`app/pages/2_Ingredient_Analysis.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/2_Ingredient_Analysis.py.bak) | Added interactive multiselect chip removal, one-click generator synchronization, and updated roadmap banner copy. |
| [`app/pages/3_Model_Comparison.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/3_Model_Comparison.py) | [`app/pages/3_Model_Comparison.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/3_Model_Comparison.py.bak) | Replaced manual comma splitting with `ner.extract_ingredients()`, added allergen/quantity screening on custom inputs, and improved CPU inference feedback. |
| [`app/pages/4_Model_Training.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/4_Model_Training.py) | [`app/pages/4_Model_Training.py.stage2k.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/4_Model_Training.py.stage2k.bak) | Dynamically loads and displays Stage 2G held-out test evaluation benchmarks across 250 test recipes; replaced 1-point chart with cross-split table. |
| [`app/main.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/main.py) | [`app/main.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/app/main.py.bak) | Added architectural pipeline diagram, technology stack, 4 navigation action cards, removed dead import, and implemented dynamic browser tab titles. |

---

## 3. Detailed Task Implementation Review

### 3.1 Task 1: Robust Ingredient Parsing (`utils/ner.py`)
- **Root Cause of Defect:** Previous implementation consisted of an 8-line function using `text.split(',')`. Any multiline input without commas was returned as a single joined block (e.g., `['potato\nonion\ntomato']`).
- **Solution:** Implemented regex alternation `re.split(r"[\n,;]+", text)` combined with prefix cleaning `re.sub(r"^\s*(?:[-*•–—]|\d+[.)])\s*", "", token).strip()`.
- **Verified Behavior:**
  - `potato\nonion\ntomato` $\rightarrow$ `["potato", "onion", "tomato"]`
  - `- 2 potatoes\n* 1 onion\n• 3 tomatoes` $\rightarrow$ `["2 potatoes", "1 onion", "3 tomatoes"]`
  - `potato\nonion, tomato\ncumin` $\rightarrow$ `["potato", "onion", "tomato", "cumin"]`
  - Whitespace-only lines and duplicate ingredients are cleanly filtered.

### 3.2 Task 2: Formatted and Raw Recipe Output (`1_Recipe_Generator.py`)
- **Formatted Tab:** Leverages [`prompt.parse_recipe_output()`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/prompt.py) to extract `<TITLE>`, `<QUANTITIES>`, and `<STEPS>`. Renders title in `###`, quantities in bullet points, and directions as numbered steps.
- **Raw Generated Text Tab:** Displays unmodified generation inside `st.code(..., language="markdown")`, preserving all tags (`<RECIPE>`, `</STEPS>`, etc.) for auditing.
- **Fail-Safe Fallback:** If output is unstructured (such as Base GPT-2 rambling), it issues a notice and displays the text directly without dropping content.

### 3.3 Task 3: Interactive Ingredient Analysis (`2_Ingredient_Analysis.py`)
- **Interactive Chip Management:** Uses Streamlit's `st.multiselect` populated with extracted tokens. Users can click `x` on any ingredient chip to remove it.
- **Dynamic Re-Evaluation:** When chips are removed, allergen screening and quantity bounds immediately recalculate on the active list.
- **Generator Synchronization:** Added button *"🔄 Sync Selection to Recipe Generator"* which updates `st.session_state["ingredients"]` so changes propagate directly to Page 1.
- **Roadmap Banner:** Updated to explicitly state:
  > *"Current Mode: Rule-Based Ingredient Extraction. BERT-based Ingredient NER remains a planned enhancement."*

### 3.4 Task 4: Model Comparison Safety & Usability (`3_Model_Comparison.py`)
- **Shared Parser:** Tab 2 now processes custom inputs through `ner.extract_ingredients(custom_input)`.
- **Preflight Safety:** Detects allergens and abnormal quantities on custom inputs before initiating generation, displaying advisory notices and disclaimers.
- **Inference Feedback:** Clarifies CPU inference requirements (~8–14s) and updates status dynamically:
  - `⏳ Step 1/2: Generating with Base Model...`
  - `⏳ Step 2/2: Generating with Fine-Tuned Model...`

### 3.5 Task 5: Dynamic Held-Out Test Metrics (`4_Model_Training.py`)
- **Dynamic Data Source:** Directly reads [`stage2g_teacher_forced.json`](file:///Users/snehagupta/Downloads/Recipe_genai/data/outputs/stage2g_teacher_forced.json) and [`stage2g_generation_eval.json`](file:///Users/snehagupta/Downloads/Recipe_genai/data/outputs/stage2g_generation_eval.json).
- **Displayed Benchmarks (250 Held-Out Recipes):**
  - Teacher-Forced Test Loss: `0.8855` (Base: `1.3618`)
  - Teacher-Forced Test Perplexity: `2.42` (Base: `3.90`)
  - Usable Structural Validity: `90.4%` (Base: `0.0%`)
  - Recipe Body Ingredient Recall: `89.21%` (Base: `4.11%`)
  - Repetition Rate: `4.0%` (Base: `60.8%`)
- **Chart Replacement:** Replaced the uninformative 1-point line chart with a Cross-Split Generalization table comparing Training vs. Validation vs. Held-Out Test, proving the model did not overfit.

### 3.6 Tasks 6 & 7: Home Page & Dynamic App Shell (`app/main.py`)
- **Clean Environment Init:** Removed dead `import utils.config` statement; `os.environ.setdefault("KERAS_BACKEND", "tensorflow")` is set directly.
- **Browser Window Titles:** Browser tab dynamically reads `RecipeGen AI — <Page Name>`.
- **Landing Page Contents:**
  - High-level project summary and motivation.
  - Text-based architectural diagram of the 5-stage inference pipeline.
  - Implemented technology stack breakdown (Python, Streamlit, TensorFlow, Keras, KerasHub, Fine-Tuned GPT-2, Rule-Based NER).
  - 4 one-click quick navigation cards routing to each subpage.

---

## 4. Verification Test Suite & Results

All tests were executed using `./venv_tf/bin/python` under standard Apple Silicon CPU mode:

| Test Suite | File / Scope | Tests Ran | Result |
| :--- | :--- | :---: | :---: |
| **Ingredient Parser Unit Tests** | [`tests/test_ingredient_parser.py`](file:///Users/snehagupta/Downloads/Recipe_genai/tests/test_ingredient_parser.py) | 9 | **PASSED (0.000s)** |
| **Phase 1 Safety & Bounds Tests** | [`tests/test_phase1_fixes.py`](file:///Users/snehagupta/Downloads/Recipe_genai/tests/test_phase1_fixes.py) | 10 | **PASSED (0.001s)** |
| **Stage 2K Enhancement Suite** | [`tests/test_stage2k_enhancements.py`](file:///Users/snehagupta/Downloads/Recipe_genai/tests/test_stage2k_enhancements.py) | 7 | **PASSED (6.87s)** |
| **Combined Regression Suite** | `tests/test_ingredient_parser.py + test_phase1_fixes.py + test_stage2k_enhancements.py` | 26 | **PASSED (6.82s)** |
| **Phase 2 Model Inference Suite** | [`tests/test_phase2_integration.py`](file:///Users/snehagupta/Downloads/Recipe_genai/tests/test_phase2_integration.py) | 5 | **PASSED (14.74s)** |

### Checkpoint Integrity Verification (MD5 Checksums)

| Checkpoint Path | Expected MD5 | Actual MD5 | Status |
| :--- | :---: | :---: | :---: |
| `models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5` | `d95eb61d471f183a67a0e7fb920369d0` | `d95eb61d471f183a67a0e7fb920369d0` | **MATCH (Unmodified)** |
| `models/gpt2_finetuned.weights.h5` | `a0ddb523b21e820c2fdcc871f2fae2a1` | `a0ddb523b21e820c2fdcc871f2fae2a1` | **MATCH (Unmodified)** |
| `models/recipe_gpt2/recipe_gpt2_best.weights.h5` | `0f733e894f4a83087164dfc1b7fe9a9a` | `0f733e894f4a83087164dfc1b7fe9a9a` | **MATCH (Unmodified)** |

### Dataset Split Integrity Verification (MD5 Checksums)

| Dataset Split Path | Expected MD5 | Actual MD5 | Status |
| :--- | :---: | :---: | :---: |
| `data/processed/recipes_train.jsonl` (4,500 rec) | `6d2487aa1719146437892318f5820dc9` | `6d2487aa1719146437892318f5820dc9` | **MATCH (Unmodified)** |
| `data/processed/recipes_validation.jsonl` (250 rec) | `01b7bf6c2d0ee71093f38e01c000a045` | `01b7bf6c2d0ee71093f38e01c000a045` | **MATCH (Unmodified)** |
| `data/processed/recipes_test.jsonl` (250 rec) | `11b581b58e352672e9ef2f7130cd10bb` | `11b581b58e352672e9ef2f7130cd10bb` | **MATCH (Unmodified)** |

---

## 5. Unresolved Defects & Phase 4 Scope

With Stage 2K complete, the application layer is stable, user-friendly, and technically aligned. The remaining architectural requirements identified in the audit report are:

1. **Requirement U3 — BERT NER Integration:**
   Ingredient extraction is currently rule-based regex parsing. Requirement U3 specifies a BERT-based Named Entity Recognition model for extracting ingredient spans from natural language prose.
2. **Requirement U3 — Diffusion Dish Image Generation:**
   Dish visual generation remains unfulfilled (no diffusion pipeline or image endpoint is currently connected).

---

## 6. Antigravity Certification

I certify that:
- No model checkpoint weights were modified, retrained, or overwritten.
- All processed training, validation, and test datasets remain untouched.
- All 5 Streamlit pages render cleanly and navigate without errors.
- The active Streamlit service is running at `http://localhost:8503` (task `task-1451`).
- No deployment or GitHub push actions were executed.
