# RecipeGen AI — Stage 2I: Phase 1 Critical Bug Fixes Report

**Date:** 2026-11-16  
**Auditor & Engineer:** Antigravity AI  
**Project Path:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Execution Environment:** `venv_tf` (Python 3.12.8, TensorFlow 2.21.0, Keras 3.15.1, KerasHub 0.33.0, Streamlit 1.67.0)  
**Status:** All 4 targeted defects resolved and verified with 10 automated unit tests. All model checkpoints and datasets remain untouched.

---

## 1. Executive Summary

In accordance with Phase 1 instructions, the four critical defects identified in `STAGE2H_FULL_PROJECT_AUDIT_REPORT.md` have been fixed and verified:
1. **Fraction Parsing Bug:** Resolved greedy integer matching so standalone and mixed fractions (`1/2`, `3/4`, `1 1/2`, `½`) parse accurately to floats (`0.5`, `0.75`, `1.5`).
2. **Plural Measurement Unit Validation Bug:** Implemented unit normalization mapping plural culinary units (`cups`, `teaspoons`, `grams`, etc.) to canonical keys in `UPPER_BOUNDS` so bounds checks trigger consistently (`50 cups salt` now triggers an upper-bound warning).
3. **Allergen Negation False Positives:** Added declared "free-from" claim recognition (`gluten-free flour`, `dairy-free cheese`, `egg-free mayo`) while ensuring `lactose-free milk` correctly retains its `Milk/Dairy` warning and explicit contradictions are flagged.
4. **Model Training Page Configuration Display:** Corrected nested dictionary extraction from `metrics["config"]` in `app/pages/4_Model_Training.py`, eliminating `None` displays for learning rate and batch size and enabling multi-run selection (Stage 2F Full Run vs. Stage 2D Pilot Run).

---

## 2. File Modification & Backup Registry

| Modified File | Backup Copy Location | MD5 (Original Backup) | MD5 (Updated File) |
| :--- | :--- | :--- | :--- |
| [`utils/validation.py`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/validation.py) | `utils/validation.py.bak` | `9d554a9c680653ffcbce4c330c6fa644` | `be2690ff0e206fe518973686cf4b47a9` |
| [`utils/allergen.py`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/allergen.py) | `utils/allergen.py.bak` | `ca0cb646702c2e557262f3a6122d1ca6` | `bb56f1a8cbeab3feeb823cf30b91e779` |
| [`app/pages/4_Model_Training.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/4_Model_Training.py) | `app/pages/4_Model_Training.py.bak` | `a90d805825bb6697a29a435889a7f34c` | `1c8280f2d3ec6fb9e9f6530a6ba292f7` |

---

## 3. Root Cause Analysis & Changes Made

### Defect 1: Standalone Fraction Parsing Bug
- **Root Cause:** In `utils/validation.py`, `match_single` used regex `r"^(\d+(?:\.\d+)?|\d+/\d+)\s*(.*)$"`. Because `\d+(?:\.\d+)?` preceded `\d+/\d+`, the input `"1/2"` greedily matched `"1"` as the quantity, leaving `"/2"` in the remainder. This corrupted the numerical value (evaluating to `1.0`) and caused unit parsing to fail because `"/2"` was not in `VALID_UNITS`.
- **Change Made:**
  1. Reordered the regex alternation: `r"^(\d+/\d+|\d+(?:\.\d+)?)\s*(.*)$"`, ensuring fractions are matched first.
  2. Mapped Unicode fraction characters (`½`, `¼`, `¾`, etc.) to clean fraction string representations (`" 1/2 "`, `" 3/4 "`) so mixed numbers like `"1 ½ cups"` resolve seamlessly to mixed fraction pattern `1 1/2`.

---

### Defect 2: Plural Measurement Unit Validation Bug
- **Root Cause:** `UPPER_BOUNDS` only defined singular keys (`"cup": 15.0`, `"tsp": 60.0`, `"g": 5000.0`), while `VALID_UNITS` accepted plural forms (`"cups"`, `"teaspoons"`, `"grams"`). When a user entered `"50 cups salt"`, `UPPER_BOUNDS.get("cups")` returned `None`, falling back to `default_count = 50.0`. Since `50.0 <= 50.0`, the system generated **no warning**.
- **Change Made:**
  1. Implemented a comprehensive `UNIT_NORMALIZATION` dictionary mapping all singular, plural, and abbreviated unit variants to their canonical singular form (e.g., `"cups"` $\to$ `"cup"`, `"teaspoons"` $\to$ `"tsp"`).
  2. Normalized units prior to checking `UPPER_BOUNDS`.
  3. Added `NON_CULINARY_UNITS` detection (`kilometer`, `meter`, `volt`, etc.) to explicitly flag non-culinary inputs with a helpful warning.

---

### Defect 3: Allergen Negation False Positives
- **Root Cause:** `utils/allergen.py` relied on naive substring/word matching. The substring `"flour"` unconditionally triggered `Wheat/Gluten`, and `"cheese"` unconditionally triggered `Milk/Dairy`, ignoring explicit declared claims like `"gluten-free flour"` or `"dairy-free cheese"`.
- **Change Made:**
  1. Added `EXPLICIT_FREE_CLAIMS` mapping regular expressions for declared free-from claims (`gluten-free`, `wheat-free`, `dairy-free`, `non-dairy`, `egg-free`, `vegan`).
  2. **Medical Safety Constraint:** Deliberately excluded `"lactose-free"` from dairy suppression. Lactose-free cow's milk still contains milk proteins (casein, whey), so `"lactose-free milk"` continues to trigger `Milk/Dairy`.
  3. Added `CONTRADICTION_KEYWORDS`: If an ingredient contains an explicit contradictory allergen (e.g. `"gluten-free flour with wheat"` or `"dairy-free cheese with whey"`), the allergen warning is strictly preserved.

---

### Defect 4: Model Training Page Configuration Display Bug
- **Root Cause:** In `app/pages/4_Model_Training.py`, the code attempted `metrics.get("learning_rate")` and `metrics.get("batch_size")`. In both `finetuned_metrics.json` and `recipe_gpt2_full_metrics.json`, these hyperparameters are stored under the nested `"config"` object (`metrics["config"]["initial_learning_rate"]`, `metrics["config"]["batch_size"]`). Consequently, the UI rendered `- **Learning Rate:** None` and `- **Batch Size:** None`.
- **Change Made:**
  1. Extracted hyperparameters safely from `metrics.get("config", {})` with fallback to top-level keys.
  2. Added fallback text `"Not recorded"` so `None` is never displayed.
  3. Added discovery for all available runs (`recipe_gpt2_full_metrics.json` and `finetuned_metrics.json`), presenting a selection dropdown when multiple runs exist so users can inspect both the Stage 2F full run (4,500 recipes) and the Stage 2D pilot run (200 recipes).

---

## 4. Before-and-After Test Results

### Quantity & Unit Validation Comparison

| Test Input | Before Fix (Parsed / Warning) | After Fix (Parsed / Warning) | Status |
| :--- | :--- | :--- | :---: |
| `1/2 tsp salt` | `(1.0, None)` \| No warning | **`(0.5, 'tsp')` \| No warning** | **RESOLVED** |
| `3/4 cup milk` | `(3.0, None)` \| No warning | **`(0.75, 'cup')` \| No warning** | **RESOLVED** |
| `1 1/2 cups flour` | `(1.5, 'cups')` \| No warning | **`(1.5, 'cups')` \| No warning** | **PASS** |
| `2 1/4 tsp sugar` | `(2.25, 'tsp')` \| No warning | **`(2.25, 'tsp')` \| No warning** | **PASS** |
| `½ tsp salt` | `(0.5, 'tsp')` \| No warning | **`(0.5, 'tsp')` \| No warning** | **PASS** |
| `¾ cup milk` | `(0.75, 'cup')` \| No warning | **`(0.75, 'cup')` \| No warning** | **PASS** |
| `0 cups flour` | `(0.0, 'cups')` \| "Zero quantity" | **`(0.0, 'cups')` \| "Zero quantity"** | **PASS** |
| `-2 cups sugar` | `(-2.0, 'cups')` \| "Negative quantity" | **`(-2.0, 'cups')` \| "Negative quantity"** | **PASS** |
| `50 cups salt` | `(50.0, 'cups')` \| **No warning (Bypassed)** | **`(50.0, 'cups')` \| "Unusually large quantity (50 cups)"** | **RESOLVED** |
| `30 teaspoons salt` | `(30.0, 'teaspoons')` \| No warning | **`(30.0, 'teaspoons')` \| No warning (Safe)** | **PASS** |
| `100 grams salt` | `(100.0, 'grams')` \| No warning | **`(100.0, 'grams')` \| No warning (Safe)** | **PASS** |
| `5000 ml oil` | `(5000.0, 'ml')` \| No warning | **`(5000.0, 'ml')` \| No warning (Safe)** | **PASS** |
| `1000 kg chicken` | `(1000.0, 'kg')` \| "Unusually large quantity" | **`(1000.0, 'kg')` \| "Unusually large quantity"** | **PASS** |
| `2 kilometers milk`| `(2.0, None)` \| No warning | **`(2.0, 'kilometers')` \| "Non-culinary measurement unit"** | **RESOLVED** |

---

### Allergen Detection Comparison

| Test Input | Expected Alert | Before Fix Output | After Fix Output | Status |
| :--- | :--- | :--- | :--- | :---: |
| `gluten-free flour` | None | `['Wheat/Gluten']` (False Positive) | **`[]` (Suppressed by declared claim)** | **RESOLVED** |
| `dairy-free cheese` | None | `['Milk/Dairy']` (False Positive) | **`[]` (Suppressed by declared claim)** | **RESOLVED** |
| `lactose-free milk` | Milk/Dairy | `['Milk/Dairy']` | **`['Milk/Dairy']` (Preserved: still dairy)** | **PASS** |
| `wheat flour` | Wheat/Gluten | `['Wheat/Gluten']` | **`['Wheat/Gluten']`** | **PASS** |
| `regular milk` | Milk/Dairy | `['Milk/Dairy']` | **`['Milk/Dairy']`** | **PASS** |
| `peanut butter` | Peanuts | `['Peanuts']` | **`['Peanuts']`** | **PASS** |
| `almond flour` | Tree Nuts | `['Tree Nuts']` | **`['Tree Nuts']`** | **PASS** |
| `soy sauce` | Soy, Wheat/Gluten | `['Soy', 'Wheat/Gluten']` | **`['Soy', 'Wheat/Gluten']`** | **PASS** |
| `butter` | Milk/Dairy | `['Milk/Dairy']` | **`['Milk/Dairy']`** | **PASS** |
| `egg-free mayonnaise` | None | `['Eggs']` (False Positive) | **`[]` (Suppressed by declared claim)** | **RESOLVED** |
| `gluten-free flour with wheat` | Wheat/Gluten | `['Wheat/Gluten']` | **`['Wheat/Gluten']` (Contradiction flagged)** | **PASS** |
| `dairy-free cheese with whey` | Milk/Dairy | `['Milk/Dairy']` | **`['Milk/Dairy']` (Contradiction flagged)** | **PASS** |

---

### Model Training Page Display Comparison

| Display Element | Before Fix | After Fix |
| :--- | :--- | :--- |
| **Learning Rate** | `- **Learning Rate:** None` | **`- **Learning Rate Schedule:** CosineDecayWithWarmup (initial lr: 5e-05)`** |
| **Batch Size** | `- **Batch Size:** None` | **`- **Batch Size:** 4`** |
| **Sequence Length** | `384 tokens` (Hardcoded fallback) | **`512 tokens` (Dynamic from config)** |
| **Run History Selection** | Single fixed file | **Dropdown selector: Stage 2F Full Dataset Run vs. Stage 2D Pilot Run** |
| **Compute Device** | Not displayed | **`Apple Silicon CPU (TensorFlow 2.21)`** |

---

## 5. Automated Regression Test Execution

A dedicated unit test suite was implemented at [`tests/test_phase1_fixes.py`](file:///Users/snehagupta/Downloads/Recipe_genai/tests/test_phase1_fixes.py).

### Execution Command:
```bash
./venv_tf/bin/python -m unittest tests/test_phase1_fixes.py
```

### Execution Output:
```text
..........
----------------------------------------------------------------------
Ran 10 tests in 0.001s

OK
```

All 10 tests passed with zero failures.

---

## 6. Streamlit Page Health & Regression Verification

The Streamlit application was verified across all four pages using `app/main.py`:
- `Home`: Loads cleanly.
- `Recipe Generator`: Verified with fractions (`1/2 tsp salt`, `3/4 cup milk`) and plural units (`50 cups salt`). Parsing and warning cards render properly.
- `Ingredient Analysis`: Verified with `"gluten-free flour"` (returns 0 allergens) and `"lactose-free milk"` (returns Milk/Dairy).
- `Model Comparison`: Side-by-side comparison executes properly.
- `Model Training`: Correctly displays `initial_lr: 5e-05`, `batch_size: 4`, and `max_len: 512` without any `None` values.

---

## 7. Model Checkpoint & Dataset Integrity Confirmation

All checkpoints and dataset splits were re-verified via MD5 checksums:

| File | Type | Verified MD5 Checksum | Status |
| :--- | :--- | :--- | :---: |
| `models/gpt2_finetuned.weights.h5` | Pilot Checkpoint | `a0ddb523b21e820c2fdcc871f2fae2a1` | **Unchanged** |
| `models/recipe_gpt2/recipe_gpt2_best.weights.h5` | Stage 2D Checkpoint | `0f733e894f4a83087164dfc1b7fe9a9a` | **Unchanged** |
| `models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5` | Stage 2F Checkpoint | `d95eb61d471f183a67a0e7fb920369d0` | **Unchanged** |
| `data/processed/recipes_train.jsonl` | Training Set (4,500) | `6d2487aa1719146437892318f5820dc9` | **Unchanged** |
| `data/processed/recipes_validation.jsonl` | Validation Set (250) | `01b7bf6c2d0ee71093f38e01c000a045` | **Unchanged** |
| `data/processed/recipes_test.jsonl` | Held-Out Test Set (250) | `11b581b58e352672e9ef2f7130cd10bb` | **Unchanged** |

---

## 8. Remaining Limitations (Out-of-Scope for Phase 1)

As strictly mandated by the Phase 1 scope instructions:
1. `utils/gpt2.py` has not been modified; connecting the Stage 2F checkpoint to the UI is reserved for Phase 2.
2. `requirements.txt` has not been modified; deployment dependency updates are reserved for subsequent phases.
3. BERT NER and Diffusion Image Generation remain unaddressed, awaiting Phase 3/4 authorization.

---

*Phase 1 is complete. Awaiting user review and authorization before proceeding to Phase 2.*
