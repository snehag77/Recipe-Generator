# RecipeGen AI — Stage 2K: Final Discrepancy & Verification Report

**Project Location:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Execution Environment:** `venv_tf` (Python 3.12.8, Apple Silicon CPU)  
**Date:** November 16, 2026  
**Audit Mode:** Read-Only Verification  

---

## 1. Executive Summary

This investigation resolves the discrepancies observed during the Stage 2K acceptance audit:
1. **Structural Validity (90.4% vs. 100.0%):** The actual code (`app/pages/4_Model_Training.py`), the underlying JSON data (`stage2g_generation_eval.json`), and unit tests (`test_stage2k_enhancements.py`) all evaluate and display **90.4%**. The "100.0%" mention was an isolated clerical typo in the text of `STAGE2K_ACCEPTANCE_AUDIT.md` (which conflated the total 1.0 partition sum with usable structural validity).
2. **Ingredient Recall (89.21% vs 100.0%):** Ingredient recall is indeed hardcoded as `89.21%` (Stage 2F) and `4.11%` (Base) in `4_Model_Training.py`. Its authoritative source is Section 3 and Section 4 of `STAGE2G_TEST_EVALUATION_REPORT.md`. It cannot be read from `stage2g_generation_eval.json` because the evaluation script had a prompt-leakage flaw (searching prompt tokens), causing both models to record an artificial `1.0` (100%) in that JSON file.
3. **KerasHub & Streamlit Versions (0.33.0 / 1.67.0 vs. 0.20.0 / 1.43.2):** `venv_tf` has **`keras_hub==0.33.0`** and **`streamlit==1.67.0`**. The references to `0.20.0` and `1.43.2` in the metadata headers of `STAGE2K_UI_OPTIMIZATION_REPORT.md` and `STAGE2K_ACCEPTANCE_AUDIT.md` were header documentation errors, whereas `requirements.txt`, `app/main.py`, and `STAGE2B_GPT2_TEST_REPORT.md` correctly specify `0.33.0` and `1.67.0`.

---

## 2. Check 1 — Stage 2G Structural Validity & Ingredient Recall

### 2.1. Source Data Inspection

#### File: `data/outputs/stage2g_generation_eval.json`
```json
{
  "num_test_records": 250,
  "base_metrics": {
    "fully_closed_rate": 0.0,
    "partially_structured_rate": 0.0,
    "unstructured_rate": 1.0,
    "empty_rate": 0.952,
    "repetitive_rate": 0.608,
    "mean_ingredient_recall": 1.0,
    "full_ingredient_adherence_rate": 1.0
  },
  "stage2f_metrics": {
    "fully_closed_rate": 0.428,
    "partially_structured_rate": 0.476,
    "unstructured_rate": 0.096,
    "empty_rate": 0.004,
    "repetitive_rate": 0.04,
    "mean_ingredient_recall": 1.0,
    "full_ingredient_adherence_rate": 1.0
  }
}
```

#### File: `data/outputs/stage2g_teacher_forced.json`
```json
{
  "base": {
    "loss": 1.3617947101593018,
    "accuracy": 0.31330469250679016,
    "perplexity": 3.90319211804127
  },
  "stage2f": {
    "loss": 0.8854899406433105,
    "accuracy": 0.3907109498977661,
    "perplexity": 2.4241718008733533
  }
}
```

---

### 2.2. Metric Definitions & Recalculation

According to `STAGE2G_TEST_EVALUATION_REPORT.md` (Section 3 & 4):
1. **Fully Closed (`is_fully_closed`):**
   - Condition: Generated valid `<TITLE>`, `<QUANTITIES>`, and `<STEPS>` and cleanly closed with `</STEPS>` and `</RECIPE>` tags before token budget exhaustion.
   - Stage 2F Count: **107 / 250 records**
   - Stage 2F Rate: **42.8%** (`0.428`)
   - Base GPT-2: **0.0%** (0 / 250)
2. **Partially Structured (`is_partially_structured`):**
   - Condition: Generated valid `<TITLE>`, `<QUANTITIES>`, and `<STEPS>`, but was truncated mid-step by the 220-token test budget before emitting `</RECIPE>`.
   - Stage 2F Count: **119 / 250 records**
   - Stage 2F Rate: **47.6%** (`0.476`)
   - Base GPT-2: **0.0%** (0 / 250)
3. **Total Usable Structural Validity:**
   $$\text{Usable Structural Validity} = \text{Fully Closed Rate} + \text{Partially Structured Rate}$$
   $$\text{Stage 2F} = 42.8\% + 47.6\% = \mathbf{90.4\%} \quad (226 / 250 \text{ recipes})$$
   $$\text{Base GPT-2} = 0.0\% + 0.0\% = \mathbf{0.0\%} \quad (0 / 250 \text{ recipes})$$
4. **Unstructured / Tag Failure (`is_unstructured`):**
   - Stage 2F Count: **24 / 250 records**
   - Stage 2F Rate: **9.6%** (`0.096`)
   - Base GPT-2: **100.0%** (250 / 250)
5. **Partition Completeness:**
   $$42.8\% + 47.6\% + 9.6\% = 100.0\% \quad (250 / 250 \text{ records})$$

---

### 2.3. Root Cause: Why Did the Audit Report 100%?

- **In the Application Code (`app/pages/4_Model_Training.py`, line 50):**
  ```python
  s2f_struct = (s2f_gen.get("fully_closed_rate", 0.0) + s2f_gen.get("partially_structured_rate", 0.0)) * 100.0
  ```
  This evaluates to: `(0.428 + 0.476) * 100.0 = 90.4%`.
- **In the Unit Test (`tests/test_stage2k_enhancements.py`, line 100):**
  ```python
  self.assertAlmostEqual(s2f["structural_validity"], 90.4, places=1)
  ```
  This passes with `90.4%`.
- **In Streamlit UI:**
  Line 224 renders `tc3.metric("Structural Validity", f"{s2f_res['structural_validity']:.1f}%")`, which displays **`90.4%`**.
- **Audit Text Typo:**
  In `STAGE2K_ACCEPTANCE_AUDIT.md`, Section 2, Item 6, the summary text mistakenly stated:
  `Structural delimiter validity rate (fully_closed_rate + partially_structured_rate: 100.0%)`
  This was an audit report clerical error. The auditor accidentally wrote `100.0%` by conflating the total distribution sum ($0.428 + 0.476 + 0.096 = 1.0$) with the usable structural rate ($0.428 + 0.476 = 0.904$). **The code and application UI are correct and display 90.4%.**

---

### 2.4. Ingredient Recall Investigation

- **Hardcoded Status:** In `app/pages/4_Model_Training.py` (lines 60 and 69), `ingredient_recall` is hardcoded:
  ```python
  "ingredient_recall": 89.21,  # Recipe body adherence verified in Stage 2G report
  ...
  "ingredient_recall": 4.11,
  ```
- **Authoritative Source:** `STAGE2G_TEST_EVALUATION_REPORT.md` (Sections 3 and 4).
- **Why It Is Not Read from `stage2g_generation_eval.json`:**
  In `scripts/evaluate_test_set.py` (lines 257–264), `evaluate_ingredients(r["ingredients"], b_raw, b_struct)` performed regex search against `b_raw`. In KerasHub, `model.generate(batch)` returns the full prompt prefix prepended to the generated tokens. Because every prompt prefix explicitly contains all supplied ingredients inside `<INGREDIENTS>\n...`, a string search over `b_raw` matched the prompt itself, yielding an uninformative **1.0 (100.0%)** for both Base GPT-2 and Stage 2F in the raw JSON.
  In `STAGE2G_TEST_EVALUATION_REPORT.md`, the evaluator corrected this by stripping prompt prefix tokens and re-evaluating recall strictly inside the **generated recipe body**, establishing:
  - **Base GPT-2 Mean Ingredient Recall:** **`4.11%`**
  - **Stage 2F Full Fine-Tuned Mean Ingredient Recall:** **`89.21%`**
  Hardcoding these verified values in `4_Model_Training.py` was therefore intentional and scientifically accurate, as using the raw JSON value (`1.0`) would have falsely reported 100% recall for the untrained base model.

---

## 3. Check 2 — KerasHub & Package Versions Verification

### 3.1. Direct Runtime Inspection

Command executed:
```bash
HOME=/tmp ./venv_tf/bin/python -c "
import tensorflow as tf
import keras
import keras_hub
import streamlit as st

print('TensorFlow:', tf.__version__)
print('Keras:', keras.__version__)
print('KerasHub:', keras_hub.__version__)
print('Streamlit:', st.__version__)
"
```

**Actual Installed Versions in `venv_tf`:**
- **TensorFlow:** `2.21.0`
- **Keras:** `3.15.1`
- **KerasHub:** `0.33.0`
- **Streamlit:** `1.67.0`

---

### 3.2. Comparison Against Project Records

| Artifact / Record | TensorFlow | Keras | KerasHub | Streamlit |
|---|:---:|:---:|:---:|:---:|
| **`venv_tf` (Actual Installed)** | **2.21.0** | **3.15.1** | **0.33.0** | **1.67.0** |
| `requirements.txt` | 2.21.0 | 3.15.1 | 0.33.0 | 1.67.0 |
| `app/main.py` (Lines 218–220) | 2.21 | 3.15 | 0.33 | 1.67 |
| `STAGE2B_GPT2_TEST_REPORT.md` (Line 82) | 2.21.0 | 3.15.1 | 0.33.0 | — |
| `venv` (Legacy Python 3.14 env) | — | — | 0.19.0 | 1.67.0 |
| `STAGE2K_UI_OPTIMIZATION_REPORT.md` (Header) | 2.21.0 | — | *0.20.0 (error)* | *1.43.2 (error)* |
| `STAGE2K_ACCEPTANCE_AUDIT.md` (Header) | 2.21.0 | — | *0.20.0 (error)* | *1.67.0* |

---

### 3.3. Root Cause: KerasHub 0.20 vs 0.33.0 Discrepancy

- `venv_tf` has **never** had `keras-hub 0.20.0`. When `venv_tf` was created and verified in Stage 2B (`STAGE2B_GPT2_TEST_REPORT.md`), `keras_hub` was explicitly tested and recorded as `0.33.0`.
- The references to `0.20.0` and `1.43.2` originated solely in the metadata header of `STAGE2K_UI_OPTIMIZATION_REPORT.md` and were copied into `STAGE2K_ACCEPTANCE_AUDIT.md`.
- `requirements.txt` correctly pins `keras-hub==0.33.0` and `streamlit==1.67.0`.
- `app/main.py` correctly displays `KerasHub 0.33` and `Streamlit 1.67`.

---

## 4. Master Metric Reference Table (Authoritative)

For all future project documentation and Phase 3 packaging, the verified authoritative metrics and environment specifications are:

### Environment Specifications
- **Python:** `3.12.8`
- **TensorFlow:** `2.21.0`
- **Keras:** `3.15.1`
- **KerasHub:** `0.33.0`
- **Streamlit:** `1.67.0`
- **Platform:** macOS arm64 (Apple Silicon CPU execution)

### Stage 2G Held-Out Test Evaluation (250 Records)
| Metric | Base GPT-2 | Stage 2F Full Fine-Tuned | Authoritative Source |
|---|:---:|:---:|---|
| **Test Loss (Teacher-Forced)** | 1.3618 | **0.8855** | `stage2g_teacher_forced.json` |
| **Test Perplexity** | 3.90 | **2.42** | `stage2g_teacher_forced.json` |
| **Next-Token Accuracy** | 31.33% | **39.07%** | `stage2g_teacher_forced.json` |
| **Fully Closed Recipes (`</RECIPE>`)** | 0.0% (0/250) | **42.8% (107/250)** | `stage2g_generation_eval.json` |
| **Partially Structured Recipes** | 0.0% (0/250) | **47.6% (119/250)** | `stage2g_generation_eval.json` |
| **Total Usable Structural Validity** | **0.0% (0/250)** | **90.4% (226/250)** | Sum: $42.8\% + 47.6\%$ |
| **Unstructured / Delimiter Failure** | 100.0% (250/250) | **9.6% (24/250)** | `stage2g_generation_eval.json` |
| **Degenerate Repetition Rate** | 60.8% (152/250) | **4.0% (10/250)** | `stage2g_generation_eval.json` |
| **Mean Ingredient Recall (Recipe Body)** | 4.11% | **89.21%** | `STAGE2G_TEST_EVALUATION_REPORT.md` (Sec 3/4) |
| **Adherence Gain over Base** | — | **+85.10%** | `STAGE2G_TEST_EVALUATION_REPORT.md` |

---

## 5. Conclusion

1. The application code (`app/pages/4_Model_Training.py`) and UI are fully consistent with the original evaluation definitions and correctly display **90.4%** structural validity and **89.21%** ingredient recall.
2. The installed KerasHub version is confirmed to be **`0.33.0`** and Streamlit is **`1.67.0`**, matching `requirements.txt`.
3. No application code, datasets, model checkpoints, or environment packages were modified.
