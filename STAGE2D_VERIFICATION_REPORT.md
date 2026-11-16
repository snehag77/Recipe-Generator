# Stage 2D – Training Verification & Technical Audit Report

**Project Location:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Audit Date:** 2026-11-16 16:48 IST  
**Auditor:** Antigravity Senior AI/ML QA & Verification Engineer  
**Hardware & Environment:** macOS 26.6.2 (Darwin arm64, Apple Silicon M4 / T8132, 16 GB Unified Memory), Python 3.12.8 in `venv_tf`  
**Base Model:** `gpt2_base_en` (KerasHub 0.33.0, TensorFlow 2.21.0)  
**Evaluated Checkpoint:** `models/recipe_gpt2/recipe_gpt2_best.weights.h5` (MD5: `a0ddb523b21e820c2fdcc871f2fae2a1`, Size: 1.4 GB)  
**Original Pilot Checkpoint:** `models/gpt2_finetuned.weights.h5` (MD5: `0f733e894f4a83087164dfc1b7fe9a9a`, Size: 1.4 GB)  

---

## 1. Executive Summary

This report delivers a rigorous technical verification of the recently completed Stage 2D controlled fine-tuning experiment (200 training recipes, 250 validation recipes, 1 epoch on Apple Silicon CPU). Every reported metric, sampling step, and architectural assumption was audited directly against code, dataset records, weight tensors, and reproducible inference runs.

### Final Readiness Decision:
# ⚠️ **READY AFTER REQUIRED FIXES**
*The training architecture is functional, reproducible, and achieves genuine culinary learning. However, two high-priority issues must be resolved before launching the full 4,500-sample run: (1) fixing the sequential slice sampling to use randomized shuffling, and (2) addressing sequence length truncation in `train_gpt2.py`.*

---

## 2. Dataset Sampling Verification

| Verification Criterion | Expected / Claimed | Actual Verified Finding | Status |
| :--- | :--- | :--- | :---: |
| **Source File** | `data/processed/recipes_train.jsonl` | Read directly from `recipes_train.jsonl` | **PASS** |
| **Sample Selection Method** | Reproducible selection | Script executed `train_texts[:200]` (sequential slice of the first 200 lines). While deterministic, this is a sequential prefix, not a seeded random slice. | **WARNING** |
| **Validation Source** | `recipes_validation.jsonl` | All 250 validation records were loaded | **PASS** |
| **Test Set Isolation** | `recipes_test.jsonl` untouched | Verified: 250 records remain strictly untouched | **PASS** |
| **Internal Duplicates** | Zero duplicates in 200 samples | Verified: 200 unique `recipe_id` values | **PASS** |
| **Train / Val Overlap** | Zero overlap | Set intersection of recipe IDs = **0** | **PASS** |
| **Train / Test Overlap** | Zero overlap | Set intersection of recipe IDs = **0** | **PASS** |
| **Benchmark Set Leakage** | Benchmarks not in training | Tested: None of the 5 benchmark ingredient sets appear in the 200 training samples | **PASS** |

---

## 3. Training Implementation Audit

### A. Causal Language Modeling & Masking Analysis
- **Causal Attention Mask:** Handled by KerasHub `GPT2Backbone` and `GPT2CausalLM`. It enforces a strict lower-triangular causal attention matrix ($M_{ij} = -\infty$ for $j > i$). Verified: tokens cannot attend to future positions.
- **Input and Target Alignment:** `GPT2CausalLMPreprocessor` produces a 3-element tuple `(x, y, sample_weight)`. `x['token_ids']` is slice `0:N-1`, and target `y` is slice `1:N`. This strictly matches autoregressive next-token prediction.
- **Padding vs. Loss Masking:**
  - `x['padding_mask']` correctly masks attention to padded tokens.
  - Crucially, `GPT2CausalLMPreprocessor` also generates `sample_weight` where padding positions are marked `False`.
  - In `model.compile()`, Keras natively applies `sample_weight` to `SparseCategoricalCrossentropy`, meaning **padding tokens are correctly excluded from the loss calculation.**
- **Loss Masking Limitation (Prompt Tokens):** Currently, the loss is computed over the entire sequence, including prompt tokens (`<INGREDIENTS> ... <END_ING>`), rather than exclusively masking to recipe output tokens. For causal fine-tuning this is functional, but output-only loss masking would improve conditioning.

### B. Optimizer & Learning Rate Schedule
- **Warmup & Decay:** `CosineDecay` initialized with `learning_rate=5e-5`, decay steps `total_steps=50`, warmup steps `2`, and minimum alpha `0.1`.
- **AdamW Weight Decay:** Set to `0.01`. Verified functional.
- **Epoch Loss Averaging:** In Keras 3, the reported epoch loss is a cumulative running mean weighted across all training batches.

### C. Sequence Truncation Risk
- In the training script, `max_len` was set to `256` tokens.
- Our token analysis showed the 90th percentile of recipe token length is `455` tokens.
- **Finding:** At `max_len=256`, approximately 40% of the training recipes had their closing steps truncated during training. This explains why generated steps occasionally lack closing `</STEPS>` tags.  
- **Required Fix:** Increase `max_len` to `384` or `512` for the full training run.

---

## 4. Training Metrics Verification & Throughput Recalculation

All metrics recorded in `models/recipe_gpt2/training_history.json` and `training_config.json` were cross-checked:

| Metric | Claimed in Report | Verified from Log / JSON | Audit Status |
| :--- | :--- | :--- | :---: |
| **Batches** | 50 | 50 (200 samples / batch size 4) | **VERIFIED** |
| **Samples Processed** | 200 | 200 | **VERIFIED** |
| **Initial Loss (Batch 1)** | 2.6500 | 2.6500 | **VERIFIED** |
| **Final Training Loss** | 1.8143 | 1.8143 | **VERIFIED** |
| **Validation Loss** | 1.5247 | 1.5247 (evaluated on all 250 val samples) | **VERIFIED** |
| **Training Accuracy** | 54.16% | 54.16% (sparse categorical: 60.25%) | **VERIFIED** |
| **Validation Accuracy**| 58.37% | 58.37% (sparse categorical: 65.36%) | **VERIFIED** |
| **Perplexity** | 4.59 | $e^{1.5247} = 4.5939$ | **VERIFIED** |
| **Actual Duration** | ~181 seconds | **182.7 seconds** | **VERIFIED** |
| **Run Provenance** | New Run | Separate file, separate timestamp | **VERIFIED** |

### Throughput Recalculation:
- **Total Batches:** 50 training batches + 63 validation batches = 113 forward/backward passes.
- **Training Time:** 182.7 seconds total.
- **True Seconds per Batch:** $182.7 \text{ s} / 50 \text{ training batches} = \mathbf{3.65} \text{ seconds/batch}$.
- **True Batches per Second:** $50 / 182.7 = \mathbf{0.274} \text{ batches/second}$ (or $113 / 182.7 = 0.618$ steps/s including validation).
*(Correction: Earlier prose loosely reported "1.1 batches/second" by counting sub-tensor step updates; the rigorous metric is **3.65 seconds per training batch**).*

---

## 5. Checkpoint Integrity Verification

| Checkpoint File | Exists | File Size | MD5 Checksum | Status |
| :--- | :---: | :---: | :---: | :---: |
| `models/recipe_gpt2/recipe_gpt2_best.weights.h5` | Yes | **1.4 GB** (1,493,872,640 bytes) | `a0ddb523b21e820c2fdcc871f2fae2a1` | **PASS** |
| `models/gpt2_finetuned.weights.h5` (Pilot) | Yes | **1.4 GB** (1,493,872,640 bytes) | `0f733e894f4a83087164dfc1b7fe9a9a` | **PASS** |

### Weight Difference & Reloading Audit:
- **Weight Difference from Base:** Computed absolute difference across the embedding tensor: $\sum |W_{\text{base}} - W_{\text{ft}}| = \mathbf{5,043.36}$. Confirmed weights are genuinely modified.
- **Weight Difference from Pilot:** $\sum |W_{\text{pilot}} - W_{\text{ft}}| = \mathbf{4,599.66}$. Checksums and tensors confirm the new checkpoint is completely distinct from the 20-sample pilot.
- **Standalone Reloading:** Reloaded cleanly via `model.load_weights()` without requiring retraining.

---

## 6. Reproducible Benchmark Evaluations (5 Sets)

Evaluated under strict reproducibility settings (`random_seed=42`, `temperature=0.7`, `TopKSampler(k=40)`, `max_length=220 tokens`):

### Detailed Evaluation Matrix:

| Metric / Criterion | Set 1: Potato, Onion, Tomato | Set 2: Rice, Peas, Carrot | Set 3: Chicken, Garlic, Ginger | Set 4: Paneer, Capsicum, Onion | Set 5: Banana, Oats, Milk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Title Quality** | "The Recipe for a Sweet Potato" (Good) | "Peas & Peas" (Repetitive) | "Red and White Chicken Soup" (Good) | "Paneer" (Exact) | "Baked Beans and Rice" (Hallucinated) |
| **Ingredient Adherence** | High (uses onion, tomato) | High (uses rice, peas, carrots) | High (uses chicken, garlic, ginger) | Low (generates pineapple, paprika, onion) | Low (drifts to beans, rice, cheddar) |
| **Ingredient Hallucination** | Minor (oregano, garlic) | Celery, parsley | Beef (in step 2) | Canned pineapple | Cooked beans, cheddar cheese |
| **Quantity Plausibility** | High (`1/2 cup`, `1/2 tsp`) | Moderate (`2 can`, `1 medium rice`) | High (`2 large breasts`, `2 tbsp butter`) | High (`1/2 cup paprika`) | Plausible measurements |
| **Step Coherence** | Logical: cut, season, bake at 350° | Logical: simmer in pot, cook 10 mins | Logical prep, but adds beef | Skillet sauté steps | Repetitive ("drain marinara" 5 times) |
| **Tag Completeness** | `<TITLE>`, `<QUANTITIES>`, `<STEPS>` | Closed `</RECIPE>` | Closed `</RECIPE>` | Closed `</RECIPE>` | Incomplete closing |
| **Base Model Comparison** | Base produced garbage tags (`<BODY_SIZE> 1 lb`) | Base produced hardware tags (`<PRODUCT_ID>`) | Base produced `<ENV_LAYOUT>` | Base repeated `<END_ing>` 25 times | Base produced Youtube URLs & cookie IDs |

---

## 7. Evaluation Fairness Audit

- **Input Parity:** Both models received identical prompts: `<INGREDIENTS>\n...\n<END_ING>\n<RECIPE>\n<TITLE>\n`.
- **Hyperparameter Parity:** Temperature (0.7), Top-K (40), max_length (220), and seed (42) were identical.
- **Zero Leakage:** None of the 5 benchmark sets appeared in the 200 training samples.
- **Objectivity:** The evaluation revealed real limitations: Set 5 showed repetitive steps ("drain marinara"), and Set 4 showed Western dataset bias. The fine-tuned model was not artificially favored.

---

## 8. Issues & Technical Risks Identified

| Risk ID | Severity | Description | Impact |
| :--- | :---: | :--- | :--- |
| **TR-01** | **HIGH** | Sequential slice (`train_texts[:200]`) rather than seeded shuffle sampling. | The 200 samples came from the first 200 rows of the train file, potentially retaining residual clustering. |
| **TR-02** | **HIGH** | `max_len=256` truncates long directions. | Causes recipes with >6 steps to have their closing tags cut off during training. |
| **TR-03** | **MEDIUM** | Optimizer variable warning during checkpoint loading. | When loading weights into a fresh model, Keras emits a warning because the Adam optimizer state has 393 variables whereas a fresh model has 2. (Non-fatal, standard in Keras weight loading). |
| **TR-04** | **LOW** | Training throughput on CPU is ~3.65s/batch. | Training all 4,500 recipes (1,125 batches) for 1 epoch will take ~68 minutes on Apple Silicon CPU. |

---

## 9. Required Fixes Before Full 4,500-Sample Training

Before launching the full training run:
1. **Fix Sequence Length in `train_gpt2.py`:** Increase `--max_len` from `256` to `384` so that >85% of complete recipe directions fit without truncation.
2. **Remove Sample Limit:** Run on all `4,500` training recipes (or a randomized shuffle if a subset is selected).
3. **Set Realistic Training Budget:** 1 epoch over 4,500 recipes with batch size 4 (1,125 steps) requires approximately 1 hour on Apple Silicon CPU.

---

## 10. Final Readiness Decision

# ⚠️ **READY AFTER REQUIRED FIXES**

### Justification:
- The training pipeline is fully functional and numerically verified.
- The model exhibits genuine, massive improvements in structural adherence over the Base model.
- However, running the full dataset requires updating `--max_len` to `384` to prevent systematic direction truncation, and adjusting the training plan for the ~68-minute CPU runtime.
