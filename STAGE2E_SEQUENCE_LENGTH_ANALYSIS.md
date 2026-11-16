# RecipeGen AI – Stage 2E Pre-Training Configuration Verification

**Date:** 2026-11-16  
**Auditor:** Antigravity AI  
**Project Path:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Dataset Analyzed:** `data/processed/recipes_train.jsonl` (4,500 records) & `data/processed/recipes_validation.jsonl` (250 records)  
**Total Recipes Evaluated:** 4,750 records  
**Tokenizer:** KerasHub GPT-2 Base (`gpt2_base_en`, 50,257 vocab)

---

## 1. Executive Summary

Prior to initiating full-dataset fine-tuning (4,500 recipes) of GPT-2, a rigorous, read-only analysis was executed to evaluate token lengths, truncation dynamics, tag preservation, and CPU compute constraints on Apple Silicon.

Key conclusions:
1. **Sequence length 256 is inadequate:** 53.35% (2,534 / 4,750) of recipes are truncated, leaving more than half the training samples without closing tags (`</STEPS>`, `</RECIPE>`) or end cooking instructions. This directly caused the missing closing tags observed during the Stage 2D test generation.
2. **Sequence length 384 leaves 22.53% truncated:** While better than 256, nearly 1 in 4 recipes remains incomplete.
3. **Sequence length 512 is the optimal sweet spot:** 90.93% (4,319 / 4,750) of recipes retain their complete body and closing structural tags, capturing through the 90th percentile of recipe lengths (499 tokens). CPU training runtime is estimated at ~2.3 hours on Apple Silicon M4 with peak RAM under 4.2 GB.
4. **Truncation direction is strictly from the right:** Because the prompt layout starts with `<INGREDIENTS>`, `<TITLE>`, and `<QUANTITIES>`, truncation *never* harms ingredient conditioning or titles; it only cuts later cooking steps and closing tags.

---

## 2. Token Length Distribution Analysis

Every recipe in `recipes_train.jsonl` and `recipes_validation.jsonl` was tokenized using the exact KerasHub `GPT2Tokenizer.from_preset(".../gpt2_base_en/3")`.

### Overall Combined Distribution (4,750 Recipes)

| Metric | Token Count | Description |
| :--- | :--- | :--- |
| **Minimum** | **79** | Shortest recipe in dataset |
| **Median (50th Percentile)** | **266** | 50% of recipes are $\le 266$ tokens |
| **Mean** | **307.99** | Average recipe token length |
| **75th Percentile** | **367** | 75% of recipes are $\le 367$ tokens |
| **85th Percentile** | **443** | 85% of recipes are $\le 443$ tokens |
| **90th Percentile** | **499** | 90% of recipes are $\le 499$ tokens |
| **95th Percentile** | **603** | 95% of recipes are $\le 603$ tokens |
| **99th Percentile** | **843** | 99% of recipes are $\le 843$ tokens |
| **Maximum** | **1,383** | Longest recipe in dataset |

### Distribution by Split

| Split | Records | Min | Median | Mean | 75th | 90th | 95th | 99th | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 4,500 | 79 | 267.0 | 307.80 | 366.0 | 499.0 | 601.1 | 842.0 | 1,383 |
| **Validation** | 250 | 101 | 263.0 | 311.33 | 383.0 | 534.6 | 619.7 | 820.1 | 901 |

The train and validation splits exhibit virtually identical distributions (median 267 vs 263, mean 308 vs 311), confirming that reservoir sampling with seed 42 produced a statistically balanced split.

---

## 3. Sequence Length Comparison (256, 384, 512, 640)

For each sequence length threshold, we calculated the exact count and percentage of truncated recipes, as well as the retention of the critical structural tags `</STEPS>` and `</RECIPE>`.

| Candidate Length | Truncated Recipes | Truncation % | Retaining `</STEPS>` | Retaining `</RECIPE>` | Complete Recipe Body % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | 2,534 / 4,750 | **53.35%** | 2,216 / 4,750 (46.65%) | 2,216 / 4,750 (46.65%) | **46.65%** |
| **384** | 1,070 / 4,750 | **22.53%** | 3,680 / 4,750 (77.47%) | 3,680 / 4,750 (77.47%) | **77.47%** |
| **512** | **431 / 4,750** | **9.07%** | **4,319 / 4,750 (90.93%)** | **4,319 / 4,750 (90.93%)** | **90.93%** |
| **640** | 182 / 4,750 | **3.83%** | 4,568 / 4,750 (96.17%) | 4,568 / 4,750 (96.17%) | **96.17%** |

### Key Observations:
- **At 256 tokens**, over half the training dataset (53.35%) is truncated. Teaching a model to generate complete recipes when half its training examples abruptly stop mid-step without `</STEPS>` or `</RECIPE>` inherently degrades generation quality.
- **At 384 tokens**, 22.53% are truncated. While an improvement, more than 1 in 5 recipes still loses closing tags.
- **At 512 tokens**, completeness jumps to **90.93%**, covering 90% of all recipes without any clipping.
- **At 640 tokens**, completeness reaches 96.17%, but the additional 5.24% completeness requires a ~35% increase in compute time.

---

## 4. Truncation Behavior & Structural Integrity

### Truncation Direction
KerasHub `GPT2CausalLMPreprocessor` truncates sequences strictly from the **right** (end of text). 

### Impact by Section
Because our standardized prompt format is structured sequentially:
```
<INGREDIENTS>
[ingredient list]
<END_ING>

<RECIPE>
<TITLE>
[recipe title]
</TITLE>
<QUANTITIES>
[quantities list]
</QUANTITIES>
<STEPS>
[cooking instructions]
</STEPS>
</RECIPE>
```

The right-truncation behavior has the following effects:
1. **`<INGREDIENTS>` & `<END_ING>`:** **100% Preserved.** Truncation *never* affects input ingredients.
2. **`<TITLE>`:** **100% Preserved.** Truncation *never* affects titles.
3. **`<QUANTITIES>`:** **100% Preserved.** Truncation *never* cuts ingredient quantities in any recipe.
4. **Early Cooking Steps (Steps 1–3):** **100% Preserved.**
5. **Later Cooking Steps & Closing Tags (`</STEPS>`, `</RECIPE>`): **Subject to truncation** when token length exceeds the threshold.

### Concrete Example: *"Peach Pie With Cut-Out Pastry"* (537 Tokens)

- **Original Recipe Ending:**
  ```text
  21. Bake 35 minutes.
  22. Remove aluminum foil.
  23. Continue baking 20-25 minutes or until crust is lightly browned and juice begins to bubble.
  24. Serve warm or cool.
  </STEPS>
  </RECIPE>
  ```
- **At Sequence Length 256:**
  ```text
  ... fork until mixture resembles coarse crumbs.
  3. Stir in enough water with fork just until flour is moistened.
  4. Divide dough in half; shape each half [CUTS OFF HERE - 20 STEPS REMOVED]
  ```
- **At Sequence Length 384:**
  ```text
  ...
  11. Crimp or flute edge.
  12. Set aside.
  13. Combine sugar and cornstarch in bowl.
  14. Add pe [CUTS OFF HERE - 10 STEPS REMOVED]
  ```
- **At Sequence Length 512:**
  ```text
  ...
  21. Bake 35 minutes.
  22. Remove aluminum foil.
  23. Continue baking 20-25 minutes or until crust is lightly browned [CUTS OFF AT VERY END - 23 OF 24 STEPS RETAINED]
  ```

---

## 5. Resource Requirements & CPU Runtime Estimates

The RecipeGen AI environment runs on an Apple Silicon M4 with 16 GB of unified memory. TensorFlow 2.21 executes on **CPU** (`/device:CPU:0`).

### Training Workload Parameters
- **Dataset Size:** 4,500 training recipes
- **Batch Size:** 4
- **Total Steps per Epoch:** $4,500 / 4 = 1,125$ optimization steps
- **Validation Dataset:** 250 recipes (63 batches)
- **Model:** GPT-2 Base (124M parameters, float32)
- **Optimizer:** AdamW with linear warmup and cosine decay

### Configuration Comparison Table

| Metric | Seq Len 256 | Seq Len 384 | Seq Len 512 (Recommended) | Seq Len 640 |
| :--- | :---: | :---: | :---: | :---: |
| **Recipes Truncated** | 2,534 (53.35%) | 1,070 (22.53%) | **431 (9.07%)** | 182 (3.83%) |
| **Recipes Complete** | 2,216 (46.65%) | 3,680 (77.47%) | **4,319 (90.93%)** | 4,568 (96.17%) |
| **Est. Step Time (CPU)** | ~3.65s / step | ~5.50s / step | **~7.50s / step** | ~10.20s / step |
| **1 Epoch Train Time** | ~68 min (1.14 h) | ~103 min (1.72 h) | **~140 min (2.34 h)** | ~191 min (3.19 h) |
| **Validation Eval Time** | ~3 min | ~4.5 min | **~6.5 min** | ~8.5 min |
| **Total Runtime (1 Epoch)** | ~71 min (1.18 h) | ~108 min (1.80 h) | **~147 min (2.45 h)** | ~200 min (3.33 h) |
| **Model Weights (FP32)** | 496 MB | 496 MB | 496 MB | 496 MB |
| **AdamW Optimizer States** | 992 MB | 992 MB | 992 MB | 992 MB |
| **Activations & Buffers** | ~250 MB | ~380 MB | ~520 MB | ~680 MB |
| **Peak Memory (RAM)** | ~3.2 GB | ~3.6 GB | **~4.1 GB** | ~4.7 GB |
| **Memory Headroom (16GB)** | ~12.8 GB (80%) | ~12.4 GB (78%) | **~11.9 GB (74%)** | ~11.3 GB (71%) |

*Note: Peak RAM is well within the 16 GB unified memory limit across all options.*

---

## 6. Dataset Ordering & Shuffling Assessment

1. **Storage Order:** `recipes_train.jsonl` was generated via reservoir sampling with random seed 42. While the sample pool is representative, sequential reading without runtime shuffling creates batches with local clustering.
2. **Current Implementation in `train_gpt2.py`:**
   ```python
   def build_tf_dataset(texts: list, batch_size: int, shuffle: bool = True):
       ds = tf.data.Dataset.from_tensor_slices(texts)
       if shuffle:
           ds = ds.shuffle(buffer_size=min(len(texts), 2000), seed=42)
       ds = ds.batch(batch_size)
       ds = ds.prefetch(tf.data.AUTOTUNE)
       return ds
   ```
3. **Recommendation:**
   - Increase `buffer_size` from `min(len(texts), 2000)` to `len(texts)` (4,500) so the entire dataset is held in the shuffle buffer for a true, uniform pseudo-random permutation.
   - Maintain `seed=42` for exact reproducibility across runs.

---

## 7. Strategic Recommendations

### Recommendation: Sequence Length 512
- **Why 512 tokens:**
  - Achieves **90.93% complete recipes**, capturing through the 90th percentile (499 tokens).
  - Vastly superior to 256 tokens (which discards over half the recipe completions).
  - Highly manageable training duration on CPU: **~2 hours 25 minutes** for 1 full epoch over all 4,500 recipes.
  - Safe memory profile: ~4.1 GB peak RSS on a 16 GB machine.

### Alternative Option: Sequence Length 384
- If training time must strictly stay under 2 hours, 384 tokens achieves **77.47% completeness** in **~1 hour 48 minutes**.

### Rejected Configurations:
- **256 tokens:** Rejected due to excessive truncation (53.35%).
- **640 tokens:** Rejected for full initial run due to diminishing returns (only +5.2% completeness over 512 tokens at the cost of +53 minutes extra CPU compute).

---

## 8. Pre-Training Configuration Checklist

| Item | Specification | Status |
| :--- | :--- | :---: |
| **Base Model** | KerasHub `gpt2_base_en` (124M params) | Verified |
| **Sequence Length** | **512 tokens** | Recommended |
| **Batch Size** | 4 | Verified safe |
| **Optimizer** | AdamW (`weight_decay=0.01`) | Verified |
| **Learning Rate** | `5e-5` with Cosine Decay and Warmup | Verified |
| **Epochs** | 1 (1,125 optimization steps) | Recommended |
| **Shuffle Buffer** | Full dataset size (4,500), `seed=42` | Recommended |
| **Output Path** | `models/recipe_gpt2/recipe_gpt2_best.weights.h5` | Verified |
| **Pilot Preserved** | `models/gpt2_finetuned.weights.h5` intact | Verified |

---

*Report prepared and validated by Antigravity AI. Awaiting user review and confirmation before updating `scripts/train_gpt2.py` and starting training.*
