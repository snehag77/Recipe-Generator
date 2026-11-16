# RecipeGen AI — Stage 3C.1 Correction Report: Preset Validator Hardening

**Date:** 2026-11-16  
**Project:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Execution Environment:** `venv_tf` (Python 3.12.8, TensorFlow 2.21.0, Keras 3.15.1, KerasHub 0.33.0, Streamlit 1.67.0)  
**Status:** COMPLETE & VERIFIED

---

## 1. Executive Summary

During the Phase 3C independent audit, runtime tracing revealed that KerasHub's `keras_hub.models.GPT2Tokenizer.from_preset()` requires `assets/tokenizer/vocabulary.json` in addition to the root configuration and weight files. In Phase 3C.1, `is_valid_preset_dir()` in `utils/gpt2.py` was updated to enforce validation of all 4 mandatory files. Unit test coverage was added to cover complete presets, partial presets, missing sub-assets, and nonexistent directories without triggering GPT-2 downloads or model loading.

All 40 unit and integration tests passed cleanly, and bit-for-bit checksums for all protected checkpoints and datasets were verified unchanged.

---

## 2. Modified Files

| File | Purpose of Edit | Backup File |
| :--- | :--- | :--- |
| `utils/gpt2.py` | Add `assets/tokenizer/vocabulary.json` check to `is_valid_preset_dir()` | `utils/gpt2.py.phase3c1.bak` |
| `tests/test_phase3b_portability.py` | Add 5 validator test cases; add mock vocab file to `test_preset_lookup_priority` | `tests/test_phase3b_portability.py.phase3c1.bak` |

---

## 3. Exact Code Changes

### 3.1 `utils/gpt2.py`

```diff
--- utils/gpt2.py.phase3c1.bak
+++ utils/gpt2.py
@@ -37,7 +37,12 @@
     """Validate that a candidate preset directory contains all required KerasHub GPT-2 files."""
     if not preset_path.is_dir():
         return False
-    required = ["config.json", "model.weights.h5", "tokenizer.json"]
+    required = [
+        "config.json",
+        "model.weights.h5",
+        "tokenizer.json",
+        "assets/tokenizer/vocabulary.json",
+    ]
     return all((preset_path / f).is_file() for f in required)
```

### 3.2 `tests/test_phase3b_portability.py`

Expanded `test_preset_dir_validator` into 5 test cases and updated `test_preset_lookup_priority`:

```diff
--- tests/test_phase3b_portability.py.phase3c1.bak
+++ tests/test_phase3b_portability.py
@@ -21,25 +21,48 @@
         self.assertTrue(Path(keras_home).name == ".keras")
 
     def test_preset_dir_validator(self):
-        """Test 2: is_valid_preset_dir correctly checks for config, weights, and tokenizer."""
+        """Test 2: is_valid_preset_dir correctly checks for config, weights, tokenizer, and vocabulary."""
         from utils.gpt2 import is_valid_preset_dir
 
         with tempfile.TemporaryDirectory() as tmpdir:
-            p = Path(tmpdir)
-            self.assertFalse(is_valid_preset_dir(p))
+            base_p = Path(tmpdir)
 
-            # Add only config
-            (p / "config.json").write_text("{}", encoding="utf-8")
-            self.assertFalse(is_valid_preset_dir(p))
+            # Case 1: Nonexistent directory
+            nonexistent = base_p / "nonexistent_dir"
+            self.assertFalse(is_valid_preset_dir(nonexistent))
 
-            # Add weights
-            (p / "model.weights.h5").write_bytes(b"dummy_weights")
-            self.assertFalse(is_valid_preset_dir(p))
+            def setup_preset(dir_name, missing_component=None):
+                target = base_p / dir_name
+                target.mkdir(parents=True, exist_ok=True)
+                (target / "assets" / "tokenizer").mkdir(parents=True, exist_ok=True)
+
+                if missing_component != "config":
+                    (target / "config.json").write_text("{}", encoding="utf-8")
+                if missing_component != "weights":
+                    (target / "model.weights.h5").write_bytes(b"dummy_weights")
+                if missing_component != "tokenizer":
+                    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
+                if missing_component != "vocab":
+                    (target / "assets" / "tokenizer" / "vocabulary.json").write_text("{}", encoding="utf-8")
+
+                return target
+
+            # Case 2: Missing configuration
+            missing_cfg = setup_preset("missing_config", missing_component="config")
+            self.assertFalse(is_valid_preset_dir(missing_cfg))
+
+            # Case 3: Missing model weights
+            missing_weights = setup_preset("missing_weights", missing_component="weights")
+            self.assertFalse(is_valid_preset_dir(missing_weights))
+
+            # Case 4: Missing tokenizer vocabulary
+            missing_vocab = setup_preset("missing_vocab", missing_component="vocab")
+            self.assertFalse(is_valid_preset_dir(missing_vocab))
+
+            # Case 5: Complete preset directory
+            complete_preset = setup_preset("complete_preset")
+            self.assertTrue(is_valid_preset_dir(complete_preset))
 
     def test_preset_lookup_priority(self):
         """Test 3: get_cached_preset_path respects KERAS_PRESET_DIR override."""
         from utils.gpt2 import get_cached_preset_path
@@ -49,6 +72,8 @@
             (p / "config.json").write_text("{}", encoding="utf-8")
             (p / "model.weights.h5").write_bytes(b"dummy")
             (p / "tokenizer.json").write_text("{}", encoding="utf-8")
+            (p / "assets" / "tokenizer").mkdir(parents=True, exist_ok=True)
+            (p / "assets" / "tokenizer" / "vocabulary.json").write_text("{}", encoding="utf-8")
```

---

## 4. Test Verification Results

The complete test suite was executed in `venv_tf`:

```bash
HOME=/tmp ./venv_tf/bin/python -m unittest -v \
  tests/test_dish_validation.py \
  tests/test_ingredient_parser.py \
  tests/test_phase1_fixes.py \
  tests/test_stage2k_enhancements.py \
  tests/test_phase3b_portability.py
```

### Test Breakdown

| Test Module | Tests | Result | Notes |
| :--- | :---: | :---: | :--- |
| `tests/test_dish_validation.py` | 7 | PASS | Biryani, Dal, Omelette title mismatch validation |
| `tests/test_ingredient_parser.py` | 9 | PASS | Delimiter handling, bullet points, numbered lists, deduplication |
| `tests/test_phase1_fixes.py` | 10 | PASS | Allergen screening, quantity bounds, metric loading |
| `tests/test_stage2k_enhancements.py` | 7 | PASS | Page imports, held-out metrics, safety integration, checkpoint identity |
| `tests/test_phase3b_portability.py` | 7 | PASS | Includes updated 4-file preset validator & preflight hash verification |
| **Total** | **40** | **ALL PASS** | **Execution time: 6.91s** |

All preset validator tests ran against temporary mock files (`tempfile.TemporaryDirectory`). At no point during testing was GPT-2 loaded, downloaded, or exported.

---

## 5. Protected Assets Integrity Verification

Bit-for-bit MD5 checksums were verified across all 6 protected assets:

| Asset Path | Expected MD5 | Actual MD5 | Status |
| :--- | :--- | :--- | :---: |
| `data/processed/recipes_train.jsonl` | `6d2487aa1719146437892318f5820dc9` | `6d2487aa1719146437892318f5820dc9` | MATCH (100% UNCHANGED) |
| `data/processed/recipes_validation.jsonl` | `01b7bf6c2d0ee71093f38e01c000a045` | `01b7bf6c2d0ee71093f38e01c000a045` | MATCH (100% UNCHANGED) |
| `data/processed/recipes_test.jsonl` | `11b581b58e352672e9ef2f7130cd10bb` | `11b581b58e352672e9ef2f7130cd10bb` | MATCH (100% UNCHANGED) |
| `models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5` | `d95eb61d471f183a67a0e7fb920369d0` | `d95eb61d471f183a67a0e7fb920369d0` | MATCH (100% UNCHANGED) |
| `models/gpt2_finetuned.weights.h5` | `a0ddb523b21e820c2fdcc871f2fae2a1` | `a0ddb523b21e820c2fdcc871f2fae2a1` | MATCH (100% UNCHANGED) |
| `models/recipe_gpt2/recipe_gpt2_best.weights.h5` | `0f733e894f4a83087164dfc1b7fe9a9a` | `0f733e894f4a83087164dfc1b7fe9a9a` | MATCH (100% UNCHANGED) |

---

## 6. Safety Compliance Checklist

- [x] Zero model training or fine-tuning performed.
- [x] Zero modifications to checkpoints or datasets.
- [x] Zero packages installed, upgraded, or uninstalled.
- [x] Zero model downloads triggered.
- [x] Base preset export not executed.
- [x] Backups created prior to modifying any file.
- [x] Streamlit app on port 8503 running uninterrupted.
- [x] All 40 unit and integration tests passing.
