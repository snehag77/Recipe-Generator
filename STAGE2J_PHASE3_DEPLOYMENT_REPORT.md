# RecipeGen AI — Stage 2J: Phase 3 Deployment, Packaging & Portability Report

**Date:** 2026-11-16  
**Auditor & Implementation Engineer:** Antigravity AI  
**Project Path:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Target Environment:** `venv_tf` (Python 3.12.8, TensorFlow 2.21.0, Keras 3.15.1, KerasHub 0.33.0, Streamlit 1.67.0)  
**Status:** Completed. P0 deployment blockers resolved, requirements.txt aligned, machine-specific paths removed, `.gitignore` large-file guardrails established, model download helper created, and full test suite passing.

---

## 1. Executive Summary

Phase 3 has resolved the critical deployment and portability blockers identified in `STAGE2H_FULL_PROJECT_AUDIT_REPORT.md`:

1. **`requirements.txt` Aligned (P0):** The previous 2-line placeholder was replaced with an exact, pinned specification of all 10 direct runtime dependencies (`tensorflow==2.21.0`, `keras==3.15.1`, `keras-hub==0.33.0`, `h5py==3.14.0`, `streamlit==1.67.0`, `numpy==2.5.5`, `pandas==3.1.1`, `pillow==13.0.0`, `requests==2.34.2`, `tqdm==4.70.1`). Deprecated `keras-nlp` was excluded.
2. **Dynamic Project-Relative Paths (P0):** Hardcoded paths and brittle relative directory assumptions were eliminated across all scripts and application pages. `pathlib.Path` anchors all paths dynamically to `PROJECT_ROOT`, allowing execution from any working directory (`Recipe_genai/`, `scripts/`, or external shells).
3. **Large Model Checkpoint Guardrails (P0):** `.gitignore` was configured to prevent standard Git staging of 1.49 GB weight files (`*.weights.h5`, `*.h5`, `*.keras`), while explicitly preserving lightweight JSON configurations (`training_config.json`, `training_history.json`).
4. **Secure Model Downloader & Verifier Implemented:** Created [`scripts/download_model.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/download_model.py) supporting atomic downloading, user-supplied SHA-256 / MD5 checksum verification, default overwrite refusal, and error cleanup. The script was syntax-checked and tested in dry-run mode (zero external connections executed).
5. **Zero Model or Dataset Weights Altered:** All model checkpoints (`recipe_gpt2_full_best.weights.h5`, `gpt2_finetuned.weights.h5`, `recipe_gpt2_best.weights.h5`) and JSONL dataset splits remain 100% byte-for-byte identical to pre-audit states.
6. **All Regression & Integration Tests Passing:** 10/10 Phase 1 tests, 5/5 Phase 2 tests, and the mock dataset fixture test passed with zero errors.

---

## 2. Pre-Implementation Audit Findings

### 2.1 Verified Package Versions (`./venv_tf/bin/python`)

| Package | Previous Report | Actual Verified in `venv_tf` | Direct vs Transitive | Status in Old `requirements.txt` |
| :--- | :---: | :---: | :---: | :---: |
| **Python** | 3.12.8 | `3.12.8` | Runtime Engine | N/A |
| **TensorFlow** | 2.21.0 | `2.21.0` | Direct | **Missing** |
| **Keras** | 3.15.1 | `3.15.1` | Direct | **Missing** |
| **KerasHub** | 0.33.0 | `0.33.0` | Direct | **Missing** |
| **h5py** | 3.14.0 | `3.14.0` | Direct (HDF5 weights) | **Missing** |
| **NumPy** | 2.5.5 | `2.5.5` | Direct | **Missing** |
| **Pandas** | 3.1.1 | `3.1.1` | Direct | **Missing** |
| **Streamlit** | 1.67.0 | `1.67.0` | Direct | `streamlit>=1.30` |
| **Pillow** | 13.0.0 | `13.0.0` | Direct | `pillow>=10.2.0` |
| **requests** | 2.34.2 | `2.34.2` | Direct (Downloader) | **Missing** |
| **tqdm** | 4.70.1 | `4.70.1` | Direct (Downloader) | **Missing** |

### 2.2 Checkpoint Inventory & File Sizes

| Checkpoint Path | Byte Size | Size (GB) | MD5 Checksum | Git Push Rejection Risk |
| :--- | :---: | :---: | :---: | :---: |
| [`models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5`](file:///Users/snehagupta/Downloads/Recipe_genai/models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5) | 1,493,862,440 | ~1.49 GB | `d95eb61d471f183a67a0e7fb920369d0` | **Hard Reject (>100MB)** |
| [`models/gpt2_finetuned.weights.h5`](file:///Users/snehagupta/Downloads/Recipe_genai/models/gpt2_finetuned.weights.h5) | 1,493,862,440 | ~1.49 GB | `a0ddb523b21e820c2fdcc871f2fae2a1` | **Hard Reject (>100MB)** |
| [`models/recipe_gpt2/recipe_gpt2_best.weights.h5`](file:///Users/snehagupta/Downloads/Recipe_genai/models/recipe_gpt2/recipe_gpt2_best.weights.h5) | 1,493,862,440 | ~1.49 GB | `0f733e894f4a83087164dfc1b7fe9a9a` | **Hard Reject (>100MB)** |
| **Total Weights Storage** | **4,481,587,320** | **~4.48 GB** | — | **Immediate Blocker for git push** |

### 2.3 Git Tracking Status
- **Repository Initialized:** No `.git` directory exists in `/Users/snehagupta/Downloads/Recipe_genai`.
- **Current Tracked Files:** 0 files (untracked project).
- **Risk Identified:** Running `git init && git add .` without updated `.gitignore` would immediately attempt to stage 4.48 GB of binary model weights and 2.29 GB of raw CSV data, which GitHub blocks upon push (100 MB hard file limit).

### 2.4 Hardcoded & Brittle Paths Identified
1. **Google Colab Hardcoded Roots:** `scripts/package_models.py` and `scripts/find_needed_files.py` hardcoded `/content`, `/content/drive/MyDrive`, and `/content/p45_models.zip`.
2. **Current Working Directory Brittleness:** `scripts/train_gpt2.py`, `scripts/evaluate_test_set.py`, `scripts/validate_recipe_dataset.py`, `scripts/analyze_recipe_dataset.py`, and `scripts/prepare_recipe_dataset.py` used relative string defaults like `data/processed/recipes_train.jsonl`. When executed from any directory other than root (e.g., `cd scripts && python validate_recipe_dataset.py`), execution crashed with `FileNotFoundError`.
3. **Empty Directory Cache Traps:** `utils/gpt2.py` and `scripts/evaluate_test_set.py` used `is_dir()` on `~/.cache/kagglehub/.../3` without verifying that `config.json` actually existed inside, causing crashes if an empty directory was created.

---

## 3. Files Inspected and Modified

### 3.1 Files Modified & Backup Locations

| File Path | Backup Copy Path | Nature of Change |
| :--- | :--- | :--- |
| [`requirements.txt`](file:///Users/snehagupta/Downloads/Recipe_genai/requirements.txt) | [`requirements.txt.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/requirements.txt.bak) | Pinned exact direct dependencies (10 packages), removed ambiguity, documented Python 3.12 target. |
| [`.gitignore`](file:///Users/snehagupta/Downloads/Recipe_genai/.gitignore) | [`.gitignore.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/.gitignore.bak) | Added rules for `*.weights.h5`, `*.h5`, `*.keras`, `.cache/`, `.keras/`, `*.bak`, `*.tmp`. Explicitly whitelisted `!models/**/*.json`. |
| [`utils/gpt2.py`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/gpt2.py) | [`utils/gpt2.py.phase2.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/utils/gpt2.py.phase2.bak) | Converted paths to `pathlib.Path`. Made preset detection multi-candidate and require `config.json` file existence. |
| [`app/pages/4_Model_Training.py`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/4_Model_Training.py) | [`app/pages/4_Model_Training.py.phase1.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/app/pages/4_Model_Training.py.phase1.bak) | Converted metrics directories to dynamic `PROJECT_ROOT` via `pathlib.Path`. |
| [`scripts/train_gpt2.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/train_gpt2.py) | [`scripts/train_gpt2.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/train_gpt2.py.bak) | Anchored all CLI defaults to `PROJECT_ROOT`. Made checkpoint isolation checks robust to dynamic paths. |
| [`scripts/evaluate_test_set.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/evaluate_test_set.py) | [`scripts/evaluate_test_set.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/evaluate_test_set.py.bak) | Anchored defaults to `PROJECT_ROOT`. Added fallback logic for local cached preset directory. |
| [`scripts/validate_recipe_dataset.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/validate_recipe_dataset.py) | [`scripts/validate_recipe_dataset.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/validate_recipe_dataset.py.bak) | Anchored default `--dataset_dir` to `PROJECT_ROOT / "data" / "processed"`. |
| [`scripts/analyze_recipe_dataset.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/analyze_recipe_dataset.py) | [`scripts/analyze_recipe_dataset.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/analyze_recipe_dataset.py.bak) | Anchored default `--dataset_dir` to `PROJECT_ROOT / "data" / "processed"`. |
| [`scripts/prepare_recipe_dataset.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/prepare_recipe_dataset.py) | [`scripts/prepare_recipe_dataset.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/prepare_recipe_dataset.py.bak) | Anchored output dir to `PROJECT_ROOT`. Added `process_dataset` alias and `valid_kept` statistic key. |
| [`scripts/check_gpt2.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/check_gpt2.py) | [`scripts/check_gpt2.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/check_gpt2.py.bak) | Converted preset path resolution to `pathlib.Path`. |
| [`scripts/find_needed_files.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/find_needed_files.py) | [`scripts/find_needed_files.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/find_needed_files.py.bak) | Added local `PROJECT_ROOT` search before Colab and Drive checks. |
| [`scripts/package_models.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/package_models.py) | [`scripts/package_models.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/package_models.py.bak) | Added local `PROJECT_ROOT` search and output zip path fallback. |
| [`tests/test_dataset_pipeline.py`](file:///Users/snehagupta/Downloads/Recipe_genai/tests/test_dataset_pipeline.py) | [`tests/test_dataset_pipeline.py.bak`](file:///Users/snehagupta/Downloads/Recipe_genai/tests/test_dataset_pipeline.py.bak) | Added `PROJECT_ROOT` to `sys.path` to enable invocation from any directory. |

### 3.2 New Files Created

| File Path | Purpose |
| :--- | :--- |
| [`scripts/download_model.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/download_model.py) | Model download utility with streaming, atomic file replacement, SHA-256 verification, and overwrite protection. |

---

## 4. Detailed Implementation Changes

### 4.1 Updated `requirements.txt`
```text
# ==============================================================================
# RecipeGen AI — Core Runtime & Training Dependencies
#
# Environment Specifications:
# - Target Python Version: 3.11.x or 3.12.x (Verified: Python 3.12.8 on Apple Silicon CPU)
# - Caution: Python 3.14 is NOT supported by TensorFlow 2.21.0.
# - Mandatory Environment Variable: KERAS_BACKEND=tensorflow
# ==============================================================================

# Core Deep Learning & NLP Framework (Keras 3 + KerasHub)
tensorflow==2.21.0
keras==3.15.1
keras-hub==0.33.0
h5py==3.14.0

# Scientific & Data Processing
numpy==2.5.5
pandas==3.1.1

# Web Application & UI Framework
streamlit==1.67.0

# Image Processing & Visualization
pillow==13.0.0

# Network & Progress Utilities (Model Artifact Downloads & Checksum Verification)
requests==2.34.2
tqdm==4.70.1
```

### 4.2 Updated `.gitignore`
```gitignore
# Python environments
.venv/
.venv-ml/
venv/
venv_tf/

# Bytecode, cache & OS metadata
__pycache__/
*.py[cod]
*$py.class
.DS_Store
Thumbs.db

# Large Model Checkpoints (>100 MB rejected by standard GitHub git push)
# Weights should be handled via Git LFS or scripts/download_model.py
*.weights.h5
*.h5
*.keras
models/*.weights.h5
models/*/*.weights.h5

# Local model cache & temporary directories
.cache/
.keras/
kagglehub/
data/cache/
/tmp/
*.tmp
*.bak

# Large raw datasets (2.29 GB RecipeNLG raw CSV excluded from repo)
data/archive 2/
data/raw_dataset.csv
*.csv

# Preserve lightweight model configuration & metric history in repository
!models/**/training_config.json
!models/**/training_history.json
```

### 4.3 Robust Multi-Candidate Preset Discovery in `utils/gpt2.py`
```python
def get_cached_preset_path() -> str:
    """Return local cached preset directory if present and valid, else preset name."""
    candidate_paths = []
    if "KAGGLE_HUB_CACHE" in os.environ:
        candidate_paths.append(
            Path(os.environ["KAGGLE_HUB_CACHE"]) / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
        )
    candidate_paths.append(
        Path.home() / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
    )
    username = os.environ.get("USER") or os.environ.get("LOGNAME")
    if username:
        user_home = Path("/home") / username if sys.platform.startswith("linux") else Path("/Users") / username
        candidate_paths.append(
            user_home / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
        )
    candidate_paths.append(PROJECT_ROOT / "models" / "gpt2_base_en")

    for p in candidate_paths:
        if p.is_dir() and (p / "config.json").is_file():
            return str(p)

    return "gpt2_base_en"
```

### 4.4 Model Downloader & Verification Architecture (`scripts/download_model.py`)
- **Invocation:** Explicit execution only via `python scripts/download_model.py --url URL --dest DEST [--sha256 SHA256] [--overwrite]`.
- **Integrity Validation:** Computes SHA-256 and MD5 across 64KB blocks.
- **Safety Gate:** If `dest` exists and `--overwrite` / `-f` is omitted, the script exits immediately with error code 1 without initiating HTTP traffic.
- **Atomic Staging:** Downloads to `dest.with_suffix(".tmp_<timestamp>")`. If connection drops or checksum fails, the staging file is purged. It only moves to `dest` after checksum verification succeeds.

---

## 5. Verification Test Suite & Results

All tests were executed using `./venv_tf/bin/python` under standard CPU mode:

| Test Name | Command Executed | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Python Syntax Compilation** | `python -m py_compile [19 python files]` | 0 syntax errors across all code | All 19 files compiled cleanly | **PASSED** |
| **Exact Version Verification** | `python -c "assert tf.__version__ == '2.21.0'..."` | All 10 direct packages match | All 10 dependencies matched exactly | **PASSED** |
| **Full Checkpoint Integrity** | Python checksum script | MD5: `d95eb61d471f183a67a0e7fb920369d0` | `d95eb61d471f183a67a0e7fb920369d0` | **PASSED** |
| **Pilot Checkpoint Integrity** | Python checksum script | MD5: `a0ddb523b21e820c2fdcc871f2fae2a1` | `a0ddb523b21e820c2fdcc871f2fae2a1` | **PASSED** |
| **Stage 2D Checkpoint Integrity** | Python checksum script | MD5: `0f733e894f4a83087164dfc1b7fe9a9a` | `0f733e894f4a83087164dfc1b7fe9a9a` | **PASSED** |
| **JSON Configuration Parsing** | Python script iterating all JSON files | 13/13 valid JSON documents | 13/13 parsed successfully | **PASSED** |
| **Gitignore Pattern Simulation** | Pattern matching on weights & configs | Weights ignored; JSONs tracked | 12/12 test fixtures matched expected | **PASSED** |
| **Model Downloader CLI Interface** | `python scripts/download_model.py --help` | Display clean CLI arguments | Help text printed cleanly | **PASSED** |
| **Model Downloader Overwrite Refusal** | `python scripts/download_model.py --url ... --dest [existing]` | Reject with error code 1 | Exited code 1 with clear error message | **PASSED** |
| **Phase 1 Unit Tests** | `python -m unittest tests/test_phase1_fixes.py` | 10 passed | 10 passed in 0.001s | **PASSED** |
| **Phase 2 Integration Tests** | `python -m unittest tests/test_phase2_integration.py` | 5 passed (seq_len=512, closed tags) | 5 passed in 17.62s | **PASSED** |
| **Dataset Pipeline Fixture Test** | `python tests/test_dataset_pipeline.py` | Mock cleaning, deduplication & split | Passed: 9 kept, 1 duplicate removed | **PASSED** |
| **Split Verification on Processed Data** | `python scripts/validate_recipe_dataset.py` | 4500 train, 250 val, 250 test | All splits verified, 0 leakage | **PASSED** |
| **Dynamic Subdirectory Execution** | `cd scripts && python validate_recipe_dataset.py` | Works from `scripts/` directory | 4500 train, 250 val, 250 test verified | **PASSED** |
| **Streamlit Page Execution** | Python importlib inspection of all 4 pages | All 4 define valid `run()` | All 4 pages validated | **PASSED** |

---

## 6. Tests Not Executed & Technical Reasons

1. **Remote Model Download:** `scripts/download_model.py` was not executed against an actual remote endpoint because Phase 3 restrictions strictly forbid uploading or downloading model checkpoints over the network. Dry-run, help output, parameter validation, and local existence collision checks were verified.
2. **Git Repository Remote Push:** `git push` was not tested because no Git remote repository is configured, and uploading to external hosting is prohibited in Phase 3.
3. **Full Re-Training Run:** Full fine-tuning was intentionally omitted to adhere to strict safety rule #1 (avoiding redundant 164-minute compute cycles and preserving frozen weights).

---

## 7. Remaining Drawbacks & Phase 4 Scope

With Phase 3 complete, deployment and packaging blockers are resolved. The remaining project gaps are:

1. **Requirement U3 Gap — Ingredient NER:**
   `utils/ner.py` remains an 8-line rule-based comma-splitter (`text.split(',')`). Requirement U3 specifies a BERT-based NER model for extracting ingredient tokens from unstructured text.
2. **Requirement U3 Gap — Diffusion Dish Image Generation:**
   Dish visualization is currently a placeholder; no diffusion model or API is connected.
3. **Large Weight Hosting Decision:**
   Before publishing to a public Git repository, the user must choose between:
   - **Option A:** Host weights on Hugging Face Hub / Kaggle Models / Google Drive and use `scripts/download_model.py` on first run.
   - **Option B:** Track weights in Git via Git LFS (`git lfs install && git lfs track "*.weights.h5"`).

---

## 8. Rollback Instructions

If Phase 3 modifications need to be completely reverted:

```bash
# 1. Restore requirements.txt and .gitignore
cp requirements.txt.bak requirements.txt
cp .gitignore.bak .gitignore

# 2. Restore Python files
cp utils/gpt2.py.phase2.bak utils/gpt2.py
cp app/pages/4_Model_Training.py.phase1.bak app/pages/4_Model_Training.py
cp scripts/train_gpt2.py.bak scripts/train_gpt2.py
cp scripts/evaluate_test_set.py.bak scripts/evaluate_test_set.py
cp scripts/validate_recipe_dataset.py.bak scripts/validate_recipe_dataset.py
cp scripts/analyze_recipe_dataset.py.bak scripts/analyze_recipe_dataset.py
cp scripts/check_gpt2.py.bak scripts/check_gpt2.py
cp scripts/find_needed_files.py.bak scripts/find_needed_files.py
cp scripts/package_models.py.bak scripts/package_models.py
cp scripts/prepare_recipe_dataset.py.bak scripts/prepare_recipe_dataset.py
cp tests/test_dataset_pipeline.py.bak tests/test_dataset_pipeline.py

# 3. Remove new utility
rm scripts/download_model.py
```

---

## 9. Antigravity Certification

I certify that:
- No model checkpoint weights were trained, modified, renamed, or deleted.
- No processed dataset records (`recipes_train.jsonl`, `recipes_validation.jsonl`, `recipes_test.jsonl`) were modified.
- All modifications are backed up and verified with unit and regression tests.
- Execution is strictly CPU-bound on Apple Silicon (`venv_tf` / TensorFlow 2.21.0).
- Phase 4 has not been started. Awaiting user authorization.
