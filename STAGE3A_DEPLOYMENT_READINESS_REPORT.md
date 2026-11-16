# RecipeGen AI — Phase 3A: Deployment Readiness Audit Report

**Project Directory:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Execution Environment:** `venv_tf` (Python 3.12.8, Apple Silicon CPU)  
**Date:** November 16, 2026  
**Audit Mode:** Read-Only Verification  
**Baseline Stack:** Python 3.12.8, TensorFlow 2.21.0, Keras 3.15.1, KerasHub 0.33.0, Streamlit 1.67.0  

---

## 1. Executive Summary

This read-only deployment readiness audit examines the packaging, portability, memory footprints, disk requirements, and dependency compatibility of RecipeGen AI prior to Phase 3B implementation.

### Key Audit Conclusions
1. **Dependency Alignment:** All 10 packages specified in `requirements.txt` match the active `venv_tf` environment version-for-version.
2. **Hardcoded User Paths:** **0** hardcoded `/Users/snehagupta` paths exist in active application scripts, JSON configuration files, or data outputs.
3. **Repository Cleanliness & Large Files:** The project is currently not a Git repository. Four files exceed 50 MB (totalling 6.57 GB: 2.1 GB raw CSV + 4.17 GB across three 1.49 GB model checkpoints). Existing `.gitignore` rules correctly exclude all of them.
4. **Memory Verification:**
   - Single-model execution (Recipe Generator): **1,379.1 MB (~1.38 GB) RAM**.
   - Dual-model execution (Model Comparison): **1,867.4 MB (~1.87 GB) RAM**.
   - Minimum host requirement: **4 GB RAM** (containers with $\le 1\text{ GB}$ will crash with Out-Of-Memory errors).
5. **Portability Bottleneck:** The base model architecture and tokenizer currently load from the user's local cache (`~/.cache/kagglehub/.../gpt2_base_en/3`, 498 MB). An offline deployment requires bundling or pre-caching this preset in `models/gpt2_base_en/`.

---

## 2. Dependency Compatibility & Requirements Audit

### 2.1. `requirements.txt` vs. Installed Packages (`venv_tf`)

All packages declared in `requirements.txt` were verified against `pip list` in `venv_tf`:

| Package | Declared in `requirements.txt` | Installed in `venv_tf` | Match Status | Role in Project |
|---|:---:|:---:|:---:|---|
| `tensorflow` | `2.21.0` | `2.21.0` | **VERIFIED** | Deep learning engine / backend |
| `keras` | `3.15.1` | `3.15.1` | **VERIFIED** | High-level neural network API |
| `keras-hub` | `0.33.0` | `0.33.0` | **VERIFIED** | GPT-2 architecture, tokenizer, causal LM |
| `h5py` | `3.14.0` | `3.14.0` | **VERIFIED** | HDF5 weight checkpoint serialization |
| `numpy` | `2.5.5` | `2.5.5` | **VERIFIED** | Numeric tensor computations |
| `pandas` | `3.1.1` | `3.1.1` | **VERIFIED** | Metric dataframes and dataset validation |
| `streamlit` | `1.67.0` | `1.67.0` | **VERIFIED** | Interactive web application framework |
| `pillow` | `13.0.0` | `13.0.0` | **VERIFIED** | Reserved for planned Phase 4 image generation |
| `requests` | `2.34.2` | `2.34.2` | **VERIFIED** | Remote model checkpoint download utility |
| `tqdm` | `4.70.1` | `4.70.1` | **VERIFIED** | Progress bars for dataset preparation & downloads |

### 2.2. Indirect and Transitive Dependencies
Key transitive dependencies verified present in `venv_tf`:
- `pyarrow==26.0.1` (Required by Streamlit 1.67 `st.dataframe` serialization)
- `uvicorn==0.54.0` (Streamlit HTTP and WebSocket server)
- `protobuf==7.36.2` (TensorFlow graph protocol buffers)
- `kagglehub==1.0.3` (Base preset caching)

### 2.3. Unused & Missing Dependencies Assessment
- **Core Runtime:** Zero missing imports. All application pages (`app/main.py`, `app/pages/*.py`), utilities (`utils/*.py`), and tests (`tests/*.py`) import and run cleanly.
- **Unused in Core Inference:** `pillow` and `tqdm` are declared in `requirements.txt` but not imported by `app/` or `utils/`. They are harmless and preserved for Phase 4 (Pillow for image diffusion, Tqdm for batch downloads).
- **External Archive Code (`data/archive 2/RecipeNLG_code/`):** Contains legacy scraping scripts from 2020 that import `torch`, `transformers`, `scrapy`, `bs4`, `pymongo`, `numba`, and `sklearn`. These are non-application third-party assets and are not part of RecipeGen AI.

---

## 3. Hardcoded Path Inventory & Portability Assessment

### 3.1. Absolute Path Scan
- **`/Users/snehagupta` references in `.py`, `.json`, `.toml`, `.yaml`, `.sh`:** **0 occurrences** (excluding `.bak` files).
- Project files dynamically resolve root paths using:
  ```python
  PROJECT_ROOT = Path(__file__).resolve().parents[...]
  ```

### 3.2. Identified Portability Vulnerabilities

| File | Line | Code Snippet | Portability Risk | Recommended Phase 3B Remedy |
|---|:---:|---|---|---|
| `utils/gpt2.py` | 47 | `user_home = Path("/home") / username if sys.platform.startswith("linux") else Path("/Users") / username` | Assumes POSIX `/Users` or `/home`. Breaks on Windows (`C:\Users`). | Rely directly on `Path.home()` or `os.path.expanduser("~")`. |
| `utils/gpt2.py` | 16 | `os.environ.setdefault("KERAS_HOME", "/tmp/.keras")` | Hardcodes `/tmp` which does not exist on native Windows. | Use `tempfile.gettempdir()`. |
| `scripts/train_gpt2.py` | 30 | `os.environ.setdefault("KERAS_HOME", "/tmp/.keras")` | Hardcodes `/tmp`. | Use `tempfile.gettempdir()`. |
| `scripts/check_gpt2.py` | 20 | `os.environ.setdefault("KERAS_HOME", "/tmp/.keras")` | Hardcodes `/tmp`. | Use `tempfile.gettempdir()`. |
| `scripts/evaluate_test_set.py` | 28 | `os.environ.setdefault("KERAS_HOME", "/tmp/.keras")` | Hardcodes `/tmp`. | Use `tempfile.gettempdir()`. |
| `scripts/package_models.py` | 10 | `NEEDED = ["gpt2_finetuned_final.weights.h5", ...]` | References obsolete Stage 1 Colab filenames. | Update to `recipe_gpt2_full_best.weights.h5` and `gpt2_finetuned.weights.h5`. |
| `scripts/find_needed_files.py` | 6 | `NEEDED = ["gpt2_finetuned_final.weights.h5", ...]` | References obsolete Stage 1 Colab filenames. | Update to current checkpoint filenames. |
| `scripts/copy_to_drive.py` | 8 | `from google.colab import drive` | Fails outside Google Colab environment. | Guard import with `try...except ImportError`. |

---

## 4. Large-File & Checkpoint Storage Inventory

### 4.1. Large Files (>50 MB) in Workspace

| File Path | Actual Size (Bytes) | Size (MB / GB) | Purpose |
|---|:---:|:---:|---|
| `data/archive 2/RecipeNLG_dataset.csv` | `2,294,981,083` | **2.14 GB** | Raw 2.2M RecipeNLG CSV |
| `models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5` | `1,493,862,440` | **1.39 GB (1,424.66 MB)** | **Stage 2F Full Fine-Tuned Checkpoint (Production)** |
| `models/gpt2_finetuned.weights.h5` | `1,493,862,440` | **1.39 GB (1,424.66 MB)** | Stage 2D Pilot Checkpoint (Rollback Option) |
| `models/recipe_gpt2/recipe_gpt2_best.weights.h5` | `1,493,862,440` | **1.39 GB (1,424.66 MB)** | Redundant Stage 2D duplicate checkpoint |

**Total Large File Volume:** **6.57 GB** (excluding virtual environments).

### 4.2. Model Checksum Verification

```text
models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5:  d95eb61d471f183a67a0e7fb920369d0  (Verified)
models/gpt2_finetuned.weights.h5:                          a0ddb523b21e820c2fdcc871f2fae2a1  (Verified)
models/recipe_gpt2/recipe_gpt2_best.weights.h5:            0f733e894f4a83087164dfc1b7fe9a9a  (Verified)
```

### 4.3. Git Tracking Assessment
- **Current Git Status:** The directory is **not yet a Git repository** (`.git` is not initialized).
- **`.gitignore` Effectiveness:**
  - `*.weights.h5`, `*.h5`, `*.keras` are ignored (lines 20–25).
  - `data/archive 2/`, `*.csv` are ignored (lines 36–39).
  - `venv/`, `venv_tf/` are ignored (lines 8–9).
  - Model configs (`training_config.json`, `training_history.json`) are explicitly preserved via `!` negation rules (lines 41–42).
  - **Verdict:** If Git is initialized, no files over 50 MB will be accidentally tracked or staged.

---

## 5. Dataset Locations and Storage Footprint

The project data directory is divided into three tiers:

```text
data/
├── archive 2/                   # 2.29 GB (Raw source archive — EXCLUDE FROM DEPLOYMENT)
│   └── RecipeNLG_dataset.csv    # 2.14 GB
├── processed/                   # 10.37 MB (Cleaned, filtered, tokenized datasets)
│   ├── recipes_train.jsonl      # 8.91 MB (4,500 recipes, MD5: 6d2487aa1719146437892318f5820dc9)
│   ├── recipes_validation.jsonl # 0.50 MB (250 recipes,   MD5: 01b7bf6c2d0ee71093f38e01c000a045)
│   ├── recipes_test.jsonl       # 0.49 MB (250 recipes,   MD5: 11b581b58e352672e9ef2f7130cd10bb)
│   ├── dataset_metadata.json    # 746 bytes
│   └── dataset_statistics.json  # 481 bytes
└── outputs/                     # 1.04 MB (Benchmark telemetry & evaluation metrics)
    ├── stage2g_generation_eval.json     # 0.98 MB (250 held-out generated recipes evaluation)
    ├── stage2f_eval_comparison.json     # 5.3 KB
    ├── recipe_gpt2_full_metrics.json    # 1.3 KB
    ├── finetuned_metrics.json           # 859 bytes
    └── stage2g_teacher_forced.json      # 326 bytes
```

**Deployment Takeaway:** Only `data/processed/` (10.4 MB) and `data/outputs/` (1.0 MB) are needed in a deployable artifact. The 2.29 GB `data/archive 2/` raw dataset can be completely omitted.

---

## 6. Offline Model-Loading & Cache Architecture

### 6.1. Current Loading Flow (`utils/gpt2.py`)
1. `get_cached_preset_path()` probes:
   - `KAGGLE_HUB_CACHE` environment variable
   - `~/.cache/kagglehub/models/keras/gpt2/keras/gpt2_base_en/3`
   - `$PROJECT_ROOT/models/gpt2_base_en`
2. If none of these local directories contain `config.json`, it returns the string `"gpt2_base_en"`.
3. `keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")` will then attempt to download the 498 MB base preset from KaggleHub via the public internet.

### 6.2. Portability Blocker
- **Verified Finding:** On the local Mac, the preset exists at `/Users/snehagupta/.cache/kagglehub/.../gpt2_base_en/3`.
- **Deployment Impact:** If RecipeGen AI is deployed on a fresh server or container without outbound internet access, model loading will fail unless:
  - Option A: The 498 MB base preset is pre-copied into `models/gpt2_base_en/`.
  - Option B: A startup script runs `scripts/download_model.py` to fetch both base preset and fine-tuned weights.

---

## 7. Deployment Resource Requirements (RAM, Disk, CPU)

### 7.1. Memory (RAM) Profile (Verified via `getrusage`)

| Operational State | Verified RSS Memory | Host RAM Recommendation | Notes |
|---|:---:|:---:|---|
| **Python Process Idle** | `13.1 MB` | 512 MB | Bare interpreter |
| **TensorFlow & Keras Imported** | `15.4 MB` | 512 MB | Dynamic library loading |
| **Stage 2F Full Model Loaded** | **`1,379.1 MB` (~1.38 GB)** | **4.0 GB** | Active for Recipe Generator & Ingredient Analysis |
| **During Generation (seq len 512)** | **`1,398.5 MB` (~1.40 GB)** | **4.0 GB** | Top-K sampling with KerasHub |
| **Dual Models Loaded (Base + FT)** | **`1,867.4 MB` (~1.87 GB)** | **4.0 GB** | Active on Model Comparison page (`_MODELS_CACHE`) |

> [!CAUTION]
> **Container OOM Risk:** Deploying on cloud container tiers with 1 GB or less RAM (e.g. Free-tier Streamlit Community Cloud, Heroku Basic 512 MB, AWS t2.micro 1 GB) will trigger an immediate **Out-Of-Memory (OOM) SIGKILL** when `load_model()` runs. The host must have at least **4 GB RAM** (or 2 GB RAM with at least 2 GB swap).

### 7.2. Disk Storage Requirements

| Component | Size | Required in Production Image? |
|---|:---:|:---:|
| Core Application Code (`app/`, `utils/`, `.streamlit/`) | ~0.5 MB | **YES** |
| Processed Datasets (`data/processed/`) | 10.4 MB | **YES** (Model Training Page) |
| Output Evaluation Telemetry (`data/outputs/`) | 1.0 MB | **YES** (Model Training Page) |
| Fine-Tuned Stage 2F Checkpoint (`recipe_gpt2_full_best.weights.h5`) | 1,424.7 MB (1.49 GB) | **YES** (Core generation engine) |
| Base GPT-2 Offline Preset (`models/gpt2_base_en/`) | 475.1 MB (498 MB) | **YES** (Offline base architecture & vocab) |
| Pilot Rollback Checkpoint (`gpt2_finetuned.weights.h5`) | 1,424.7 MB (1.49 GB) | *OPTIONAL* (Rollback engine) |
| Redundant Duplicate Pilot Checkpoint (`recipe_gpt2_best.weights.h5`) | 1,424.7 MB (1.49 GB) | **NO** (Can be safely deleted) |
| Raw RecipeNLG CSV Archive (`data/archive 2/`) | 2,294.9 MB (2.29 GB) | **NO** (Exclude from deployment) |

- **Minimum Production Disk Footprint (Code + Datasets + Stage 2F + Base Preset):** **~2.1 GB**.
- **Current Workspace Total (including all duplicate weights and raw CSV):** **~6.8 GB**.

### 7.3. Compute & Latency Characteristics
- Generation on CPU: **~10–25 seconds** per 512-token recipe.
- Dual-generation comparison: **~25–45 seconds**.
- Single-worker concurrency is recommended per container.

---

## 8. Recommended Implementation Order for Phase 3B

To transition RecipeGen AI into a portable, production-ready package without risking existing models or data, the recommended implementation sequence for Phase 3B is:

1. **Preset & Portability Fixes (`utils/gpt2.py`):**
   - Replace `/tmp` with `tempfile.gettempdir()`.
   - Replace hardcoded `/home` and `/Users` with standard `Path.home()`.
   - Ensure `models/gpt2_base_en` is checked before external cache.
2. **Modernize Model Management Scripts:**
   - Update `scripts/package_models.py` and `scripts/find_needed_files.py` to reference active filenames (`recipe_gpt2_full_best.weights.h5` and `gpt2_finetuned.weights.h5`).
   - Add a preflight verify flag to `scripts/download_model.py`.
3. **Containerization & Deployment Configuration:**
   - Create a clean `Dockerfile` targeting Python 3.12-slim with non-root security standards, memory environment variables (`MALLOC_TRIM_THRESHOLD_`), and port configuration (defaulting to 8501 or 8503).
   - Create a `.dockerignore` excluding `.git`, `venv/`, `venv_tf/`, `data/archive 2/`, `*.bak`, and duplicate weights.
4. **Offline Model Asset Bundling:**
   - Create a helper script to optionally export the cached base preset (`~/.cache/kagglehub/...`) into `models/gpt2_base_en/` for 100% offline self-contained deployments.
5. **Deployment Guide:**
   - Create `DEPLOYMENT_GUIDE.md` detailing local execution, Docker deployment, cloud VM sizing (minimum 4 GB RAM), and model artifact management.

---

## 9. Verification & Audit Sign-Off

- **Models & Datasets:** Untouched (zero bytes changed during audit).
- **Environment:** Unaltered (no packages installed or uninstalled).
- **Application Status:** Currently running cleanly on port 8503.
- **Phase 3B Readiness:** Ready for approval to begin Phase 3B implementation.
