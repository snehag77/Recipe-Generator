# RecipeGen AI — Deployment & Portability Guide

This guide provides technical specifications, operational procedures, and containerization instructions for deploying **RecipeGen AI**.

---

## 1. Local Environment Execution (`venv_tf`)

### 1.1. Prerequisites
- **Operating System:** macOS (Apple Silicon arm64 / Intel) or Linux (Ubuntu 22.04+ / Debian 12+)
- **Python Version:** **Python 3.12.8** (Python 3.14 is incompatible with TensorFlow 2.21)
- **Host Architecture:** 64-bit CPU (Apple Silicon M-series or modern x86_64)

### 1.2. Environment Activation & Dependencies
If running in an existing project workspace:
```bash
# 1. Navigate to project root
cd /path/to/Recipe_genai

# 2. Activate dedicated TensorFlow virtual environment
source venv_tf/bin/activate

# 3. Verify installed core dependencies
pip list | grep -E "tensorflow|keras|keras-hub|streamlit"
```

Expected baseline output:
- `tensorflow == 2.21.0`
- `keras == 3.15.1`
- `keras-hub == 0.33.0`
- `streamlit == 1.67.0`

### 1.3. Launching the Streamlit Application
```bash
# Local development launch on custom port (e.g. 8503)
streamlit run app/main.py \
  --server.port 8503 \
  --server.headless true \
  --browser.gatherUsageStats false
```
The application will be accessible at `http://localhost:8503`.

---

## 2. Model Checkpoint Architecture & Locations

RecipeGen AI manages two distinct model weight components:

### 2.1. Stage 2F Fine-Tuned Checkpoint (Production Engine)
- **Path:** `models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5`
- **File Size:** `1,493,862,440` bytes (~1.49 GB / 1,424.66 MB)
- **MD5 Checksum:** `d95eb61d471f183a67a0e7fb920369d0`
- **Architecture:** 124M-parameter GPT-2 causal language model fine-tuned on 4,500 RecipeNLG recipes over 512-token sequence lengths.

### 2.2. Stage 2D Pilot Checkpoint (Rollback Option)
- **Path:** `models/gpt2_finetuned.weights.h5`
- **File Size:** `1,493,862,440` bytes (~1.49 GB / 1,424.66 MB)
- **MD5 Checksum:** `a0ddb523b21e820c2fdcc871f2fae2a1`
- **Role:** Preserved for rollback evaluation and historical comparison.

### 2.3. Checkpoint Verification Preflight
To verify the integrity of an existing checkpoint before launching or deploying:
```bash
python scripts/download_model.py \
  --verify-only \
  --dest models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5 \
  --expected-size 1493862440 \
  --md5 d95eb61d471f183a67a0e7fb920369d0
```

---

## 3. Base GPT-2 Preset Caching & Offline Deployment

KerasHub requires the base model architecture, configuration, and tokenizer vocabulary (`gpt2_base_en`, ~476.5 MB) before applying fine-tuned weights.

### 3.1. Preset Lookup Priority
The model loader (`utils/gpt2.py:get_cached_preset_path`) probes for local presets in this exact order:
1. **Local Project Directory:** `models/gpt2_base_en/` (or custom path via `KERAS_PRESET_DIR` / `GPT2_PRESET_DIR`).
2. **KaggleHub User Cache:** `~/.cache/kagglehub/models/keras/gpt2/keras/gpt2_base_en/3`.
3. **Environment Cache:** `$KAGGLE_HUB_CACHE` or `$XDG_CACHE_HOME`.
4. **Remote Preset Fallback:** `"gpt2_base_en"` (attempts public download if connected).

### 3.2. Optional Offline Preset Export (User-Invoked)
For 100% offline self-containment (e.g. air-gapped servers or Docker builds without internet access), run the export helper:

```bash
# 1. Preview source files and disk space requirements (~476.5 MB)
python scripts/export_base_preset.py --dry-run

# 2. Copy and cryptographically verify SHA-256 checksums
python scripts/export_base_preset.py
```
This copies:
- `models/gpt2_base_en/config.json`
- `models/gpt2_base_en/model.weights.h5` (475.08 MB)
- `models/gpt2_base_en/tokenizer.json`
- `models/gpt2_base_en/assets/tokenizer/merges.txt`
- `models/gpt2_base_en/assets/tokenizer/vocabulary.json`

Once exported, RecipeGen AI boots completely offline without querying external services.

---

## 4. Containerized Deployment (Docker)

### 4.1. Prerequisites
- Docker Engine 24.0+ installed on host
- Minimum host memory allocation: **4 GB RAM**

### 4.2. Build Image
```bash
# Build the production image tagged 'recipegen-ai:latest'
docker build -t recipegen-ai:latest .
```

### 4.3. Run Container
```bash
# Run container mapping host port 8501 to container port 8501
docker run -d \
  --name recipegen \
  -p 8501:8501 \
  --memory=4g \
  recipegen-ai:latest
```

### 4.4. Dynamic Port Configuration (Cloud Run / AWS ECS / Render)
The container dynamically binds to the `$PORT` environment variable:
```bash
docker run -d \
  -e PORT=8080 \
  -p 8080:8080 \
  --memory=4g \
  recipegen-ai:latest
```

---

## 5. System Requirements & Resource Sizing

### 5.1. Memory (RAM) Profile
The following table contrasts **verified runtime measurements** (measured via `resource.getrusage` on macOS Apple Silicon) against **deployment recommendations**:

| Operational State | Verified Process RSS Memory | Deployment Target Recommendation |
|---|:---:|:---:|
| **Interpreter Idle** | `13.1 MB` | 512 MB |
| **TensorFlow & Keras Loaded** | `15.4 MB` | 512 MB |
| **Stage 2F Model Loaded (512 seq len)** | **`1,379.1 MB` (~1.38 GB)** | **4.0 GB Host RAM** |
| **During Generation (Top-K sampling)** | **`1,398.5 MB` (~1.40 GB)** | **4.0 GB Host RAM** |
| **Dual Models (Model Comparison Page)** | **`1,867.4 MB` (~1.87 GB)** | **4.0 GB Host RAM** |

> [!WARNING]
> **Memory Allocation Advisory:**
> While the absolute peak resident memory measured locally was ~1.87 GB, **4 GB RAM is the recommended initial deployment target** (e.g. AWS `t3.medium`, GCP `e2-medium`, or Azure `Standard_B2s`).
> This recommendation accounts for OS overhead, Streamlit multi-user session state buffers, and memory fragmentation under Linux glibc.
> Deploying on micro instances with $\le 1\text{ GB}$ RAM (such as standard free tiers) will result in container crashes (`OOM-Killed`).

### 5.2. Disk Storage Requirements

| Tier | Component | Verified Size | Deployment Status |
|---|---|:---:|:---:|
| **Required** | Application Code & Config | ~0.5 MB | Included |
| **Required** | Processed Datasets (`data/processed/`) | 10.4 MB | Included |
| **Required** | Output Telemetry (`data/outputs/`) | 1.0 MB | Included |
| **Required** | Stage 2F Checkpoint (`recipe_gpt2_full_best.weights.h5`) | 1.49 GB | Included |
| **Optional** | Base GPT-2 Offline Preset (`models/gpt2_base_en/`) | 476.5 MB | Optional for offline mode |
| **Optional** | Stage 2D Rollback Checkpoint (`models/gpt2_finetuned.weights.h5`) | 1.49 GB | Excluded from lean image |
| **Excluded** | Raw RecipeNLG Dataset (`data/archive 2/RecipeNLG_dataset.csv`) | 2.14 GB | **Excluded via `.dockerignore`** |

- **Lean Production Container Size:** **~2.1 GB**
- **Complete Local Workspace (all weights & raw CSV):** **~6.8 GB**

### 5.3. Compute & Latency Characteristics
- **Inference Hardware:** Local CPU execution (Apple Silicon or modern x86_64).
- **Latency (Verified):** ~10–25 seconds per 512-token recipe generation.
- **Concurrency Recommendation:** Single-worker process per container instance. CPU saturation during autoregressive decoding makes horizontal container replication preferable over multi-threaded worker pooling.

---

## 6. Required & Supported Environment Variables

| Variable | Default Value | Purpose |
|---|:---:|---|
| `KERAS_BACKEND` | `tensorflow` | Enforces TensorFlow execution backend for Keras 3. |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Suppresses TensorFlow informational messages. |
| `PORT` | `8501` | Port on which Streamlit listens (useful for cloud environments). |
| `KERAS_HOME` | `<tempdir>/.keras` | Directory where Keras caches internal config and state. |
| `KERAS_PRESET_DIR` | `None` | Optional override path to a local GPT-2 preset directory. |
| `KAGGLE_HUB_CACHE` | `None` | Optional override path to KaggleHub cache root. |
| `MALLOC_TRIM_THRESHOLD_` | `100000` | Linux glibc allocator tuning to prevent memory fragmentation. |

---

## 7. Troubleshooting Common Deployment Issues

### 7.1. "Out of Memory" / SIGKILL (Exit Code 137)
- **Symptom:** Container abruptly terminates as soon as a user clicks "Generate Recipe" or accesses "Model Comparison".
- **Cause:** Host has less than 2 GB of physical RAM available, triggering Linux OOM-Killer.
- **Resolution:** Increase host memory to at least **4 GB RAM** (or configure 2 GB RAM + 2 GB swap).

### 7.2. "Failed to load base GPT-2 preset" / Connection Refused
- **Symptom:** `RuntimeError: Failed to load base GPT-2 preset from 'gpt2_base_en'`.
- **Cause:** Application is running in an air-gapped environment or container without external internet access, and no local preset was found.
- **Resolution:** Run `python scripts/export_base_preset.py` on a machine with the preset cached, and copy `models/gpt2_base_en/` into the deployment directory.

### 7.3. "Missing required checkpoint"
- **Symptom:** Application falls back to base model or reports checkpoint missing.
- **Cause:** Stage 2F checkpoint was excluded during packaging or Git cloning.
- **Resolution:** Verify file presence with `python scripts/find_needed_files.py`. If deploying from a remote storage bucket, run:
  ```bash
  python scripts/download_model.py \
    --url "https://your-storage-bucket.com/recipe_gpt2_full_best.weights.h5" \
    --dest "models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5" \
    --md5 "d95eb61d471f183a67a0e7fb920369d0"
  ```
