# Stage 2A Environment & Compatibility Audit Report

**Project Location:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Date & Time:** 2026-11-16 14:28 IST  
**Status:** Environment Analysis & Verification Scripts Complete (STOP Point)

---

## 1. System & Hardware Inspection

| Parameter | Value / Finding |
| :--- | :--- |
| **Operating System** | macOS 26.6.2 (Darwin Kernel arm64) |
| **Processor Architecture** | Apple Silicon (`arm64`, T8132 - MacBook Air) |
| **System Memory (RAM)** | 16 GB unified RAM |
| **Available Disk Space** | 177 GB free on `/` |
| **System Python Versions** | Python 3.14.7 (in `/Library/Frameworks/...`), Python 3.9.6 (`/usr/bin/python3`) |
| **Python 3.12 Status** | Not currently detected in standard system search paths |
| **Virtual Environments** | `venv` (Python 3.14, Stage 1 demo preserved) |
| **Target Environment** | `venv_tf` (To be created with Python 3.12) |
| **Network Connectivity** | Sandboxed execution environment has blocked external internet access (`[Errno 8] nodename nor servname provided`). All package downloads must be executed in your local Mac terminal. |

---

## 2. Package & Compatibility Analysis

1. **TensorFlow Distribution:**
   - Legacy `tensorflow-macos` releases were pinned for older versions (up to 2.15 on Python 3.9–3.11).
   - In modern releases (TF 2.16+), regular `tensorflow` wheels include native ARM64 / macOS builds and officially support **Python 3.12**.
   - Attempting to install `tensorflow-macos==2.15.0` on Python 3.12 fails with distribution errors because that wheel was never built for Python 3.12. Therefore, standard `tensorflow` is the correct package.
2. **Keras & Backend:**
   - Keras 3 (`keras>=3.3.0`) supports multiple backends via `KERAS_BACKEND=tensorflow`.
3. **KerasNLP vs KerasHub:**
   - KerasNLP has transitioned into **KerasHub** (`keras-hub`), which provides the official `GPT2CausalLM` architecture and presets (`gpt2_base_en`).
   - The test script [`scripts/check_gpt2.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/check_gpt2.py) supports both `keras_hub` and `keras_nlp` interchangeably.
4. **Apple Metal GPU:**
   - `tensorflow-metal` provides graph-level Metal acceleration on Apple Silicon. It is optional; if binary compatibility issues occur during installation, TensorFlow CPU operates reliably for inference and recipe tasks.

---

## 3. Scripts Created & Verified

Both verification scripts have been created and verified with `py_compile`:

1. [`scripts/check_tensorflow.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/check_tensorflow.py):
   - Sets `KERAS_BACKEND="tensorflow"` and `TF_CPP_MIN_LOG_LEVEL="2"`.
   - Tests imports: `tensorflow`, `keras`, and `keras_hub`/`keras_nlp`.
   - Runs a 2x2 matrix multiplication tensor operation.
   - Lists physical devices and checks for Metal GPU presence.
2. [`scripts/check_gpt2.py`](file:///Users/snehagupta/Downloads/Recipe_genai/scripts/check_gpt2.py):
   - Loads the official GPT-2 preset (`gpt2_base_en`).
   - Verifies preprocessor tokenizer and causal LM loading.
   - Executes base-model generation with test prompt:
     `"Create a recipe using potato, onion and tomato."`
   - Validates non-empty output generation.
3. [`ENVIRONMENT_SETUP.md`](file:///Users/snehagupta/Downloads/Recipe_genai/ENVIRONMENT_SETUP.md):
   - Complete guide with step-by-step installation instructions for your Mac terminal.

---

## 4. Exact Commands to Run on Your Mac

Because the Antigravity sandbox blocks outgoing package downloads, execute these steps directly in your Mac terminal:

```bash
# 1. Ensure Python 3.12 is installed (install via Homebrew if needed)
brew install python@3.12

# 2. Navigate to project root
cd /Users/snehagupta/Downloads/Recipe_genai

# 3. Create the dedicated virtual environment
python3.12 -m venv venv_tf

# 4. Activate venv_tf
source venv_tf/bin/activate

# 5. Upgrade build tooling
pip install --upgrade pip setuptools wheel

# 6. Install the validated TensorFlow & Keras stack
pip install tensorflow keras keras-hub

# 7. (Optional) Install Metal GPU plugin
pip install tensorflow-metal

# 8. Run Verification Script 1: TensorFlow & Keras
python scripts/check_tensorflow.py

# 9. Run Verification Script 2: GPT-2 Base Preset & Inference
python scripts/check_gpt2.py
```

---

## 5. Next Steps

- As instructed, I have stopped here.
- Please run the commands above in your Mac terminal and share the output of `check_tensorflow.py` and `check_gpt2.py`.
- Once verified, we will proceed to dataset preparation and GPT-2 fine-tuning.
