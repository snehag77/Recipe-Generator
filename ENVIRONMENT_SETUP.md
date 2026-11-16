# 🛠️ RecipeGen AI – Stage 2A Environment Setup & Compatibility Guide

This guide details the exact steps to create and configure a clean, modern virtual environment (`venv_tf`) for **TensorFlow**, **Keras 3**, and **KerasNLP / KerasHub** on Apple Silicon macOS.

---

## 1. Machine & Environment Analysis

- **System:** macOS 26.6.2 (Darwin arm64 – Apple Silicon)
- **RAM:** 16 GB unified memory
- **Free Disk Space:** ~177 GB
- **Existing `venv`:** Python 3.14 (Stage 1 Streamlit demo – **preserved untouched**)
- **Target `venv_tf`:** Dedicated Python 3.12 environment

---

## 2. Package Compatibility Selection

TensorFlow and Keras have evolved significantly:
- Starting in **TensorFlow 2.16+**, standard `pip install tensorflow` ships native ARM64 wheels for macOS; the deprecated `tensorflow-macos` package is no longer used or recommended.
- **Keras 3** is multi-backend (TensorFlow, PyTorch, JAX). We set `KERAS_BACKEND=tensorflow`.
- **KerasNLP** was unified into **KerasHub** (`keras-hub`), which is the modern distribution providing presets such as `gpt2_base_en`. Both `keras-nlp` and `keras-hub` packages share the same API architecture.
- **`tensorflow-metal`:** Optional Apple Metal acceleration plugin. If installed, it enables GPU acceleration on Apple Silicon. (If version conflicts occur, standard CPU inference runs reliably without it).

### Recommended Version Matrix (Python 3.12 on Apple Silicon):
| Component | Package / Version | Role |
| :--- | :--- | :--- |
| **Python** | 3.12 | Supported runtime for TensorFlow 2.16+ |
| **TensorFlow** | `tensorflow>=2.16.1` (or latest stable) | Deep learning backend |
| **Keras** | `keras>=3.3.0` | High-level neural network API |
| **KerasNLP / Hub**| `keras-hub>=0.19.0` (or `keras-nlp>=0.19.0`) | Model presets (GPT-2, BERT) |
| **Metal GPU** | `tensorflow-metal` (optional) | Apple Silicon acceleration |

---

## 3. Step-by-Step Installation on Your Mac

Run the following commands in your Mac terminal:

### Step 3.1: Install Python 3.12 (if not already installed)
Check if Python 3.12 is on your system:
```bash
python3.12 --version
```
If missing, install it via Homebrew:
```bash
brew install python@3.12
```
*(Or download the macOS Universal installer from [python.org](https://www.python.org/downloads/macos/)).*

---

### Step 3.2: Create and Activate `venv_tf`
Navigate to your project root and create the dedicated environment:
```bash
cd /Users/snehagupta/Downloads/Recipe_genai

# Create clean virtual environment using Python 3.12
python3.12 -m venv venv_tf

# Activate the environment
source venv_tf/bin/activate
```

---

### Step 3.3: Upgrade pip and Install Dependencies
```bash
pip install --upgrade pip setuptools wheel

# Install core TensorFlow and Keras stack
pip install tensorflow keras keras-hub

# Optional: Attempt Apple Metal acceleration
pip install tensorflow-metal
```

*(Note: If `tensorflow-metal` throws distribution or build errors, simply omit it. TensorFlow will use Apple Silicon's high-performance multi-threaded CPU instructions without issues).*

---

## 4. Verification and Sanity Testing

Two dedicated test scripts are prepared in the `scripts/` folder:

### Test 1: Verify TensorFlow & Keras
```bash
python scripts/check_tensorflow.py
```
**Expected Output:**
- Imports `tensorflow`, `keras`, and `keras_hub`.
- Reports version strings and active backend (`tensorflow`).
- Performs a matrix multiplication tensor operation.
- Lists physical compute devices (`CPU` and, if available, `GPU`).

### Test 2: Verify GPT-2 Preset & Base Inference
```bash
python scripts/check_gpt2.py
```
**Expected Output:**
- Resolves the `gpt2_base_en` (or `gpt2_small`) preset.
- Automatically caches pre-trained weights and preprocessor tokenizer on first run.
- Generates a sample recipe text continuation from the test prompt:
  `"Create a recipe using potato, onion and tomato."`
- Demonstrates working base-model autoregressive generation.

---

## 5. Returning to Stage 1 Demo
Because `venv_tf` is completely isolated, you can switch back to the Stage 1 demo at any time:
```bash
deactivate
source venv/bin/activate
streamlit run app/main.py
```
