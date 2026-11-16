# -*- coding: utf-8 -*-
"""GPT-2 Base Model & Preprocessor sanity-check script for KerasNLP / KerasHub.

Tests loading the official GPT-2 preset, preprocessor tokenization,
and base-model inference without fine-tuning.

Usage:
    source venv_tf/bin/activate
    python scripts/check_gpt2.py
"""

import os
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(tempfile.gettempdir()) / ".keras"))


def main():
    print("=" * 60)
    print("RecipeGen AI: GPT-2 Base Preset & Inference Test")
    print("=" * 60)

    # 1. Resolve KerasNLP or KerasHub
    nlp_pkg = None
    try:
        import keras_nlp
        nlp_pkg = keras_nlp
        pkg_name = f"keras_nlp ({keras_nlp.__version__})"
    except ImportError:
        try:
            import keras_hub
            nlp_pkg = keras_hub
            pkg_name = f"keras_hub ({keras_hub.__version__})"
        except ImportError:
            print("❌ Neither keras_nlp nor keras_hub is installed in this environment.")
            print("   Please run: pip install keras-hub (or keras-nlp)")
            sys.exit(1)

    print(f"✅ Using NLP library: {pkg_name}")

    # 2. Inspect GPT-2 Causal LM class
    try:
        GPT2CausalLM = nlp_pkg.models.GPT2CausalLM
    except AttributeError:
        print("❌ Could not locate GPT2CausalLM in the installed NLP library.")
        sys.exit(1)

    preset_name = "gpt2_base_en"
    local_cached_path = Path.home() / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"

    model = None
    # Check if weights were already downloaded locally to kagglehub cache
    if local_cached_path.is_dir() and (local_cached_path / "model.weights.h5").is_file():
        print(f"🔍 Found local cached weights at: {local_cached_path}")
        print("⏳ Loading GPT-2 model and preprocessor from local cache...")
        try:
            model = GPT2CausalLM.from_preset(str(local_cached_path))
            print("✅ Model and preprocessor loaded successfully from local cache.")
        except Exception as e:
            print(f"⚠️ Local cache load failed: {e}. Falling back to preset name '{preset_name}'...")

    if model is None:
        print(f"🔍 Loading preset '{preset_name}' (first run downloads weights from Kaggle)...")
        try:
            model = GPT2CausalLM.from_preset(preset_name)
            print("✅ Model and preprocessor loaded successfully from preset.")
        except Exception as e:
            print(f"❌ Failed to load model preset '{preset_name}': {e}")
            print("   If network access is blocked, ensure preset weights are accessible.")
            sys.exit(1)

    # 3. Test prompt generation
    test_prompt = "Create a simple recipe using potato, onion and tomato."
    print(f"\n📝 Test Prompt:\n   \"{test_prompt}\"")
    print("⏳ Generating text (max 80 tokens, temperature 0.8)...")

    try:
        output = model.generate(test_prompt, max_length=80)
        # Handle string or list/tensor return types
        if isinstance(output, (list, tuple)):
            output_text = str(output[0])
        elif hasattr(output, "numpy"):
            output_text = str(output.numpy())
        else:
            output_text = str(output)

        print("\n✅ Generated Output (Base Model Sample):")
        print("-" * 50)
        print(output_text.strip())
        print("-" * 50)

        if len(output_text.strip()) > len(test_prompt):
            print("\n✅ Verification PASSED: GPT-2 base model generated non-empty continuation.")
        else:
            print("\n⚠️ Generation returned but output seems truncated or matches prompt exactly.")

    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("GPT-2 Base Model Check Complete. (Model is NOT fine-tuned).")
    print("=" * 60)


if __name__ == "__main__":
    main()
