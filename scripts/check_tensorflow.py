# -*- coding: utf-8 -*-
"""TensorFlow, Keras, and KerasNLP / KerasHub environment sanity-check script.

Usage:
    source venv_tf/bin/activate
    python scripts/check_tensorflow.py
"""

import os
import sys

# Ensure Keras backend is TensorFlow unless explicitly overridden
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main():
    print("=" * 60)
    print("RecipeGen AI: TensorFlow & Keras Verification")
    print("=" * 60)

    # 1. Imports
    try:
        import tensorflow as tf
    except Exception as e:
        print(f"❌ Failed to import tensorflow: {e}")
        sys.exit(1)

    try:
        import keras
    except Exception as e:
        print(f"❌ Failed to import keras: {e}")
        sys.exit(1)

    keras_nlp_mod = None
    try:
        import keras_nlp
        keras_nlp_mod = ("keras_nlp", keras_nlp.__version__)
    except ImportError:
        try:
            import keras_hub as keras_nlp
            keras_nlp_mod = ("keras_hub (successor to keras_nlp)", keras_nlp.__version__)
        except ImportError:
            pass

    # 2. Versions
    print(f"✅ Python:       {sys.version.split()[0]}")
    print(f"✅ TensorFlow:   {tf.__version__}")
    backend_name = getattr(keras.backend, "backend", lambda: "tensorflow")()
    print(f"✅ Keras:        {keras.__version__} (backend: {backend_name})")
    if keras_nlp_mod:
        print(f"✅ {keras_nlp_mod[0]}: {keras_nlp_mod[1]}")
    else:
        print("⚠️ Neither keras_nlp nor keras_hub is installed yet.")

    # 3. Basic Tensor Operation
    try:
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("\n✅ Basic Tensor Operation (Matrix Multiplication):")
        print(f"   Shape: {c.shape}, Result sum: {float(tf.reduce_sum(c)):g}")
    except Exception as e:
        print(f"❌ Tensor operation failed: {e}")
        sys.exit(1)

    # 4. Device Listing
    print("\n🔍 Physical Devices:")
    devices = tf.config.list_physical_devices()
    for d in devices:
        print(f"   - {d.device_type}: {d.name}")

    # 5. Metal GPU Acceleration Check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n⚡ Metal GPU Acceleration Active ({len(gpus)} GPU detected):")
        for g in gpus:
            print(f"   - {g.name}")
    else:
        print("\nℹ️ No Metal GPU detected by TensorFlow. Running in standard CPU mode.")
        print("   (Note: CPU mode is completely functional for inference and recipe tasks).")

    print("\n" + "=" * 60)
    print("All fundamental TensorFlow & Keras checks PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
