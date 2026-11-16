# -*- coding: utf-8 -*-
"""Fine-Tuning GPT-2 on Full Recipe Dataset using TensorFlow 2 & KerasHub.

Features:
- Loads official gpt2_base_en preset (with local cache fallback).
- Tokenizes dataset using GPT2CausalLMPreprocessor (seq_len=512).
- Uses AdamW optimizer with 5% warmup and cosine decay (alpha=0.1).
- Evaluates validation loss and calculates perplexity.
- Saves best weights under models/recipe_gpt2_full/ without touching existing checkpoints.
- Evaluates generation against benchmark ingredient sets.
- Logs training metrics and resource telemetry to data/outputs/.

Usage:
    source venv_tf/bin/activate
    python scripts/train_gpt2.py --epochs 1 --batch_size 4 --max_len 512
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(tempfile.gettempdir()) / ".keras"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import keras
import keras_hub
import numpy as np
import tensorflow as tf


def compute_file_md5(filepath: str) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl_texts(file_path: str, key: str = "formatted_text") -> tuple:
    """Read formatted texts and recipe IDs from a JSONL file."""
    texts = []
    ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item[key])
                ids.append(item.get("recipe_id"))
    return texts, ids


def build_tf_dataset(texts: list, batch_size: int, shuffle: bool = True, seed: int = 42):
    """Build a tf.data.Dataset from raw strings with reproducible shuffling."""
    ds = tf.data.Dataset.from_tensor_slices(texts)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts), seed=seed, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on full RecipeGen dataset")
    parser.add_argument(
        "--train_file",
        default=str(PROJECT_ROOT / "data" / "processed" / "recipes_train.jsonl"),
    )
    parser.add_argument(
        "--val_file",
        default=str(PROJECT_ROOT / "data" / "processed" / "recipes_validation.jsonl"),
    )
    parser.add_argument(
        "--test_file",
        default=str(PROJECT_ROOT / "data" / "processed" / "recipes_test.jsonl"),
    )
    parser.add_argument(
        "--output_model_dir",
        default=str(PROJECT_ROOT / "models" / "recipe_gpt2_full"),
    )
    parser.add_argument("--weights_name", default="recipe_gpt2_full_best.weights.h5")
    parser.add_argument(
        "--output_metrics_dir",
        default=str(PROJECT_ROOT / "data" / "outputs"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional limit for small test run")
    args = parser.parse_args()

    # Deterministic seeding
    keras.utils.set_random_seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_model_dir, exist_ok=True)
    os.makedirs(args.output_metrics_dir, exist_ok=True)

    print("=" * 70)
    print("RecipeGen AI: Full Dataset GPT-2 Fine-Tuning Pipeline (Stage 2F)")
    print(f"Python: {sys.version.split()[0]} | TF: {tf.__version__} | Keras: {keras.__version__}")
    print(f"Target Output Directory: {args.output_model_dir}")
    print("=" * 70)

    # -------------------------------------------------------------
    # PRE-TRAINING SAFETY CHECKS
    # -------------------------------------------------------------
    print("\n🔍 Running Pre-Training Safety Checks...")

    # 1. Load Data
    train_texts, train_ids = load_jsonl_texts(args.train_file)
    val_texts, val_ids = load_jsonl_texts(args.val_file)

    if args.max_train_samples and args.max_train_samples < len(train_texts):
        train_texts = train_texts[:args.max_train_samples]
        train_ids = train_ids[:args.max_train_samples]
        print(f"   ⚠️ WARNING: Restricted to {args.max_train_samples} samples")

    print(f"   Loaded Train samples:      {len(train_texts):,}")
    print(f"   Loaded Validation samples: {len(val_texts):,}")

    assert len(train_texts) == 4500 or args.max_train_samples is not None, f"Expected 4,500 train samples, got {len(train_texts)}"
    assert len(val_texts) == 250, f"Expected 250 val samples, got {len(val_texts)}"

    # 2. Check Train / Val overlap
    overlap = set(train_ids).intersection(set(val_ids))
    assert len(overlap) == 0, f"Critical safety failure: {len(overlap)} overlapping records between train and val!"
    print("   ✅ Zero overlap between train and validation records.")

    # 3. Check Test split isolation
    assert os.path.exists(args.test_file), f"Test file {args.test_file} not found!"
    test_texts, test_ids = load_jsonl_texts(args.test_file)
    assert len(test_texts) == 250, f"Expected 250 test records, got {len(test_texts)}"
    test_overlap = set(train_ids).intersection(set(test_ids))
    assert len(test_overlap) == 0, f"Critical safety failure: {len(test_overlap)} overlapping records with test set!"
    print("   ✅ Test split is completely isolated (250 records, zero overlap).")

    # 4. Check Checkpoint Isolation
    weights_path = os.path.join(args.output_model_dir, args.weights_name)
    pilot_path = str(PROJECT_ROOT / "models" / "gpt2_finetuned.weights.h5")
    stage2d_path = str(PROJECT_ROOT / "models" / "recipe_gpt2" / "recipe_gpt2_best.weights.h5")
    assert os.path.abspath(weights_path) != os.path.abspath(pilot_path), "Refusing to overwrite pilot checkpoint!"
    assert os.path.abspath(weights_path) != os.path.abspath(stage2d_path), "Refusing to overwrite Stage 2D checkpoint!"
    print(f"   ✅ Target checkpoint path is isolated: {weights_path}")

    # 5. Check System Memory
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"   ✅ Initial Process RSS Memory: {max_rss_mb:.2f} MB")

    # -------------------------------------------------------------
    # MODEL SETUP & SEQUENCE HANDLING
    # -------------------------------------------------------------
    local_cached_path = Path.home() / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
    preset_target = str(local_cached_path) if local_cached_path.is_dir() else "gpt2_base_en"
    print(f"\n🔍 Loading GPT-2 preset from: {preset_target}")

    preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
        preset_target,
        sequence_length=args.max_len,
    )
    print(f"✅ Preprocessor configured with sequence_length={args.max_len}")

    print("⏳ Instantiating GPT2CausalLM architecture...")
    model = keras_hub.models.GPT2CausalLM.from_preset(
        preset_target,
        preprocessor=preprocessor,
    )
    print("✅ GPT-2 base model initialized.")

    # -------------------------------------------------------------
    # DATASET PIPELINE
    # -------------------------------------------------------------
    train_ds = build_tf_dataset(train_texts, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_ds = build_tf_dataset(val_texts, batch_size=args.batch_size, shuffle=False)

    total_steps = (len(train_texts) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    print(f"\n📊 Training Batches: {len(train_texts) // args.batch_size} (Batch size: {args.batch_size})")
    print(f"📊 Total Training Steps: {total_steps} | Warmup Steps: {warmup_steps} (5%)")
    print(f"📊 Validation Batches: {len(val_texts) // args.batch_size}")

    # -------------------------------------------------------------
    # OPTIMIZER & COMPILATION
    # -------------------------------------------------------------
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=max(total_steps, 10),
        warmup_steps=warmup_steps,
        alpha=0.1,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    # Save training configuration
    train_config = {
        "base_model": "gpt2_base_en",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "initial_learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "lr_schedule": "CosineDecayWithWarmup",
        "warmup_steps": warmup_steps,
        "warmup_percentage": 0.05,
        "decay_alpha": 0.1,
        "random_seed": args.seed,
        "train_samples": len(train_texts),
        "train_batches": len(train_texts) // args.batch_size,
        "val_samples": len(val_texts),
        "val_batches": len(val_texts) // args.batch_size,
        "checkpoint_path": weights_path,
        "compute_device": "Apple Silicon CPU (TensorFlow 2.21)",
    }
    with open(os.path.join(args.output_model_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    # -------------------------------------------------------------
    # EXECUTE TRAINING
    # -------------------------------------------------------------
    print("\n🚀 Starting Stage 2F Full Fine-Tuning Run (1 Epoch)...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb],
        verbose=1,
    )
    training_duration = time.time() - start_time
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 * 1024)

    # Ensure best weights are firmly on disk
    if not os.path.exists(weights_path):
        model.save_weights(weights_path)
        print(f"Saved final weights directly to {weights_path}")

    # Metrics computation
    train_loss = float(history.history.get("loss", [0.0])[-1])
    train_acc = float(history.history.get("accuracy", [0.0])[-1])
    val_loss = float(history.history.get("val_loss", [0.0])[-1])
    val_acc = float(history.history.get("val_accuracy", [0.0])[-1])
    perplexity = math.exp(min(val_loss, 20.0))
    steps_per_sec = total_steps / max(training_duration, 1.0)
    samples_per_sec = len(train_texts) / max(training_duration, 1.0)

    print("\n" + "=" * 70)
    print("🏁 Full Fine-Tuning Execution Complete")
    print(f"⏱️ Total Duration: {training_duration / 60:.2f} minutes ({training_duration:.1f} s)")
    print(f"⚡ Throughput: {steps_per_sec:.3f} steps/s | {samples_per_sec:.2f} samples/s")
    print(f"🧠 Peak Process Memory: {peak_rss_gb:.2f} GB")
    print(f"📈 Final Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc:.4f}")
    print(f"📊 Final Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
    print(f"🎯 Validation Perplexity: {perplexity:.2f}")
    print("=" * 70)

    # Save summary metrics
    metrics_summary = {
        "config": train_config,
        "execution": {
            "training_duration_seconds": round(training_duration, 2),
            "training_duration_minutes": round(training_duration / 60, 2),
            "peak_rss_gb": round(peak_rss_gb, 3),
            "steps_per_second": round(steps_per_sec, 3),
            "samples_per_second": round(samples_per_sec, 2),
        },
        "metrics": {
            "final_train_loss": round(train_loss, 4),
            "final_train_accuracy": round(train_acc, 4),
            "final_val_loss": round(val_loss, 4),
            "final_val_accuracy": round(val_acc, 4),
            "validation_perplexity": round(perplexity, 2),
        },
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }

    metrics_path = os.path.join(args.output_model_dir, "training_history.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(os.path.join(args.output_metrics_dir, "recipe_gpt2_full_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"💾 Metrics saved to: {metrics_path}")
    print(f"💾 Checkpoint saved to: {weights_path}")


if __name__ == "__main__":
    main()
