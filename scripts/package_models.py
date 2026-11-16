import argparse
import os
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Primary Stage 2F Production Assets
PRIMARY_PRODUCTION = [
    ("models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5", "models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5"),
    ("models/recipe_gpt2_full/training_config.json", "models/recipe_gpt2_full/training_config.json"),
    ("models/recipe_gpt2_full/training_history.json", "models/recipe_gpt2_full/training_history.json"),
    ("data/outputs/recipe_gpt2_full_metrics.json", "data/outputs/recipe_gpt2_full_metrics.json"),
    ("data/outputs/stage2g_teacher_forced.json", "data/outputs/stage2g_teacher_forced.json"),
    ("data/outputs/stage2g_generation_eval.json", "data/outputs/stage2g_generation_eval.json"),
]

ROLLBACK_ASSETS = [
    ("models/gpt2_finetuned.weights.h5", "models/gpt2_finetuned.weights.h5"),
    ("data/outputs/finetuned_metrics.json", "data/outputs/finetuned_metrics.json"),
]


def main():
    parser = argparse.ArgumentParser(description="Package RecipeGen AI production models into a deployable archive.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "models" / "recipegen_production_models.zip"),
        help="Target zip archive path",
    )
    parser.add_argument(
        "--include-rollback",
        action="store_true",
        help="Include Stage 2D pilot rollback checkpoint (~1.49 GB additional)",
    )
    args = parser.parse_args()

    items_to_pack = list(PRIMARY_PRODUCTION)
    if args.include_rollback:
        items_to_pack.extend(ROLLBACK_ASSETS)

    # Verify all required files exist
    missing = []
    for rel_src, _ in items_to_pack:
        full_src = PROJECT_ROOT / rel_src
        if not full_src.is_file():
            missing.append(rel_src)

    if missing:
        print(f"❌ Cannot package models: missing required files: {missing}")
        raise SystemExit(1)

    out_zip = Path(args.output).resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RecipeGen AI: Packaging Model Artifacts")
    print("=" * 70)
    print(f"Output Archive: {out_zip}")
    print(f"Files to pack:  {len(items_to_pack)}")

    # Store without re-compression (weights are already dense HDF5)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for rel_src, arcname in items_to_pack:
            full_src = PROJECT_ROOT / rel_src
            size_mb = full_src.stat().st_size / (1024 * 1024)
            print(f"📦 Adding {rel_src} ({size_mb:.2f} MB)...")
            z.write(full_src, arcname=arcname)

    total_gb = out_zip.stat().st_size / (1024 * 1024 * 1024)
    print(f"\n✅ Created archive: {out_zip} ({total_gb:.2f} GB)")


if __name__ == "__main__":
    main()
