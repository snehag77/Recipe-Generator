import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Primary Stage 2F Production Assets
PRIMARY_NEEDED = [
    "recipe_gpt2_full_best.weights.h5",
    "training_config.json",
    "recipe_gpt2_full_metrics.json",
    "stage2g_teacher_forced.json",
    "stage2g_generation_eval.json",
]

# Rollback Stage 2D Assets (Optional)
ROLLBACK_NEEDED = [
    "gpt2_finetuned.weights.h5",
    "finetuned_metrics.json",
]

ALL_NEEDED = PRIMARY_NEEDED + ROLLBACK_NEEDED


def search(root, names, skip=()):
    found = {}
    for dp, dn, fn in os.walk(root):
        if any(dp.startswith(s) for s in skip):
            dn[:] = []
            continue
        for n in fn:
            if n in names and n not in found:
                found[n] = os.path.join(dp, n)
    return found


def main():
    # 1) in current project directory
    found = search(
        str(PROJECT_ROOT),
        ALL_NEEDED,
        skip=(str(PROJECT_ROOT / "venv"), str(PROJECT_ROOT / "venv_tf")),
    )

    # 2) on Colab machine (if running on Colab)
    if os.path.isdir("/content"):
        for k, v in search("/content", ALL_NEEDED, skip=("/content/drive", "/content/sample_data")).items():
            found.setdefault(k, v)

    # 3) inside Google Drive, if it is mounted
    if os.path.isdir("/content/drive/MyDrive"):
        for k, v in search("/content/drive/MyDrive", ALL_NEEDED).items():
            found.setdefault(k, v)

    print("=" * 70)
    print("RecipeGen AI: Model & Asset Discovery Status")
    print("=" * 70)

    print("\n--- [Stage 2F Production Assets] ---")
    for n in PRIMARY_NEEDED:
        if n in found:
            size_mb = os.path.getsize(found[n]) / (1024 * 1024)
            print(f"✅ FOUND   {n:<35} ({size_mb:>8.2f} MB) -> {found[n]}")
        else:
            print(f"❌ MISSING {n:<35}")

    print("\n--- [Stage 2D Rollback Assets (Optional)] ---")
    for n in ROLLBACK_NEEDED:
        if n in found:
            size_mb = os.path.getsize(found[n]) / (1024 * 1024)
            print(f"ℹ️ FOUND   {n:<35} ({size_mb:>8.2f} MB) -> {found[n]}")
        else:
            print(f"⚠️ MISSING {n:<35} (Rollback not available)")


if __name__ == "__main__":
    main()
