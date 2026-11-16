from google.colab import drive
import os
import shutil

def main():
    # Mount the user's Google Drive
    drive.mount("/content/drive")
    # Ensure the destination folder exists
    dst_dir = "/content/drive/MyDrive/P45_export"
    os.makedirs(dst_dir, exist_ok=True)
    # Copy the zip archive to Drive
    src = "/content/p45_models.zip"
    dst = os.path.join(dst_dir, "p45_models.zip")
    shutil.copy(src, dst)
    print(f"Saved to {dst}")

if __name__ == "__main__":
    main()
