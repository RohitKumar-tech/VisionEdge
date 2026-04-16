"""
Downloads datasets for VisionEdge fine-tuning:
  1. IMFDB — Indian Movie Face Database (34K images, 100 Indian actors)
  2. QMUL-SurvFace — Surveillance face recognition (463K images, 15K identities)

Run: python training/download_datasets.py

For IMFDB you need a Kaggle account:
  1. Go to https://www.kaggle.com/settings → API → Create New Token
  2. Save the downloaded kaggle.json to ~/.config/kaggle/kaggle.json
  3. Run this script
"""
import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"


def download_imfdb():
    """Download IMFDB via kagglehub."""
    print("\n── Downloading IMFDB (Indian Movie Face Database) ──────────────────────")

    kaggle_json = Path.home() / ".config" / "kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("  Kaggle credentials not found.")
        print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print(f"  2. Save kaggle.json to {kaggle_json}")
        print("  3. Re-run this script")
        return False

    import kagglehub
    print("  Downloading from Kaggle (may take a few minutes)...")
    path = kagglehub.dataset_download(
        "anirudhsimhachalam/indian-movie-faces-datasetimfdb-face-recognition"
    )
    dest = DATA / "imfdb"
    if Path(path) != dest:
        shutil.copytree(path, dest, dirs_exist_ok=True)
    print(f"  IMFDB saved to {dest}")
    return True


def download_qmul_survface():
    """
    QMUL-SurvFace — surveillance face recognition dataset.
    463,507 face images across 15,573 identities from real CCTV footage.
    Request access at: https://qmul-survface.github.io/
    """
    print("\n── QMUL-SurvFace ────────────────────────────────────────────────────────")
    print("  QMUL-SurvFace requires manual registration:")
    print("  1. Visit https://qmul-survface.github.io/")
    print("  2. Fill the request form with your institution email")
    print("  3. You'll receive a download link within 24-48 hours")
    print("  4. Download and extract to: training/data/qmul_survface/")
    print()
    print("  Why this dataset matters for VisionEdge:")
    print("  - 463K faces from real CCTV cameras (not studio photos)")
    print("  - Includes blur, low-res, bad lighting — exactly your use case")
    print("  - 15K identities = model learns to distinguish many people")


def check_imfdb_structure(data_dir: Path):
    """Verify IMFDB downloaded correctly and show stats."""
    if not data_dir.exists():
        return False
    # Dataset may be nested inside subdirectories (e.g. "IMFDB FR dataset/IMFDB FR dataset/")
    persons = [d for d in data_dir.rglob("*")
               if d.is_dir() and any(f.suffix.lower() in (".jpg", ".png")
                                     for f in d.iterdir() if f.is_file())]
    total_images = sum(len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
                       for p in persons)
    print(f"\n  IMFDB: {len(persons)} identities, {total_images} images")
    return len(persons) > 0


if __name__ == "__main__":
    print("VisionEdge Dataset Downloader")
    print("=" * 50)

    imfdb_ok = download_imfdb()
    download_qmul_survface()

    print("\n── Status ───────────────────────────────────────────────────────────────")
    imfdb_dir = DATA / "imfdb"
    if check_imfdb_structure(imfdb_dir):
        print("  ✓ IMFDB ready")
    else:
        print("  ✗ IMFDB not downloaded yet (set up Kaggle credentials)")
    print("  ✗ QMUL-SurvFace requires manual registration")
    print()
    print("  Next: python training/prepare_data.py")
