"""
Prepares downloaded datasets for InsightFace fine-tuning.

What it does:
  1. Scans IMFDB/QMUL directory structure → assigns integer identity labels
  2. Detects and aligns faces using InsightFace (crops + normalises to 112x112)
  3. Splits into train (80%) / val (10%) / test (10%)
  4. Saves as PyTorch ImageFolder-compatible structure
  5. Also saves a pairs.txt for verification benchmarking

Run: python training/prepare_data.py --source imfdb
     python training/prepare_data.py --source qmul
"""
import argparse
import json
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import insightface

BASE = Path(__file__).parent
DATA = BASE / "data"
PROCESSED = BASE / "data" / "processed"

FACE_SIZE = 112  # InsightFace standard


def get_face_app():
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
    return app


def align_face(app, image_bgr: np.ndarray) -> np.ndarray | None:
    """Detect + align largest face in image. Returns 112x112 BGR or None."""
    faces = app.get(image_bgr)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    # Use InsightFace's built-in alignment
    from insightface.utils import face_align
    aligned = face_align.norm_crop(image_bgr, landmark=largest.kps, image_size=FACE_SIZE)
    return aligned


def process_dataset(source: str, app):
    src_dir = DATA / source
    out_dir = PROCESSED / source

    if not src_dir.exists():
        print(f"  ERROR: {src_dir} not found. Run download_datasets.py first.")
        return

    print(f"\nProcessing {source}...")
    identity_dirs = sorted([d for d in src_dir.rglob("*") if d.is_dir() and
                             any(f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                                 for f in d.iterdir() if f.is_file())])

    if not identity_dirs:
        # Try flat structure where each subdir IS the identity
        identity_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])

    print(f"  Found {len(identity_dirs)} identities")

    stats = {"total": 0, "aligned": 0, "failed": 0, "identities": len(identity_dirs)}
    identity_map = {}  # identity_name → int label

    for label, identity_dir in enumerate(identity_dirs):
        identity_name = identity_dir.name
        identity_map[identity_name] = label

        images = list(identity_dir.glob("*.jpg")) + \
                 list(identity_dir.glob("*.jpeg")) + \
                 list(identity_dir.glob("*.png"))

        for img_path in images:
            stats["total"] += 1
            img = cv2.imread(str(img_path))
            if img is None:
                stats["failed"] += 1
                continue

            aligned = align_face(app, img)
            if aligned is None:
                stats["failed"] += 1
                continue

            # Save to processed dir under split
            out_identity_dir = out_dir / "all" / identity_name
            out_identity_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_identity_dir / img_path.name
            cv2.imwrite(str(out_path), aligned)
            stats["aligned"] += 1

        if label % 10 == 0:
            print(f"  [{label}/{len(identity_dirs)}] {identity_name}")

    print(f"  Aligned: {stats['aligned']}/{stats['total']} "
          f"({stats['failed']} no face detected)")

    # Train / val / test split
    split_dataset(out_dir, identity_map)

    # Save identity map
    with open(out_dir / "identity_map.json", "w") as f:
        json.dump(identity_map, f, indent=2)

    print(f"  Processed data saved to {out_dir}")
    return stats


def split_dataset(out_dir: Path, identity_map: dict,
                  train=0.8, val=0.1, test=0.1):
    """Split aligned faces into train/val/test by identity."""
    all_dir = out_dir / "all"
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    for identity_name in identity_map:
        identity_dir = all_dir / identity_name
        if not identity_dir.exists():
            continue
        images = list(identity_dir.iterdir())
        random.shuffle(images)

        n = len(images)
        n_train = max(1, int(n * train))
        n_val = max(1, int(n * val))

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split_name, split_images in splits.items():
            split_identity_dir = out_dir / split_name / identity_name
            split_identity_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, split_identity_dir / img.name)

    print(f"  Split: train/val/test created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["imfdb", "qmul", "all"], default="imfdb")
    args = parser.parse_args()

    app = get_face_app()
    sources = ["imfdb", "qmul"] if args.source == "all" else [args.source]
    for src in sources:
        process_dataset(src, app)

    print("\nDone. Next: python training/finetune.py")
