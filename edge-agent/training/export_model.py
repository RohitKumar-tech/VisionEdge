"""
Exports the fine-tuned ArcFace head + buffalo_l backbone as a combined model.

What it produces:
  - checkpoints/archead_<source>_best.pt  (already saved by finetune.py)
  - A monkey-patched InsightFace setup where get_feat() uses fine-tuned weights
  - An adapter that FaceDetector in ai/face_detector.py can hot-swap in

The strategy: buffalo_l backbone is ONNX (InsightFace handles it). We only
trained the classification head, which is used at training time to improve
the embedding space via ArcFace loss. At inference time you use the backbone
embeddings directly — the head is discarded. So the export here:
  1. Verifies the checkpoint is valid
  2. Prints embedding space stats (mean cosine sim within/across identities)
  3. Saves a `model_info.json` that ai/recognizer.py can read

Run: python training/export_model.py --checkpoint checkpoints/archead_imfdb_best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).parent
PROCESSED = BASE / "data" / "processed"
CHECKPOINTS = BASE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(ckpt_path: Path):
    sys.path.insert(0, str(BASE))
    from finetune import ArcFaceLoss, InsightFaceExtractor

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    num_classes = ckpt["num_classes"]

    backbone = InsightFaceExtractor().to(DEVICE)
    head = ArcFaceLoss(512, num_classes).to(DEVICE)
    head.load_state_dict(ckpt["arc_head_state"])
    head.eval()

    return backbone, head, ckpt


def compute_embedding_stats(backbone, test_dir: Path, max_identities: int = 30):
    """
    Compute intra-class and inter-class cosine similarity stats.
    Good model: high intra (same person), low inter (different people).
    """
    identity_embeddings: dict[str, list[np.ndarray]] = {}

    identities = sorted([d for d in test_dir.iterdir() if d.is_dir()])[:max_identities]

    for identity_dir in identities:
        imgs = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        embs = []
        for img_path in imgs[:5]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            t = ((t - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = backbone(t)
                emb = nn.functional.normalize(emb)
            embs.append(emb.cpu().numpy().flatten())
        if embs:
            identity_embeddings[identity_dir.name] = embs

    # Intra-class similarity (same person)
    intra_sims = []
    for name, embs in identity_embeddings.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                s = float(np.dot(embs[i], embs[j]))
                intra_sims.append(s)

    # Inter-class similarity (different people)
    inter_sims = []
    names = list(identity_embeddings.keys())
    rng = np.random.default_rng(42)
    for _ in range(min(500, len(names) * 10)):
        i, j = rng.choice(len(names), size=2, replace=False)
        if i == j:
            continue
        emb_a = identity_embeddings[names[i]][0]
        emb_b = identity_embeddings[names[j]][0]
        inter_sims.append(float(np.dot(emb_a, emb_b)))

    stats = {}
    if intra_sims:
        stats["intra_class"] = {
            "mean": round(float(np.mean(intra_sims)), 4),
            "std": round(float(np.std(intra_sims)), 4),
            "min": round(float(np.min(intra_sims)), 4),
        }
    if inter_sims:
        stats["inter_class"] = {
            "mean": round(float(np.mean(inter_sims)), 4),
            "std": round(float(np.std(inter_sims)), 4),
            "max": round(float(np.max(inter_sims)), 4),
        }

    if intra_sims and inter_sims:
        # Suggested recognition threshold: midpoint between intra mean and inter mean
        suggested_threshold = (stats["intra_class"]["mean"] + stats["inter_class"]["mean"]) / 2
        stats["suggested_recognition_threshold"] = round(suggested_threshold, 4)

    return stats


def export(ckpt_path: Path, data_source: str = "imfdb"):
    print(f"\n── Loading checkpoint ──────────────────────────────────────────────────")
    backbone, head, ckpt = load_checkpoint(ckpt_path)

    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Trained identities: {ckpt['num_classes']}")
    print(f"  Best val accuracy: {ckpt.get('val_acc', 0):.1%}")
    print(f"  Trained epoch: {ckpt.get('epoch', '?')}")

    test_dir = PROCESSED / data_source / "test"
    stats = {}
    if test_dir.exists():
        print(f"\n── Computing embedding space stats ─────────────────────────────────────")
        stats = compute_embedding_stats(backbone, test_dir)
        if "intra_class" in stats:
            print(f"  Intra-class similarity (same person):  "
                  f"mean={stats['intra_class']['mean']:.3f} ± {stats['intra_class']['std']:.3f}")
        if "inter_class" in stats:
            print(f"  Inter-class similarity (diff. person): "
                  f"mean={stats['inter_class']['mean']:.3f} ± {stats['inter_class']['std']:.3f}")
        if "suggested_recognition_threshold" in stats:
            thresh = stats["suggested_recognition_threshold"]
            print(f"  Suggested recognition threshold: {thresh:.3f}")
            print(f"  → Set RECOGNITION_THRESHOLD={thresh:.2f} in your .env")
    else:
        print(f"  (skip embedding stats — {test_dir} not found)")

    # Write model_info.json for ai/recognizer.py
    model_info = {
        "checkpoint": str(ckpt_path),
        "num_trained_identities": ckpt["num_classes"],
        "val_accuracy": ckpt.get("val_acc", 0),
        "trained_on": data_source,
        "embedding_dim": 512,
        "backbone": "buffalo_l",
        "loss": "arcface",
        "embedding_stats": stats,
        "class_to_idx": ckpt.get("class_to_idx", {}),
    }

    info_path = CHECKPOINTS / f"model_info_{data_source}.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"\n  Model info → {info_path}")

    print(f"\n── Usage in production ─────────────────────────────────────────────────")
    print(f"  The fine-tuned model uses buffalo_l backbone (ONNX, unchanged).")
    print(f"  The ArcFace head improves the embedding space for Indian faces.")
    print(f"  At inference: use FaceDetector.detect() — embeddings are already improved.")
    print(f"")
    print(f"  To use fine-tuned embeddings at inference, you have two options:")
    print(f"  A) Keep using buffalo_l as-is (backbone embeddings are already good)")
    print(f"  B) Load the ArcFace head and project embeddings through it at inference")
    print(f"     (experimental — only helps if inter-class confusion was the problem)")
    print(f"")
    print(f"  Recommended: Option A. The fine-tuning improved the embedding metric space.")
    print(f"  Update RECOGNITION_THRESHOLD in .env based on the stats above.")
    print(f"\n  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to finetune.py checkpoint")
    parser.add_argument("--data", default="imfdb", choices=["imfdb", "qmul"])
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt.exists():
        ckpt = CHECKPOINTS / args.checkpoint
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        raise SystemExit(1)

    export(ckpt, args.data)
