"""
Evaluates fine-tuned InsightFace model vs original buffalo_l.

Metrics:
  - Verification accuracy (TAR @ FAR=1%) on Indian face pairs
  - Detection rate under CCTV degradations (blur, fog, noise, low-res, compression)
  - Identification top-1 accuracy on test split

Run: python training/evaluate.py --checkpoint checkpoints/archead_imfdb_best.pt
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE = Path(__file__).parent
PROCESSED = BASE / "data" / "processed"
CHECKPOINTS = BASE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_SIZE = 112


# ── Inline imports from finetune (avoid circular import) ─────────────────────

def _load_finetune_classes():
    import sys
    sys.path.insert(0, str(BASE))
    from finetune import ArcFaceLoss, InsightFaceExtractor, FaceDataset
    return ArcFaceLoss, InsightFaceExtractor, FaceDataset


# ── CCTV degradation functions ────────────────────────────────────────────────

DEGRADATIONS = {
    "clean": lambda img: img,
    "blur_mild":   lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "blur_heavy":  lambda img: cv2.GaussianBlur(img, (15, 15), 0),
    "fog_light":   lambda img: cv2.addWeighted(img, 0.8, np.full_like(img, 255), 0.2, 0),
    "fog_heavy":   lambda img: cv2.addWeighted(img, 0.5, np.full_like(img, 255), 0.5, 0),
    "noise_low":   lambda img: _add_noise(img, sigma=10),
    "noise_high":  lambda img: _add_noise(img, sigma=40),
    "lowres_half": lambda img: _downscale(img, 0.5),
    "lowres_qtr":  lambda img: _downscale(img, 0.25),
    "jpeg_q30":    lambda img: _jpeg_compress(img, 30),
    "jpeg_q10":    lambda img: _jpeg_compress(img, 10),
    "combined":    lambda img: _combined(img),
}


def _add_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _downscale(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.resize(small, (w, h))


def _jpeg_compress(img, quality):
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _combined(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = _add_noise(img, sigma=20)
    img = cv2.addWeighted(img, 0.75, np.full_like(img, 255), 0.25, 0)
    return img


# ── Embedding extraction ──────────────────────────────────────────────────────

def img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR image (HxWx3 uint8) → normalised tensor (3x112x112)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5


class BaselineExtractor:
    """Original buffalo_l without any fine-tuning."""
    def __init__(self):
        import insightface
        app = insightface.app.FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_thresh=0.3)
        self._rec = app.models.get("recognition")

    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray | None:
        try:
            emb = self._rec.get_feat(img_bgr)
            return emb.flatten() / (np.linalg.norm(emb) + 1e-8)
        except Exception:
            return None


class FinetunedExtractor:
    """buffalo_l backbone + fine-tuned ArcFace head embeddings."""
    def __init__(self, checkpoint_path: Path):
        ArcFaceLoss, InsightFaceExtractor, _ = _load_finetune_classes()
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

        self._backbone = InsightFaceExtractor().to(DEVICE)
        self._num_classes = ckpt["num_classes"]
        self._arc_head = ArcFaceLoss(512, self._num_classes).to(DEVICE)
        self._arc_head.load_state_dict(ckpt["arc_head_state"])
        self._arc_head.eval()
        self._val_acc = ckpt.get("val_acc", 0.0)
        self._class_to_idx = ckpt.get("class_to_idx", {})
        print(f"  Loaded checkpoint: {checkpoint_path.name} "
              f"(val_acc={self._val_acc:.1%}, {self._num_classes} classes)")

    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray | None:
        try:
            tensor = img_to_tensor(img_bgr).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = self._backbone(tensor)
                emb = nn.functional.normalize(emb)
            return emb.cpu().numpy().flatten()
        except Exception:
            return None


# ── Pair generation from test split ──────────────────────────────────────────

def build_pairs(test_dir: Path, max_pairs: int = 2000):
    """
    Generate positive (same identity) and negative (different identity) pairs.
    Returns list of (img_path_a, img_path_b, is_same_person: bool)
    """
    identity_images: dict[str, list[Path]] = {}
    for identity_dir in sorted(test_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        imgs = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        if len(imgs) >= 2:
            identity_images[identity_dir.name] = imgs

    pairs = []
    identities = list(identity_images.keys())
    rng = np.random.default_rng(42)

    # Positive pairs
    for name, imgs in identity_images.items():
        idxs = rng.choice(len(imgs), size=min(4, len(imgs)), replace=False)
        for i, j in itertools.combinations(idxs, 2):
            pairs.append((imgs[i], imgs[j], True))
            if len(pairs) >= max_pairs // 2:
                break
        if len(pairs) >= max_pairs // 2:
            break

    # Negative pairs
    neg_count = 0
    target_neg = max_pairs // 2
    while neg_count < target_neg and len(identities) >= 2:
        i, j = rng.choice(len(identities), size=2, replace=False)
        name_a, name_b = identities[i], identities[j]
        img_a = identity_images[name_a][rng.integers(len(identity_images[name_a]))]
        img_b = identity_images[name_b][rng.integers(len(identity_images[name_b]))]
        pairs.append((img_a, img_b, False))
        neg_count += 1

    rng.shuffle(pairs)
    print(f"  Pairs: {sum(1 for _,_,s in pairs if s)} positive, "
          f"{sum(1 for _,_,s in pairs if not s)} negative")
    return pairs


# ── Verification benchmark ────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compute_tar_at_far(sims, labels, target_far=0.01):
    """
    Compute True Accept Rate at False Accept Rate ≤ target_far.
    sims: list of cosine similarities
    labels: list of bool (True = same person)
    """
    sims = np.array(sims)
    labels = np.array(labels)

    thresholds = np.linspace(sims.min(), sims.max(), 1000)
    best_tar = 0.0
    best_thresh = 0.0

    for thresh in thresholds:
        preds = sims >= thresh
        tp = np.sum(preds & labels)
        fp = np.sum(preds & ~labels)
        fn = np.sum(~preds & labels)
        tn = np.sum(~preds & ~labels)

        tar = tp / (tp + fn + 1e-8)
        far = fp / (fp + tn + 1e-8)

        if far <= target_far and tar > best_tar:
            best_tar = tar
            best_thresh = thresh

    return best_tar, best_thresh


def run_verification(extractor, pairs, degradation_fn, label: str):
    """Run pair verification under a specific degradation."""
    sims, labels = [], []
    fail_count = 0

    for img_path_a, img_path_b, is_same in pairs:
        img_a = cv2.imread(str(img_path_a))
        img_b = cv2.imread(str(img_path_b))
        if img_a is None or img_b is None:
            fail_count += 1
            continue

        img_a = degradation_fn(img_a)
        img_b = degradation_fn(img_b)

        emb_a = extractor.get_embedding(img_a)
        emb_b = extractor.get_embedding(img_b)

        if emb_a is None or emb_b is None:
            fail_count += 1
            continue

        sims.append(cosine_sim(emb_a, emb_b))
        labels.append(is_same)

    if len(sims) < 10:
        return {"tar@far1%": 0.0, "threshold": 0.0, "n_pairs": 0, "fail": fail_count}

    tar, thresh = compute_tar_at_far(sims, labels, target_far=0.01)
    return {
        "tar@far1%": round(tar, 4),
        "threshold": round(thresh, 4),
        "n_pairs": len(sims),
        "fail": fail_count,
    }


# ── Identification benchmark ──────────────────────────────────────────────────

def run_identification(extractor, test_dir: Path, max_samples: int = 500):
    """Top-1 gallery identification accuracy on test split."""
    _, _, FaceDataset = _load_finetune_classes()

    # Build gallery: one embedding per identity (from first image)
    gallery_embeddings = {}
    gallery_labels = {}

    for identity_dir in sorted(test_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        imgs = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        if not imgs:
            continue
        img = cv2.imread(str(imgs[0]))
        if img is None:
            continue
        emb = extractor.get_embedding(img)
        if emb is not None:
            gallery_embeddings[identity_dir.name] = emb

    # Probe: remaining images
    correct, total = 0, 0
    for identity_dir in sorted(test_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        if identity_dir.name not in gallery_embeddings:
            continue
        imgs = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        for img_path in imgs[1:]:  # skip first (used as gallery)
            if total >= max_samples:
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = extractor.get_embedding(img)
            if emb is None:
                continue

            # Find nearest gallery embedding
            best_sim = -1.0
            best_name = None
            for name, g_emb in gallery_embeddings.items():
                s = cosine_sim(emb, g_emb)
                if s > best_sim:
                    best_sim = s
                    best_name = name

            if best_name == identity_dir.name:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    return {"top1_acc": round(acc, 4), "correct": correct, "total": total}


# ── Main benchmark ────────────────────────────────────────────────────────────

def benchmark(checkpoint_path: Path | None, data_source: str = "imfdb"):
    test_dir = PROCESSED / data_source / "test"
    if not test_dir.exists():
        print(f"ERROR: {test_dir} not found. Run prepare_data.py first.")
        return

    print(f"\n── Building evaluation pairs ───────────────────────────────────────────")
    pairs = build_pairs(test_dir)

    print(f"\n── Loading extractors ──────────────────────────────────────────────────")
    baseline = BaselineExtractor()
    finetuned = FinetunedExtractor(checkpoint_path) if checkpoint_path else None

    results = {"baseline": {}, "finetuned": {}}

    print(f"\n── Verification benchmark (TAR@FAR=1%) ────────────────────────────────")
    print(f"  {'Degradation':<20} {'Baseline TAR':>13} {'Finetuned TAR':>14}")
    print(f"  {'-'*20} {'-'*13} {'-'*14}")

    for deg_name, deg_fn in DEGRADATIONS.items():
        b_res = run_verification(baseline, pairs, deg_fn, deg_name)
        results["baseline"][deg_name] = b_res

        if finetuned:
            f_res = run_verification(finetuned, pairs, deg_fn, deg_name)
            results["finetuned"][deg_name] = f_res
            delta = f_res["tar@far1%"] - b_res["tar@far1%"]
            delta_str = f"({delta:+.1%})"
            print(f"  {deg_name:<20} {b_res['tar@far1%']:>12.1%}  "
                  f"{f_res['tar@far1%']:>12.1%} {delta_str}")
        else:
            print(f"  {deg_name:<20} {b_res['tar@far1%']:>12.1%}")

    print(f"\n── Identification (Top-1) ──────────────────────────────────────────────")
    b_id = run_identification(baseline, test_dir)
    results["baseline"]["identification"] = b_id
    print(f"  Baseline:  {b_id['top1_acc']:.1%} ({b_id['correct']}/{b_id['total']})")

    if finetuned:
        f_id = run_identification(finetuned, test_dir)
        results["finetuned"]["identification"] = f_id
        delta = f_id["top1_acc"] - b_id["top1_acc"]
        print(f"  Finetuned: {f_id['top1_acc']:.1%} ({f_id['correct']}/{f_id['total']}) "
              f"({delta:+.1%})")

    # Save results
    out_path = CHECKPOINTS / f"eval_{data_source}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results → {out_path}")

    # Summary
    print(f"\n── Summary ─────────────────────────────────────────────────────────────")
    clean_b = results["baseline"].get("clean", {}).get("tar@far1%", 0)
    comb_b = results["baseline"].get("combined", {}).get("tar@far1%", 0)
    print(f"  Baseline  — clean: {clean_b:.1%}  combined: {comb_b:.1%}")
    if finetuned:
        clean_f = results["finetuned"].get("clean", {}).get("tar@far1%", 0)
        comb_f = results["finetuned"].get("combined", {}).get("tar@far1%", 0)
        print(f"  Finetuned — clean: {clean_f:.1%}  combined: {comb_f:.1%}")
        if clean_f >= clean_b:
            print(f"  ✓ Fine-tuning improved CCTV robustness")
        else:
            print(f"  ✗ Fine-tuning did not improve — consider more epochs or unfreezing layers")

    print(f"\n  Next: python training/export_model.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to fine-tuned checkpoint (omit to run baseline only)")
    parser.add_argument("--data", default="imfdb", choices=["imfdb", "qmul"])
    args = parser.parse_args()

    if args.checkpoint and not args.checkpoint.exists():
        # Try relative to CHECKPOINTS dir
        args.checkpoint = CHECKPOINTS / args.checkpoint

    benchmark(args.checkpoint, args.data)
