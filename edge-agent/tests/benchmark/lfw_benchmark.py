"""
LFW (Labeled Faces in the Wild) accuracy benchmark for FaceDetector.

Measures two things across degradation levels:
  1. Detection rate   — what % of faces does the model actually find
  2. Recognition accuracy — TAR@FAR=0.01 (standard metric: True Accept Rate at 1% False Accept Rate)
     This tells you: "when matching faces, how often is the right person accepted
     while wrong persons are rejected 99% of the time?"

LFW pairs dataset: 3000 same-person pairs + 3000 different-person pairs.
Downloaded automatically by scikit-learn (~200MB, cached in ~/scikit_learn_data/).

Run:
    python tests/benchmark/lfw_benchmark.py
    python tests/benchmark/lfw_benchmark.py --levels clean blur_moderate fog_moderate combined_typical
    python tests/benchmark/lfw_benchmark.py --tune   # also runs parameter tuning
"""
import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from degradation import DEGRADATION_LEVELS, degrade

DEFAULT_LEVELS = [
    "clean",
    "blur_mild", "blur_moderate", "blur_heavy", "blur_severe",
    "fog_mild", "fog_moderate", "fog_heavy",
    "noise_mild", "noise_moderate",
    "lowres_half", "lowres_quarter",
    "compression_heavy",
    "combined_mild", "combined_typical", "combined_severe",
]


def load_lfw_pairs():
    """
    Download and return LFW pairs as numpy arrays (RGB uint8).
    Returns:
        pairs     : ndarray (N, 2, H, W, 3) — N pairs, 2 images each
        is_same   : ndarray (N,) bool — True if same person
    """
    from sklearn.datasets import fetch_lfw_pairs
    print("Loading LFW pairs dataset (downloads ~200MB on first run)...")
    data = fetch_lfw_pairs(subset="test", color=True, resize=1.0)
    # sklearn stores as (N, 2, H, W, C) float [0,1] in RGB order
    pairs_float = data.pairs  # shape (N, 2, H, W, 3)
    pairs = (pairs_float * 255).astype(np.uint8)
    is_same = data.target.astype(bool)  # 1 = same person, 0 = different
    print(f"Loaded {len(pairs)} pairs ({is_same.sum()} same, {(~is_same).sum()} different)")
    return pairs, is_same


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def compute_tar_at_far(similarities_same, similarities_diff, target_far=0.01):
    """
    TAR@FAR: at the threshold where FAR=target_far, what is TAR?
    FAR = False Accept Rate (different person wrongly accepted)
    TAR = True Accept Rate (same person correctly accepted)
    Higher is better. >0.90 is good for CCTV use.
    """
    all_thresholds = np.linspace(-1, 1, 1000)
    best_tar = 0.0
    best_thresh = 0.0
    for thresh in all_thresholds:
        far = np.mean(similarities_diff >= thresh)
        if far <= target_far:
            tar = np.mean(similarities_same >= thresh)
            if tar > best_tar:
                best_tar = tar
                best_thresh = thresh
    return best_tar, best_thresh


def embed_face(app, image_rgb: np.ndarray):
    """Run InsightFace on image, return embedding of largest face or None."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(image_bgr)
    if not faces:
        return None
    # Pick the largest face by bbox area
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return largest.embedding.copy()


def run_benchmark(pairs, is_same, app, levels=None, max_pairs=1000):
    """
    For each degradation level, compute detection rate + TAR@FAR.
    max_pairs: limit pairs per level to keep runtime reasonable (use all 3000 for final report)
    """
    if levels is None:
        levels = DEFAULT_LEVELS

    n = min(max_pairs, len(pairs))
    pairs_subset = pairs[:n]
    labels_subset = is_same[:n]

    results = {}
    for level in levels:
        _, _, description = DEGRADATION_LEVELS.get(level, (None, None, level))
        sims_same, sims_diff = [], []
        detected, total = 0, 0

        for i, (pair, same) in enumerate(zip(pairs_subset, labels_subset)):
            img1 = degrade(pair[0], level)
            img2 = degrade(pair[1], level)

            emb1 = embed_face(app, img1)
            emb2 = embed_face(app, img2)
            total += 2
            if emb1 is not None: detected += 1
            if emb2 is not None: detected += 1

            if emb1 is not None and emb2 is not None:
                sim = cosine_similarity(emb1, emb2)
                if same:
                    sims_same.append(sim)
                else:
                    sims_diff.append(sim)

        detection_rate = detected / total if total > 0 else 0.0
        if sims_same and sims_diff:
            tar, thresh = compute_tar_at_far(
                np.array(sims_same), np.array(sims_diff), target_far=0.01
            )
        else:
            tar, thresh = 0.0, 0.0

        results[level] = {
            "description": description,
            "detection_rate": detection_rate,
            "tar_at_far_1pct": tar,
            "optimal_threshold": thresh,
            "pairs_evaluated": len(sims_same) + len(sims_diff),
        }
        print(f"  [{level:25s}]  detect={detection_rate:.1%}  TAR@FAR1%={tar:.1%}  thresh={thresh:.3f}  ({description})")

    return results


def tune_parameters(pairs, is_same, max_pairs=300):
    """
    Test different det_thresh and det_size values on 'combined_typical' degradation.
    Returns the best parameter combination.
    """
    import insightface

    configs = [
        {"det_thresh": 0.5, "det_size": (640, 640)},
        {"det_thresh": 0.3, "det_size": (640, 640)},
        {"det_thresh": 0.2, "det_size": (640, 640)},
        {"det_thresh": 0.3, "det_size": (320, 320)},
        {"det_thresh": 0.3, "det_size": (960, 960)},
    ]

    print("\n── Parameter Tuning on 'combined_typical' ──────────────────────────────")
    best_score, best_config = 0.0, configs[0]

    for cfg in configs:
        app = insightface.app.FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_thresh=cfg["det_thresh"], det_size=cfg["det_size"])

        n = min(max_pairs, len(pairs))
        sims_same, sims_diff = [], []
        detected, total = 0, 0

        for pair, same in zip(pairs[:n], is_same[:n]):
            img1 = degrade(pair[0], "combined_typical")
            img2 = degrade(pair[1], "combined_typical")
            emb1 = embed_face(app, img1)
            emb2 = embed_face(app, img2)
            total += 2
            if emb1: detected += 1
            if emb2: detected += 1
            if emb1 is not None and emb2 is not None:
                sim = cosine_similarity(emb1, emb2)
                (sims_same if same else sims_diff).append(sim)

        det_rate = detected / total
        tar, _ = compute_tar_at_far(np.array(sims_same or [0]),
                                     np.array(sims_diff or [0]))
        # Combined score: weight detection rate + TAR equally
        score = 0.4 * det_rate + 0.6 * tar
        print(f"  det_thresh={cfg['det_thresh']}  det_size={cfg['det_size']}  → detect={det_rate:.1%}  TAR@FAR1%={tar:.1%}  score={score:.3f}")

        if score > best_score:
            best_score = score
            best_config = cfg

    print(f"\n  Best config: {best_config}  (score={best_score:.3f})")
    return best_config


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Degradation':<28} {'Detect':>8} {'TAR@FAR1%':>10} {'Threshold':>10}")
    print("-" * 70)
    for level, r in results.items():
        flag = ""
        if r["tar_at_far_1pct"] >= 0.90:
            flag = " ✓"
        elif r["tar_at_far_1pct"] >= 0.75:
            flag = " ~"
        else:
            flag = " ✗"
        print(f"  {level:<26} {r['detection_rate']:>7.1%}  {r['tar_at_far_1pct']:>9.1%}{flag}  {r['optimal_threshold']:>9.3f}")
    print("-" * 70)
    print("  ✓ >= 90%  ~  >= 75%  ✗ < 75%")
    print()

    # Recommendations
    bad = [k for k, v in results.items() if v["tar_at_far_1pct"] < 0.75 and k != "clean"]
    if bad:
        print(f"  Low accuracy conditions: {', '.join(bad)}")
        print("  → Consider lowering det_thresh, increasing det_size, or pre-processing frames")
    else:
        print("  All tested conditions meet 75%+ accuracy threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFW accuracy benchmark for FaceDetector")
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS,
                        choices=list(DEGRADATION_LEVELS.keys()),
                        help="Degradation levels to test")
    parser.add_argument("--max-pairs", type=int, default=500,
                        help="Max pairs per level (use 3000 for full benchmark, ~30min)")
    parser.add_argument("--tune", action="store_true",
                        help="Also run parameter tuning (finds best det_thresh + det_size)")
    args = parser.parse_args()

    import insightface
    pairs, is_same = load_lfw_pairs()

    # Optional: find best parameters first
    if args.tune:
        best_cfg = tune_parameters(pairs, is_same)
        det_thresh = best_cfg["det_thresh"]
        det_size = best_cfg["det_size"]
    else:
        det_thresh = 0.3   # lower than default (0.5) — better for degraded footage
        det_size = (640, 640)

    print(f"\n── Running benchmark  det_thresh={det_thresh}  det_size={det_size}  max_pairs={args.max_pairs} ──")
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    results = run_benchmark(pairs, is_same, app, levels=args.levels, max_pairs=args.max_pairs)
    print_summary(results)
