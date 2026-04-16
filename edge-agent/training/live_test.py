"""
Live webcam test — runs InsightFace detection + identity matching on laptop camera.

What it does:
  - Opens laptop webcam (device 0)
  - Detects faces every frame using buffalo_l
  - Matches each face against the 100 trained IMFDB identities
  - Shows bounding box + top-3 matches with confidence scores

Controls:
  Q       — quit
  E       — enroll current face as "Me" (adds your embedding to match list)
  S       — save snapshot
  C       — clear enrolled faces

Run:
  python training/live_test.py
  python training/live_test.py --checkpoint checkpoints/archead_imfdb_best.pt
  python training/live_test.py --camera 1   (if 0 doesn't work)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE.parent))  # edge-agent root on path


# ── Load InsightFace ──────────────────────────────────────────────────────────

def load_insightface():
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
    rec_model = app.models.get("recognition")
    return app, rec_model


# ── Load trained gallery (100 IMFDB identities) ───────────────────────────────

def load_gallery(checkpoint_path: Path | None, data_source: str = "imfdb"):
    """
    Build a gallery of embeddings from the test split.
    Returns {name: np.ndarray (normalised embedding)}
    """
    test_dir = BASE / "data" / "processed" / data_source / "test"
    if not test_dir.exists():
        print(f"  [warn] No test data at {test_dir} — gallery will be empty")
        return {}

    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_thresh=0.3)
    rec = app.models.get("recognition")

    gallery = {}
    identities = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    print(f"  Loading gallery from {len(identities)} identities...")

    # Use ALL splits (train+val+test) for richest gallery
    all_splits_dir = BASE / "data" / "processed" / data_source

    for identity_dir in identities:
        embs = []
        # Pull images from all splits for this identity
        for split in ["train", "val", "test"]:
            split_id_dir = all_splits_dir / split / identity_dir.name
            if not split_id_dir.exists():
                continue
            imgs = list(split_id_dir.glob("*.jpg")) + list(split_id_dir.glob("*.png"))
            for img_path in imgs[:10]:  # up to 10 per split = up to 30 total
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                try:
                    emb = rec.get_feat(img).flatten()
                    embs.append(emb)
                except Exception:
                    pass
        if embs:
            avg = np.mean(embs, axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            gallery[identity_dir.name] = avg

    print(f"  Gallery: {len(gallery)} identities loaded")
    return gallery


# ── Drawing helpers ────────────────────────────────────────────────────────────

COLORS = {
    "match":   (0, 255, 100),   # green
    "low":     (0, 200, 255),   # yellow
    "unknown": (60, 60, 255),   # red
    "enrolled":(255, 200, 0),   # cyan
}

def draw_face(frame, bbox, label_lines, color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Draw label background
    line_h = 20
    bg_h = line_h * len(label_lines) + 8
    cv2.rectangle(frame, (x1, y1 - bg_h), (x2, y1), color, -1)
    for i, line in enumerate(label_lines):
        cv2.putText(
            frame, line,
            (x1 + 4, y1 - bg_h + line_h * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA,
        )


def draw_hud(frame, fps, n_faces, enrolled_names):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
    cv2.putText(frame, f"VisionEdge Live Test  |  FPS: {fps:.1f}  |  Faces: {n_faces}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 180), 1)
    # Enrolled list
    if enrolled_names:
        cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"Enrolled: {', '.join(enrolled_names)}",
                    (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 1)
    # Controls
    cv2.putText(frame, "Q=quit  E=enroll  S=save  C=clear",
                (w - 250, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)


# ── CLAHE enhancement (same as production face_detector.py) ───────────────────

def enhance_frame(frame, clahe):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(camera_id: int = 0, checkpoint: Path | None = None,
        gallery_source: str = "imfdb", match_threshold: float = 0.45):

    print("\n── VisionEdge Live Test ────────────────────────────────────────────────")
    print(f"  Camera: {camera_id}")
    print(f"  Match threshold: {match_threshold}")

    print("\n── Loading InsightFace buffalo_l ───────────────────────────────────────")
    app, rec_model = load_insightface()

    print("\n── Loading identity gallery ────────────────────────────────────────────")
    gallery = load_gallery(checkpoint, gallery_source)

    enrolled: dict[str, np.ndarray] = {}  # user-enrolled faces (name → embedding)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enroll_counter = 0

    # Multi-frame embedding buffer: track_id → deque of recent embeddings
    from collections import deque
    emb_buffer: dict[int, deque] = {}   # face index → rolling buffer of embeddings
    BUFFER_SIZE = 7  # average last 7 frames before matching

    # Temporal smoothing: lock identity for N frames once confident
    identity_lock: dict[int, dict] = {}   # face index → {name, sim, ttl}
    LOCK_TTL = 15     # frames to hold identity once matched
    LOCK_MIN_SIM = match_threshold + 0.08  # confidence needed to lock

    print("\n── Opening camera ──────────────────────────────────────────────────────")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_id}")
        print("Try: python live_test.py --camera 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("  Camera opened. Press Q to quit.")

    fps_t = time.time()
    fps_count = 0
    fps = 0.0
    snapshot_n = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera read failed — retrying...")
            time.sleep(0.1)
            continue

        # FPS
        fps_count += 1
        if fps_count >= 10:
            fps = fps_count / (time.time() - fps_t)
            fps_count = 0
            fps_t = time.time()

        # Enhance + detect
        enhanced = enhance_frame(frame, clahe)
        try:
            faces = app.get(enhanced)
        except Exception as e:
            faces = []

        display = frame.copy()

        # Clean up buffers for faces no longer visible
        active_ids = set(range(len(faces)))
        for fid in list(emb_buffer.keys()):
            if fid not in active_ids:
                del emb_buffer[fid]
        for fid in list(identity_lock.keys()):
            if fid not in active_ids:
                del identity_lock[fid]

        for fid, face in enumerate(faces):
            bbox = face.bbox
            det_score = float(face.det_score)

            # Skip very low quality detections
            if det_score < 0.5:
                continue

            try:
                raw_emb = rec_model.get_feat(
                    enhanced[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                ).flatten()
            except Exception:
                raw_emb = face.embedding.flatten() if face.embedding is not None else None

            if raw_emb is None:
                continue

            emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-8)

            # ── Multi-frame averaging ──────────────────────────────────────
            if fid not in emb_buffer:
                emb_buffer[fid] = deque(maxlen=BUFFER_SIZE)
            emb_buffer[fid].append(emb)

            # Average all buffered embeddings — smoother than single frame
            avg_emb = np.mean(np.stack(emb_buffer[fid]), axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

            # ── Temporal lock: skip matching if recently confident ─────────
            _locked = False
            if fid in identity_lock:
                lock = identity_lock[fid]
                lock["ttl"] -= 1
                if lock["ttl"] > 0:
                    best_sim  = lock["sim"]
                    best_name = lock["name"]
                    top3 = [(best_sim, best_name)]
                    _locked = True
                else:
                    del identity_lock[fid]

            if not _locked:
                # ── Match against enrolled + gallery ──────────────────────
                all_candidates = {**{f"[me] {k}": v for k, v in enrolled.items()}, **gallery}
                top3 = sorted(
                    [(float(np.dot(avg_emb, g_emb)), name)
                     for name, g_emb in all_candidates.items()],
                    reverse=True,
                )[:3]
                best_sim, best_name = (top3[0][0], top3[0][1]) if top3 else (-1.0, "?")

                # Lock identity for next N frames if very confident
                if best_sim >= LOCK_MIN_SIM:
                    identity_lock[fid] = {"name": best_name, "sim": best_sim, "ttl": LOCK_TTL}

            # ── Color + label ──────────────────────────────────────────────
            buf_len = len(emb_buffer[fid])
            buf_tag = f" [{buf_len}/{BUFFER_SIZE}f]" if buf_len < BUFFER_SIZE else ""

            if best_sim >= match_threshold:
                color = COLORS["enrolled"] if best_name.startswith("[me]") else COLORS["match"]
                display_name = best_name.replace("[me] ", "")
                label_lines = [f"{display_name}  {best_sim:.2f}{buf_tag}"]
                if not _locked and len(top3) > 1:
                    label_lines.append(f"#2: {top3[1][1]}  {top3[1][0]:.2f}")
            elif best_sim >= match_threshold * 0.7:
                color = COLORS["low"]
                label_lines = [f"? {best_name}  {best_sim:.2f}{buf_tag}"]
            else:
                color = COLORS["unknown"]
                label_lines = [
                    f"Unknown (best: {best_name}){buf_tag}",
                    f"sim={best_sim:.2f}  need>={match_threshold:.2f}",
                ]

            draw_face(display, bbox, label_lines, color)

            # Landmark dots
            if hasattr(face, "kps") and face.kps is not None:
                for pt in face.kps:
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        draw_hud(display, fps, len(faces), list(enrolled.keys()))
        cv2.imshow("VisionEdge Live Test", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("e"):
            # Multi-frame enrollment: capture 15 frames and average embeddings
            if faces:
                print("  Enrolling — hold still for 15 frames...")
                enroll_embs = []
                for _ in range(15):
                    ret2, frame2 = cap.read()
                    if not ret2:
                        continue
                    enh2 = enhance_frame(frame2, clahe)
                    try:
                        faces2 = app.get(enh2)
                        if faces2:
                            lf = max(faces2, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                            raw2 = rec_model.get_feat(
                                enh2[int(lf.bbox[1]):int(lf.bbox[3]),
                                     int(lf.bbox[0]):int(lf.bbox[2])]
                            ).flatten()
                            enroll_embs.append(raw2 / (np.linalg.norm(raw2) + 1e-8))
                    except Exception:
                        pass
                if enroll_embs:
                    avg = np.mean(enroll_embs, axis=0)
                    avg = avg / (np.linalg.norm(avg) + 1e-8)
                    enroll_counter += 1
                    name = f"Person{enroll_counter}"
                    enrolled[name] = avg
                    print(f"  Enrolled '{name}' from {len(enroll_embs)} frames")
                else:
                    print("  Enrollment failed — no face detected")
            else:
                print("  No face detected to enroll")

        elif key == ord("s"):
            snapshot_n += 1
            path = BASE / f"snapshot_{snapshot_n:03d}.jpg"
            cv2.imwrite(str(path), display)
            print(f"  Saved snapshot: {path}")

        elif key == ord("c"):
            enrolled.clear()
            enroll_counter = 0
            print("  Cleared all enrolled faces")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0 = laptop webcam)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to fine-tuned checkpoint (optional)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Cosine similarity match threshold (default: 0.45)")
    parser.add_argument("--data", default="imfdb", choices=["imfdb", "qmul"])
    args = parser.parse_args()

    run(
        camera_id=args.camera,
        checkpoint=args.checkpoint,
        gallery_source=args.data,
        match_threshold=args.threshold,
    )
