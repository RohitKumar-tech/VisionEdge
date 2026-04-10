"""
Manual smoke test for FaceDetector — downloads buffalo_l model on first run (~500MB).
Detects faces in a test image (or webcam frame if no image provided).

Run: python tests/fixtures/smoke_face_detector.py [path/to/image.jpg]

When you get a real camera, replace the image path with a frame captured via StreamManager.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ai.face_detector import FaceDetector

detector = FaceDetector()
print("Loading InsightFace buffalo_l model (downloads ~500MB on first run)...")
detector.load()
print("Model loaded.")

if len(sys.argv) > 1:
    import cv2
    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)
else:
    # Generate a blank frame — will return 0 faces, but confirms model runs without error
    print("No image provided — using blank frame (expect 0 faces).")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

faces = detector.detect(frame)
print(f"\nDetected {len(faces)} face(s):")
for i, f in enumerate(faces):
    print(f"  [{i+1}] score={f['det_score']:.3f}  bbox={f['bbox'].astype(int).tolist()}  embedding={f['embedding'].shape}")

if len(faces) > 0:
    print("\nOK — InsightFace is working correctly on this device.")
else:
    print("\nOK — model ran without errors (pass a photo with a face to verify detection).")
