"""
Generates a synthetic 10-second test video (test_feed.mp4) for smoke testing
StreamManager without a real camera.
Each frame is colored and stamped with the frame number.
Run once: python tests/fixtures/make_test_video.py
"""
import cv2
import numpy as np
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "test_feed.mp4")
WIDTH, HEIGHT, FPS, DURATION = 640, 480, 15, 10

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

colors = [(200, 80, 80), (80, 200, 80), (80, 80, 200)]  # BGR
total = FPS * DURATION

for i in range(total):
    frame = np.full((HEIGHT, WIDTH, 3), colors[i % len(colors)], dtype=np.uint8)
    cv2.putText(frame, f"Frame {i}", (40, HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    writer.write(frame)

writer.release()
print(f"Created {OUTPUT}  ({total} frames @ {FPS}fps)")
