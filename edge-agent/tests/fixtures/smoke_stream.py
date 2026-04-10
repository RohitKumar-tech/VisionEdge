"""
Smoke test — runs StreamManager against the local test_feed.mp4.
Verifies the capture loop actually reads frames end-to-end.
Run: python tests/fixtures/smoke_stream.py
"""
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from camera.stream_manager import StreamManager

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test_feed.mp4")

cameras = [{"camera_id": "test-cam", "rtsp_url": VIDEO_PATH}]

sm = StreamManager(cameras=cameras)
sm.start()
time.sleep(0.3)  # let the capture thread read at least one frame

frame = sm.get_frame("test-cam")
if frame is not None:
    print(f"OK — got frame: shape={frame.shape}, dtype={frame.dtype}")
else:
    print("FAIL — no frame received")

sm.stop()
print("StreamManager stopped cleanly.")
