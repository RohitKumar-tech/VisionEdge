"""
Camera stream manager — RTSP connection per camera.
Each camera runs in its own background thread storing the latest frame.
Designed to run on Jetson Nano / Mini PC at the client site.
"""
import re
import time
import cv2
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

_RECONNECT_DELAY = 5.0   # seconds to wait before reconnect attempt
_FRAME_SLEEP = 0.01      # seconds to sleep when no frame is ready (prevent CPU spin)


def _redact_rtsp_url(url: str) -> str:
    """
    Remove credentials from RTSP URL before logging.
    rtsp://user:pass@host → rtsp://***@host
    Handles passwords containing '@' by matching up to the last '@'.
    """
    return re.sub(r"(rtsp://).+@", r"\1***@", url)


class StreamManager:
    """
    Manages RTSP sub-stream connections for all cameras at this site.

    Each camera gets its own capture thread that continuously reads the latest
    frame. Disconnected cameras are automatically reconnected after a delay.

    Thread safety: get_frame() is safe to call from any thread.

    Usage:
        sm = StreamManager(cameras=[
            {"camera_id": "cam-1", "rtsp_url": "rtsp://192.168.1.10:554/stream1"},
        ])
        sm.start()
        frame = sm.get_frame("cam-1")  # returns a copy — safe to modify
        sm.stop()
    """

    def __init__(self, cameras: list):
        """
        cameras: list of dicts — required keys: camera_id, rtsp_url
        """
        self.cameras = cameras
        self._rtsp_urls: dict[str, str] = {}
        self._latest_frames: dict[str, np.ndarray | None] = {}
        self._lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []

    def start(self):
        """Connect to all cameras and start per-camera capture threads."""
        self._running = True
        for cam in self.cameras:
            cam_id = cam["camera_id"]
            rtsp_url = cam["rtsp_url"]
            self._rtsp_urls[cam_id] = rtsp_url
            with self._lock:
                self._latest_frames[cam_id] = None
            t = threading.Thread(target=self._capture_loop, args=(cam_id,), daemon=True)
            self._threads.append(t)
            t.start()
            logger.info(f"Started capture thread for {cam_id} ({_redact_rtsp_url(rtsp_url)})")

    def _capture_loop(self, camera_id: str):
        """
        Continuously read frames from the camera. Runs in background thread.
        Reconnects automatically if the stream drops.
        """
        rtsp_url = self._rtsp_urls[camera_id]
        cap = cv2.VideoCapture(rtsp_url)

        while self._running:
            if not cap.isOpened():
                logger.warning(f"{camera_id}: stream disconnected, reconnecting in {_RECONNECT_DELAY}s")
                cap.release()
                time.sleep(_RECONNECT_DELAY)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frames[camera_id] = frame
            else:
                time.sleep(_FRAME_SLEEP)  # brief pause — prevents CPU spin on decode failure

        cap.release()

    def get_frame(self, camera_id: str) -> np.ndarray | None:
        """
        Return a copy of the latest captured frame for camera_id.
        Returns None if camera is unknown or no frame has been captured yet.
        The returned array is a copy — safe to modify without affecting the stored frame.
        """
        with self._lock:
            frame = self._latest_frames.get(camera_id)
            return frame.copy() if frame is not None else None

    def stop(self):
        """Stop all capture threads and wait for them to finish cleanly."""
        self._running = False
        for t in self._threads:
            t.join(timeout=_RECONNECT_DELAY + 1.0)
        self._threads.clear()
        logger.info("StreamManager stopped")
