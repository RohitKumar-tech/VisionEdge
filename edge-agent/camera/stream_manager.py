"""
Camera stream manager — RTSP connection per camera.
Each camera runs in its own background thread storing the latest frame.
Designed to run on Jetson Nano / Mini PC at the client site.
"""
import cv2
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)


class StreamManager:
    """
    Manages RTSP sub-stream connections for all cameras at this site.

    Usage:
        sm = StreamManager(cameras=[
            {"camera_id": "cam-1", "rtsp_url": "rtsp://192.168.1.10:554/stream1"},
        ])
        sm.start()
        frame = sm.get_frame("cam-1")  # numpy array or None
        sm.stop()
    """

    def __init__(self, cameras: list):
        """
        cameras: list of dicts — required keys: camera_id, rtsp_url
        """
        self.cameras = cameras
        self._caps: dict[str, cv2.VideoCapture] = {}
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
            cap = cv2.VideoCapture(rtsp_url)
            self._caps[cam_id] = cap
            with self._lock:
                self._latest_frames[cam_id] = None
            t = threading.Thread(target=self._capture_loop, args=(cam_id,), daemon=True)
            self._threads.append(t)
            t.start()
            logger.info(f"Started capture thread for {cam_id} ({rtsp_url})")

    def _capture_loop(self, camera_id: str):
        """Continuously read frames from the camera. Runs in background thread."""
        cap = self._caps[camera_id]
        while self._running:
            if not cap.isOpened():
                logger.warning(f"{camera_id}: stream not open, skipping frame")
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frames[camera_id] = frame

    def get_frame(self, camera_id: str) -> np.ndarray | None:
        """
        Return the latest captured frame for camera_id.
        Returns None if camera is unknown or no frame has been captured yet.
        """
        with self._lock:
            return self._latest_frames.get(camera_id)

    def stop(self):
        """Stop all capture threads and release camera connections."""
        self._running = False
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._threads.clear()
        logger.info("StreamManager stopped, all captures released")
