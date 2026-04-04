"""
Camera stream manager — ONVIF primary, RTSP fallback.
Maintains persistent connections, auto-reconnects on disconnect.
"""
# TODO Phase 1: Implement


class StreamManager:
    """Manages RTSP sub-stream connections for all cameras at this site."""

    def __init__(self, cameras: list, config):
        self.cameras = cameras
        self.config = config
        self._streams = {}  # camera_id → cv2.VideoCapture

    def start(self):
        """Connect to all cameras and begin frame capture."""
        raise NotImplementedError

    def get_frame(self, camera_id: str):
        """Return latest frame for a camera_id, or None if unavailable."""
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
