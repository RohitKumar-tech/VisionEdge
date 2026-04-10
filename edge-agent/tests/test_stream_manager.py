import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from camera.stream_manager import StreamManager

FAKE_CAMERAS = [
    {"camera_id": "cam-1", "rtsp_url": "rtsp://192.168.1.10:554/stream1"},
    {"camera_id": "cam-2", "rtsp_url": "rtsp://192.168.1.11:554/stream1"},
]


def make_mock_cap(width=640, height=480, readable=True):
    cap = MagicMock()
    cap.isOpened.return_value = True
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cap.read.return_value = (readable, frame if readable else None)
    return cap


def test_start_connects_to_all_cameras():
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        mock_vc.side_effect = [make_mock_cap(), make_mock_cap()]
        sm = StreamManager(cameras=FAKE_CAMERAS)
        sm.start()
        time.sleep(0.1)  # let capture threads run
        assert mock_vc.call_count == 2
        sm.stop()


def test_get_frame_returns_numpy_array():
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = make_mock_cap()
        sm = StreamManager(cameras=FAKE_CAMERAS[:1])
        sm.start()
        time.sleep(0.1)
        frame = sm.get_frame("cam-1")
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        sm.stop()


def test_get_frame_returns_none_for_unknown_camera():
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = make_mock_cap()
        sm = StreamManager(cameras=FAKE_CAMERAS[:1])
        sm.start()
        time.sleep(0.1)
        assert sm.get_frame("cam-does-not-exist") is None
        sm.stop()


def test_failed_read_returns_none():
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = make_mock_cap(readable=False)
        sm = StreamManager(cameras=FAKE_CAMERAS[:1])
        sm.start()
        time.sleep(0.1)
        assert sm.get_frame("cam-1") is None
        sm.stop()


def test_stop_releases_all_captures():
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        caps = [make_mock_cap(), make_mock_cap()]
        mock_vc.side_effect = caps
        sm = StreamManager(cameras=FAKE_CAMERAS)
        sm.start()
        time.sleep(0.1)
        sm.stop()
        for cap in caps:
            cap.release.assert_called_once()
