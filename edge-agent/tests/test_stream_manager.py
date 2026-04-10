import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call
from camera.stream_manager import StreamManager, _redact_rtsp_url

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


# ── Core behaviour ──────────────────────────────────────────────────────────

def test_start_connects_to_all_cameras():
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        mock_vc.side_effect = [make_mock_cap(), make_mock_cap()]
        sm = StreamManager(cameras=FAKE_CAMERAS)
        sm.start()
        time.sleep(0.1)
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
            cap.release.assert_called()


# ── Security ────────────────────────────────────────────────────────────────

def test_credentials_not_logged_in_rtsp_url(caplog):
    """RTSP URLs with credentials must never appear in logs."""
    cameras = [{"camera_id": "cam-sec", "rtsp_url": "rtsp://admin:secret123@192.168.1.10:554/stream"}]
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = make_mock_cap()
        import logging
        with caplog.at_level(logging.INFO, logger="camera.stream_manager"):
            sm = StreamManager(cameras=cameras)
            sm.start()
            time.sleep(0.1)
            sm.stop()
    assert "secret123" not in caplog.text
    assert "admin" not in caplog.text


def test_redact_rtsp_url_strips_credentials():
    assert _redact_rtsp_url("rtsp://admin:password@192.168.1.10/stream") == "rtsp://***@192.168.1.10/stream"
    assert _redact_rtsp_url("rtsp://192.168.1.10/stream") == "rtsp://192.168.1.10/stream"
    assert _redact_rtsp_url("rtsp://user:p@ssw0rd!@10.0.0.1:554/ch01") == "rtsp://***@10.0.0.1:554/ch01"


# ── Frame isolation ──────────────────────────────────────────────────────────

def test_get_frame_returns_copy_not_reference():
    """Modifying the returned frame must not corrupt the stored frame."""
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = make_mock_cap()
        sm = StreamManager(cameras=FAKE_CAMERAS[:1])
        sm.start()
        time.sleep(0.1)

        frame1 = sm.get_frame("cam-1")
        frame1[:] = 255  # paint it white

        frame2 = sm.get_frame("cam-1")
        assert frame2 is not None
        assert not np.all(frame2 == 255), "Stored frame was corrupted by modifying the returned copy"
        sm.stop()


# ── Reconnect ────────────────────────────────────────────────────────────────

def test_reconnects_after_camera_disconnect():
    """
    When a camera becomes unavailable (isOpened → False), the manager must
    attempt to reconnect rather than spinning forever or giving up.
    """
    with patch("camera.stream_manager.cv2.VideoCapture") as mock_vc, \
         patch("camera.stream_manager._RECONNECT_DELAY", 0.05):  # fast reconnect in test

        first_cap = MagicMock()
        first_cap.isOpened.return_value = False   # simulate dropped camera

        second_cap = make_mock_cap()              # reconnect succeeds
        mock_vc.side_effect = [first_cap, second_cap]

        sm = StreamManager(cameras=FAKE_CAMERAS[:1])
        sm.start()
        time.sleep(0.3)  # wait long enough for reconnect cycle

        # VideoCapture called twice — once on start, once on reconnect
        assert mock_vc.call_count == 2
        sm.stop()
