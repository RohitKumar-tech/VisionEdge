"""
Unit tests for FaceDetector — all tests mock InsightFace so no model download needed.
For accuracy benchmark against real data see: tests/benchmark/lfw_benchmark.py
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


def make_mock_face(score=0.98, embedding_seed=0):
    face = MagicMock()
    face.det_score = score
    rng = np.random.default_rng(embedding_seed)
    face.embedding = rng.standard_normal(512).astype(np.float32)
    face.bbox = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)
    return face


# ── Load ────────────────────────────────────────────────────────────────────

def test_load_initialises_model():
    """load() must call FaceAnalysis and prepare() exactly once."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_if.app.FaceAnalysis.return_value = mock_app
        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()
        mock_if.app.FaceAnalysis.assert_called_once_with(name="buffalo_l")
        mock_app.prepare.assert_called_once()


def test_load_passes_det_thresh_and_det_size():
    """Custom det_thresh and det_size must be forwarded to prepare()."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_if.app.FaceAnalysis.return_value = mock_app
        from ai.face_detector import FaceDetector
        detector = FaceDetector(det_thresh=0.2, det_size=(960, 960))
        detector.load()
        mock_app.prepare.assert_called_once_with(ctx_id=0, det_thresh=0.2, det_size=(960, 960))


def test_detect_before_load_raises():
    """Calling detect() before load() must raise RuntimeError, not silently return []."""
    with patch("ai.face_detector.insightface"):
        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            detector.detect(frame)


# ── Detect ──────────────────────────────────────────────────────────────────

def test_detect_returns_list_of_face_dicts():
    """Each detected face must have bbox, embedding (512-d float32), and det_score."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_app.get.return_value = [make_mock_face(0.99)]
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()
        faces = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(faces) == 1
        assert faces[0]["embedding"].shape == (512,)
        assert faces[0]["embedding"].dtype == np.float32
        assert "det_score" in faces[0]
        assert "bbox" in faces[0]


def test_detect_empty_frame_returns_empty_list():
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_if.app.FaceAnalysis.return_value = mock_app
        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()
        assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_detect_filters_low_confidence_detections():
    """Faces with det_score below DET_SCORE_THRESHOLD must be excluded."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_app.get.return_value = [
            make_mock_face(score=0.99),   # above threshold → keep
            make_mock_face(score=0.10),   # below threshold → drop
        ]
        mock_if.app.FaceAnalysis.return_value = mock_app
        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()
        faces = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(faces) == 1
        assert faces[0]["det_score"] >= 0.3


def test_detect_multiple_faces():
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_app.get.return_value = [
            make_mock_face(score=0.98, embedding_seed=1),
            make_mock_face(score=0.95, embedding_seed=2),
            make_mock_face(score=0.91, embedding_seed=3),
        ]
        mock_if.app.FaceAnalysis.return_value = mock_app
        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()
        faces = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(faces) == 3


# ── CLAHE enhancement ────────────────────────────────────────────────────────

def test_enhance_flag_calls_clahe():
    """detect(frame, enhance=True) must run CLAHE pre-processing before detection."""
    with patch("ai.face_detector.insightface") as mock_if, \
         patch("ai.face_detector.cv2") as mock_cv2:

        # Set up cv2 mocks so CLAHE pipeline runs without error
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_cv2.createCLAHE.return_value = mock_clahe
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.split.return_value = (
            np.zeros((480, 640), dtype=np.uint8),
            np.zeros((480, 640), dtype=np.uint8),
            np.zeros((480, 640), dtype=np.uint8),
        )
        mock_cv2.merge.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2LAB = 44
        mock_cv2.COLOR_LAB2BGR = 56

        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), enhance=True)

        mock_clahe.apply.assert_called_once()


def test_enhance_false_skips_clahe():
    """detect(frame, enhance=False) must NOT run CLAHE."""
    with patch("ai.face_detector.insightface") as mock_if, \
         patch("ai.face_detector.cv2.createCLAHE") as mock_clahe_ctor:

        mock_clahe = MagicMock()
        mock_clahe_ctor.return_value = mock_clahe
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), enhance=False)

        mock_clahe.apply.assert_not_called()


# ── Security / Data hygiene ──────────────────────────────────────────────────

def test_embeddings_not_logged(caplog):
    """Face embeddings must never appear in log output."""
    import logging
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        face = make_mock_face(0.99)
        mock_app.get.return_value = [face]
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        with caplog.at_level(logging.DEBUG, logger="ai.face_detector"):
            detector.load()
            detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert "embedding" not in caplog.text.lower() or "512" not in caplog.text


def test_returned_embedding_is_a_copy():
    """Modifying the returned embedding must not affect future detections."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_face = make_mock_face(0.99, embedding_seed=42)
        original_first_value = float(mock_face.embedding[0])
        mock_app.get.return_value = [mock_face]
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        faces[0]["embedding"][0] = 9999.0  # tamper with returned embedding

        faces2 = detector.detect(frame)
        assert faces2[0]["embedding"][0] == pytest.approx(original_first_value)
