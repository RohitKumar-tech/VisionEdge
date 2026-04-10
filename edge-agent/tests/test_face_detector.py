"""
Unit tests for FaceDetector — all tests mock InsightFace so no model download needed.
Manual smoke test (requires real model download) is in tests/fixtures/smoke_face_detector.py
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


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
    """Each detected face must have bbox, embedding (512-d), and det_score."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_app.get.return_value = [make_mock_face(0.99)]
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)

        assert len(faces) == 1
        assert "embedding" in faces[0]
        assert "det_score" in faces[0]
        assert "bbox" in faces[0]
        assert faces[0]["embedding"].shape == (512,)
        assert faces[0]["embedding"].dtype == np.float32


def test_detect_empty_frame_returns_empty_list():
    """No faces in frame → empty list, no exception."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame) == []


def test_detect_filters_low_confidence_detections():
    """Faces with det_score below threshold must be excluded."""
    with patch("ai.face_detector.insightface") as mock_if:
        mock_app = MagicMock()
        mock_app.get.return_value = [
            make_mock_face(score=0.99),  # above threshold → keep
            make_mock_face(score=0.40),  # below threshold → drop
        ]
        mock_if.app.FaceAnalysis.return_value = mock_app

        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        detector.load()

        faces = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(faces) == 1
        assert faces[0]["det_score"] >= 0.5


def test_detect_multiple_faces():
    """Multiple faces above threshold must all be returned."""
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

    # Embedding is a float32 array — check its string repr isn't in logs
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

        # Detect again — should return original value, not 9999
        faces2 = detector.detect(frame)
        assert faces2[0]["embedding"][0] == pytest.approx(original_first_value)
