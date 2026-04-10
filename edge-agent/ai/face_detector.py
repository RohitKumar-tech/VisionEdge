"""
InsightFace wrapper — detects faces and generates 512-d embeddings.
Uses buffalo_l model (99%+ accuracy). Runs on CPU or GPU (Jetson).

Security note: face embeddings are never logged or included in error traces.
"""
import insightface
import numpy as np
import logging

logger = logging.getLogger(__name__)

DET_SCORE_THRESHOLD = 0.5  # minimum detection confidence to return a face


class FaceDetector:
    """
    Detects all faces in a frame and returns their 512-d embeddings.

    Usage:
        detector = FaceDetector()
        detector.load()          # call once on startup (~2-5s on Jetson)
        faces = detector.detect(frame)
        # faces → [{"bbox": array, "embedding": array(512,), "det_score": float}, ...]
    """

    MODEL_NAME = "buffalo_l"

    def __init__(self):
        self._app = None

    def load(self):
        """
        Download (first run) and load InsightFace buffalo_l model.
        Must be called before detect(). Safe to call once at agent startup.
        """
        self._app = insightface.app.FaceAnalysis(name=self.MODEL_NAME)
        self._app.prepare(ctx_id=0)  # ctx_id=0 → GPU if available, else CPU
        logger.info(f"FaceDetector loaded ({self.MODEL_NAME})")

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect all faces in frame above DET_SCORE_THRESHOLD.

        Returns list of dicts:
            bbox      — np.ndarray [x1, y1, x2, y2] float32
            embedding — np.ndarray (512,) float32 copy — safe to modify
            det_score — float, detection confidence

        Face embedding values are never logged.
        Raises RuntimeError if load() has not been called.
        """
        if self._app is None:
            raise RuntimeError("FaceDetector.load() must be called before detect()")

        raw_faces = self._app.get(frame)
        return [
            {
                "bbox": face.bbox.copy(),
                "embedding": face.embedding.copy(),  # copy — caller can modify freely
                "det_score": float(face.det_score),
            }
            for face in raw_faces
            if face.det_score >= DET_SCORE_THRESHOLD
        ]
