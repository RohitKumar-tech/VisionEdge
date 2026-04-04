"""
InsightFace wrapper — detects and embeds faces from frames.
Samples at 1 fps (configurable). Uses buffalo_l model for 99%+ accuracy.
"""
# TODO Phase 2: Implement
import numpy as np


class FaceDetector:
    """Detects faces and generates 512-d embeddings via InsightFace."""

    MODEL_NAME = "buffalo_l"

    def __init__(self):
        self._app = None  # insightface.app.FaceAnalysis

    def load(self):
        """Load InsightFace model. Call once on startup."""
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns list of detected faces with embeddings.
        Each face: {bbox, embedding (512-d float32), det_score}
        """
        raise NotImplementedError
