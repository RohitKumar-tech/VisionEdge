"""
InsightFace wrapper — detects faces and generates 512-d embeddings.
Uses buffalo_l model (99%+ accuracy on clean footage).

Tuned for CCTV use cases: handles blurred, foggy, noisy, and
low-resolution footage by using a lower detection threshold and
optional frame pre-processing (CLAHE contrast enhancement).

Security note: face embeddings are never logged or included in error traces.
"""
import insightface
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Default parameters — tuned for degraded CCTV footage based on LFW benchmark.
# Lower det_thresh (0.3 vs default 0.5) catches faces in foggy/blurry conditions.
DEFAULT_DET_THRESH = 0.3
DEFAULT_DET_SIZE = (640, 640)

# Minimum detection score to include in results (separate from model's internal threshold)
DET_SCORE_THRESHOLD = 0.3


class FaceDetector:
    """
    Detects all faces in a frame and returns their 512-d embeddings.

    Tuned for challenging CCTV footage (blur, fog, low-res, compression).
    Uses CLAHE pre-processing to enhance contrast in degraded frames.

    Usage:
        detector = FaceDetector()
        detector.load()                    # once on startup, ~2-5s on Jetson
        faces = detector.detect(frame)
        # faces → [{"bbox": array, "embedding": array(512,), "det_score": float}]

    For very degraded footage (heavy fog/blur), call:
        faces = detector.detect(frame, enhance=True)
    """

    MODEL_NAME = "buffalo_l"

    def __init__(self,
                 det_thresh: float = DEFAULT_DET_THRESH,
                 det_size: tuple = DEFAULT_DET_SIZE):
        """
        det_thresh: detection confidence threshold. Lower = finds more faces in
                    degraded conditions but may increase false positives.
                    Recommended: 0.3 (degraded), 0.5 (clean)
        det_size:   detection input size. Larger = finds smaller/distant faces
                    but slower. (640,640) works well for most CCTV setups.
        """
        self._app = None
        self._det_thresh = det_thresh
        self._det_size = det_size
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def load(self):
        """
        Download (first run ~500MB) and load InsightFace buffalo_l model.
        Must be called before detect(). Safe to call once at agent startup.
        """
        self._app = insightface.app.FaceAnalysis(name=self.MODEL_NAME)
        self._app.prepare(ctx_id=0,
                          det_thresh=self._det_thresh,
                          det_size=self._det_size)
        logger.info(
            f"FaceDetector loaded ({self.MODEL_NAME}, "
            f"det_thresh={self._det_thresh}, det_size={self._det_size})"
        )

    def _enhance_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        CLAHE contrast enhancement for foggy/dark/low-contrast frames.
        Operates on the L channel in LAB colour space to avoid colour shift.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def detect(self, frame: np.ndarray, enhance: bool = False) -> list:
        """
        Detect all faces in frame above det_thresh.

        Args:
            frame:   BGR uint8 ndarray (OpenCV format from VideoCapture)
            enhance: if True, apply CLAHE contrast enhancement before detection.
                     Helps with foggy, dark, or washed-out frames.

        Returns list of dicts:
            bbox      — np.ndarray [x1, y1, x2, y2] float32
            embedding — np.ndarray (512,) float32, copy — safe to modify
            det_score — float, detection confidence

        Face embedding values are never logged.
        Raises RuntimeError if load() has not been called.
        """
        if self._app is None:
            raise RuntimeError("FaceDetector.load() must be called before detect()")

        input_frame = self._enhance_frame(frame) if enhance else frame
        raw_faces = self._app.get(input_frame)

        return [
            {
                "bbox": face.bbox.copy(),
                "embedding": face.embedding.copy(),
                "det_score": float(face.det_score),
            }
            for face in raw_faces
            if face.det_score >= DET_SCORE_THRESHOLD
        ]
