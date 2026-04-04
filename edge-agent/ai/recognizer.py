"""
Face recognizer — matches detected embeddings against local SQLite person DB.
Fires event if confidence >= threshold (default 95%).
"""
# TODO Phase 2: Implement
import numpy as np


class Recognizer:
    """Matches face embeddings against locally-cached person database."""

    def __init__(self, db_path: str, confidence_threshold: float = 0.95):
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold
        self._persons = []  # loaded from SQLite on start + on sync

    def reload_persons(self):
        """Reload face embeddings from local SQLite (called after face sync)."""
        raise NotImplementedError

    def match(self, embedding: np.ndarray) -> dict | None:
        """
        Compare embedding against all known persons.
        Returns {person_id, name, type, confidence} or None if no match above threshold.
        """
        raise NotImplementedError
