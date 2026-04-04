"""
Face enrollment — generates InsightFace embedding from uploaded photo,
encrypts with per-client AES-256 key, stores in DB.
"""
# TODO Phase 2: Implement


class FaceEnrollmentService:
    def enroll(self, client_id: str, person_id: str, photo_bytes: bytes) -> bytes:
        """
        1. Run InsightFace on photo to get 512-d embedding
        2. Encrypt embedding with client's AES-256 key from secrets manager
        3. Return encrypted blob for storage in DB
        Face data is NEVER logged or included in error traces.
        """
        raise NotImplementedError
