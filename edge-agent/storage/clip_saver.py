"""
Clip saver — saves a 10s video clip around a detection event,
then uploads to cloud storage with client-scoped path.
"""
# TODO Phase 2: Implement


class ClipSaver:
    """Saves and uploads event clips triggered by AI detections."""

    CLIP_DURATION_SECONDS = 10

    def __init__(self, cloud_base_url: str, agent_token: str, local_clip_dir: str):
        self.cloud_base_url = cloud_base_url
        self.agent_token = agent_token
        self.local_clip_dir = local_clip_dir

    def save_and_upload(self, camera_id: str, timestamp: float, event_id: str) -> str:
        """
        Extract clip from camera buffer around timestamp, upload to cloud.
        Returns clip_url on cloud storage.
        """
        raise NotImplementedError
