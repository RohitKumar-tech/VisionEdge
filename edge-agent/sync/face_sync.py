"""
Face sync — polls cloud backend every 60s for updated face embeddings.
Updates local SQLite. Triggers recognizer reload after sync.
"""
# TODO Phase 2: Implement
import time
import logging

logger = logging.getLogger(__name__)


class FaceSync:
    """Keeps local face DB in sync with cloud."""

    POLL_INTERVAL = 60  # seconds

    def __init__(self, cloud_base_url: str, agent_token: str, db_path: str, on_sync=None):
        self.cloud_base_url = cloud_base_url
        self.agent_token = agent_token
        self.db_path = db_path
        self.on_sync = on_sync  # callback to reload recognizer

    def run(self):
        """Run sync loop forever (intended to run in a thread)."""
        while True:
            try:
                self._sync()
            except Exception as e:
                logger.error(f"Face sync error: {e}")
            time.sleep(self.POLL_INTERVAL)

    def _sync(self):
        """Fetch pending sync actions from cloud, update local SQLite."""
        raise NotImplementedError
