"""
Event uploader — POSTs recognition events to cloud backend.
Stores in SQLite offline queue when internet is unavailable,
then uploads when connectivity resumes.
"""
# TODO Phase 2: Implement
import logging

logger = logging.getLogger(__name__)


class EventUploader:
    """Reliably uploads recognition events to cloud with offline queue."""

    def __init__(self, cloud_base_url: str, agent_token: str, db_path: str):
        self.cloud_base_url = cloud_base_url
        self.agent_token = agent_token
        self.db_path = db_path

    def enqueue(self, event: dict):
        """Add event to SQLite queue for upload."""
        raise NotImplementedError

    def run(self):
        """Drain upload queue forever (intended to run in a thread)."""
        raise NotImplementedError
