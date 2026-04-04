"""
HLS streaming server — serves each camera as an HLS stream via ffmpeg.
Endpoint: http://edge-agent-local-ip:{port}/cam/{camera_id}/stream.m3u8
Validates short-lived stream tokens issued by cloud backend.
"""
# TODO Phase 4: Implement


class HLSServer:
    """Wraps ffmpeg to transcode RTSP → HLS segments for local viewing."""

    def __init__(self, cameras: list, port: int = 8080):
        self.cameras = cameras
        self.port = port

    def start(self):
        raise NotImplementedError

    def validate_stream_token(self, token: str, camera_id: str) -> bool:
        """Validate token issued by cloud. Reject if expired or wrong camera scope."""
        raise NotImplementedError
