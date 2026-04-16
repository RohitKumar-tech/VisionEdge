"""
Edge Agent configuration — reads from /etc/visionedge/agent.conf or env vars.
"""
import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    site_id: str
    agent_token: str
    cloud_base_url: str
    # Each camera dict: {camera_id, rtsp_url, role}
    # role: "entry" → checkin only | "exit" → checkout only | "both" → checkin only (default)
    cameras: list
    recognition_fps: int = 1
    confidence_threshold: float = 0.95
    hls_port: int = 8080


def load_config() -> AgentConfig:
    # TODO: Load from /etc/visionedge/agent.conf (TOML or YAML)
    return AgentConfig(
        site_id=os.environ.get("SITE_ID", ""),
        agent_token=os.environ.get("AGENT_TOKEN", ""),
        cloud_base_url=os.environ.get("CLOUD_BASE_URL", "https://api.visionedge.in"),
        cameras=[],
    )
