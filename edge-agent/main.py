"""
VisionEdge Edge Agent — Entry point
Runs on Jetson Nano / Mini PC at client site.
Starts camera stream manager, AI recognition loop, face sync, and event uploader.
"""
import threading
import logging
from config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("edge-agent")


def main():
    config = load_config()
    logger.info(f"Starting VisionEdge Edge Agent — site_id={config.site_id}")

    # TODO Phase 1: Start camera stream manager
    # TODO Phase 1: Start face sync thread (polls cloud every 60s)
    # TODO Phase 1: Start event uploader thread
    # TODO Phase 2: Start AI recognition loop
    # TODO Phase 4: Start HLS streaming server

    logger.info("Edge Agent started. Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down Edge Agent...")


if __name__ == "__main__":
    main()
