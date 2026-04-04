"""
WebSocket events hub — broadcasts real-time recognition events to connected
React dashboard sessions for the same client.
"""
# TODO Phase 3: Implement
from fastapi import WebSocket
from collections import defaultdict


class EventsHub:
    """Manages WebSocket connections per client_id and broadcasts events."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self._connections[client_id].append(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str):
        self._connections[client_id].remove(websocket)

    async def broadcast(self, client_id: str, event: dict):
        """Send event to all dashboard sessions for this client only."""
        for ws in list(self._connections[client_id]):
            try:
                await ws.send_json(event)
            except Exception:
                self._connections[client_id].remove(ws)


hub = EventsHub()
