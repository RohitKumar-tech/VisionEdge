"""Receive recognition events from Edge Agents."""
# TODO Phase 2: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/events", tags=["events"])

# POST /events              → receive event from Edge Agent (agent token auth)
# GET  /events              → list events for client (dashboard use)
# GET  /events/{id}         → event detail + clip URL
