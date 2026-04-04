"""Camera CRUD, PTZ control relay."""
# TODO Phase 1: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/cameras", tags=["cameras"])

# GET    /cameras              → list cameras for current client
# POST   /cameras              → add camera
# PUT    /cameras/{id}         → update camera
# DELETE /cameras/{id}         → remove camera
# POST   /cameras/{id}/ptz     → relay PTZ command to Edge Agent
# GET    /cameras/{id}/stream-token → issue short-lived HLS token
