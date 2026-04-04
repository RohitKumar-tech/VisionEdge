"""Site management — each site has one Edge Agent."""
# TODO Phase 1: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/sites", tags=["sites"])

# GET    /sites              → list sites for client
# POST   /sites              → create site, generate agent token
# PUT    /sites/{id}         → update site
# DELETE /sites/{id}         → remove site
# POST   /sites/{id}/rotate-token → issue new agent token (revokes old)
