"""Platform super-admin — internal use only. Not exposed to clients."""
# TODO Phase 1: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/admin", tags=["admin"])

# GET  /admin/clients           → list all clients
# POST /admin/clients           → create client + first owner user
# PUT  /admin/clients/{id}/tier → change subscription tier
# GET  /admin/agents            → list all Edge Agents + last-seen
