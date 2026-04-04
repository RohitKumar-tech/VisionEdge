"""
JWT auth middleware — validates token, extracts client_id server-side.
client_id is NEVER accepted from request body or query params.
All downstream queries are scoped to this client_id automatically.
"""
# TODO Phase 1: Implement
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()


async def get_current_client(token=Depends(security)):
    """
    FastAPI dependency injected into every protected endpoint.
    Returns {user_id, client_id, role, site_ids}.
    Raises 401 if token invalid/expired.
    """
    # TODO: Decode JWT, validate signature, check expiry
    # client_id extracted from token payload — never from request
    raise NotImplementedError


async def get_agent(token=Depends(security)):
    """
    Dependency for Edge Agent endpoints.
    Returns {site_id, client_id} from agent token.
    Raises 401 if agent token invalid.
    """
    raise NotImplementedError
