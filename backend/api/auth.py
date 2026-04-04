"""Login, refresh token, logout endpoints."""
# TODO Phase 1: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["auth"])

# POST /auth/login       → {access_token, refresh_token}
# POST /auth/refresh     → {access_token}
# POST /auth/logout      → 204
