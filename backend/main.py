"""
VisionEdge Cloud Backend — FastAPI app entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# TODO Phase 1: Import and register routers
# from api import auth, cameras, persons, events, attendance, reports, notifications, sites, admin

app = FastAPI(title="VisionEdge API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to dashboard domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO Phase 1: app.include_router(auth.router, prefix="/api/v1")
# TODO Phase 1: app.include_router(cameras.router, prefix="/api/v1")
# etc.


@app.get("/health")
def health():
    return {"status": "ok"}
