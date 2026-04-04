"""Notification preferences — per person type, per channel."""
# TODO Phase 3: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/notifications", tags=["notifications"])

# GET  /notifications/preferences       → get client notification config
# PUT  /notifications/preferences       → update (which events trigger FCM/WhatsApp)
# POST /notifications/test              → send a test notification
