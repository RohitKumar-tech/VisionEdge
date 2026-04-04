"""Excel / PDF report generation."""
# TODO Phase 2: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/reports", tags=["reports"])

# GET /reports/attendance/excel   → download .xlsx attendance report
# GET /reports/attendance/pdf     → download PDF attendance report
# GET /reports/events/pdf         → download PDF event log
