"""Attendance records, payroll calculation."""
# TODO Phase 2: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/attendance", tags=["attendance"])

# GET /attendance              → records (filter by person, date range)
# GET /attendance/summary      → daily/monthly summary per person
# GET /attendance/payroll      → calculated payroll for a period
