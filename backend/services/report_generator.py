"""
Report generator — produces Excel (.xlsx) and PDF attendance/event reports.
Uses openpyxl for Excel, ReportLab for PDF.
"""
# TODO Phase 2: Implement


class ReportGenerator:
    def attendance_excel(self, client_id: str, start_date: str, end_date: str) -> bytes:
        """Generate .xlsx attendance report. Returns file bytes."""
        raise NotImplementedError

    def attendance_pdf(self, client_id: str, start_date: str, end_date: str) -> bytes:
        """Generate PDF attendance report. Returns file bytes."""
        raise NotImplementedError
