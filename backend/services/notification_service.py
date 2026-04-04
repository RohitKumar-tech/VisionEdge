"""
Notification service — FCM push + WhatsApp Business API.
Called by Celery workers, not directly from API endpoints.
"""
# TODO Phase 3: Implement


class NotificationService:
    def send_fcm(self, device_tokens: list[str], title: str, body: str, data: dict):
        """Send FCM push notification to all owner/manager devices."""
        raise NotImplementedError

    def send_whatsapp(self, phone: str, message: str):
        """Send WhatsApp Business API message. ~₹0.40–0.80 per message."""
        raise NotImplementedError
