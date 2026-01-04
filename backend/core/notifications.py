import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import logging
from typing import List

from backend.core.config import settings

logger = logging.getLogger(__name__)

def send_match_notification_email(
    identity_id: str,
    confidence: float,
    location: str,
    timestamp: str,
    snapshot: bytes,
    dashboard_link: str,
):
    """
    Sends an email notification for a high-confidence match.
    """
    if not settings.SMTP_HOST or not settings.INVESTIGATOR_EMAILS:
        logger.warning("SMTP settings not configured. Skipping email notification.")
        return

    msg = MIMEMultipart('related')
    msg['Subject'] = f"High-Confidence Suspect Match: {identity_id}"
    msg['From'] = f"{settings.SMTP_SENDER_NAME} <{settings.SMTP_USER}>"
    
    recipient_list = [email.strip() for email in settings.INVESTIGATOR_EMAILS.split(',')]
    msg['To'] = ", ".join(recipient_list)

    html_body = f"""
    <html>
    <body>
        <h2>High-Confidence Suspect Match Detected</h2>
        <p><strong>Identity ID:</strong> {identity_id}</p>
        <p><strong>Confidence Score:</strong> {confidence:.2%}</p>
        <p><strong>Location:</strong> {location}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p>A snapshot of the suspect is attached below:</p>
        <p><img src="cid:snapshot"></p>
        <p><a href="{dashboard_link}">Click here to view the case on the dashboard.</a></p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, 'html'))

    image = MIMEImage(snapshot, name="snapshot.jpg")
    image.add_header('Content-ID', '<snapshot>')
    msg.attach(image)

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_USER, recipient_list, msg.as_string())
            logger.info(f"Successfully sent notification email to {recipient_list}")
    except Exception as e:
        logger.error(f"Failed to send notification email: {e}")
