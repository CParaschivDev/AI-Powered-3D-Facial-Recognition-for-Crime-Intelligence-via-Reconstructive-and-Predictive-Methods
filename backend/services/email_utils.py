import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import BackgroundTasks
from typing import Any

def send_email(subject: str, body: str, to_email: str):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER", "your_email@gmail.com")
    smtp_password = os.getenv("SMTP_PASSWORD", "your_password")

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, msg.as_string())

def send_analysis_email(analysis: str, to_email: str = "2026077@student.uwtsd.ac.uk"):
    subject = "Automated Model Analysis Result"
    body = f"Here is the result of your model analysis:\n\n{analysis}"
    send_email(subject, body, to_email)
