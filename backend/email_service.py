import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Ensure environment variables from backend/.env and workspace root .env are loaded
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))
load_dotenv()


def get_smtp_config() -> Dict[str, Any]:
    """Retrieves and normalizes SMTP configuration from environment variables."""
    host = os.getenv("SMTP_HOST", "").strip()
    port_str = os.getenv("SMTP_PORT", "587").strip()
    try:
        port = int(port_str)
    except ValueError:
        port = 587

    user = (os.getenv("SMTP_USER") or os.getenv("SMTP_USERNAME") or "").strip()
    password = (os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS") or "").strip()
    from_email = (os.getenv("SMTP_FROM_EMAIL") or user).strip()
    from_name = os.getenv("SMTP_FROM_NAME", "Offline Study AI").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")
    use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in ("true", "1", "yes")
    timeout_str = os.getenv("SMTP_TIMEOUT", "10").strip()
    try:
        timeout = int(timeout_str)
    except ValueError:
        timeout = 10

    # Consider SMTP configured if host and either user or from_email are set
    is_configured = bool(host and (user or from_email))

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "from_email": from_email,
        "from_name": from_name,
        "use_tls": use_tls,
        "use_ssl": use_ssl,
        "timeout": timeout,
        "is_configured": is_configured,
    }


def send_password_reset_email(to_email: str, reset_code: str, username: Optional[str] = None) -> bool:
    """Sends a 6-digit password reset verification code to the recipient via secure SMTP.

    Raises:
        ValueError: If SMTP configuration is missing or recipient is invalid.
        smtplib.SMTPException: If the email server refuses or fails delivery.
        OSError: If connection to SMTP server times out or fails.
    """
    config = get_smtp_config()
    if not config["is_configured"]:
        raise ValueError(
            "SMTP email service is not configured. "
            "Please configure SMTP_HOST, SMTP_USER, and SMTP_PASSWORD in backend/.env to receive password reset emails."
        )

    clean_to = (to_email or "").strip()
    if not clean_to or "@" not in clean_to:
        raise ValueError(f"Invalid recipient email address: '{to_email}'")

    display_name = (username or clean_to.split("@")[0]).strip()

    subject = f"Your Offline Study AI Password Reset Code: {reset_code}"

    text_body = f"""Hello {display_name},

You requested to reset your password for Offline Study AI.

Your 6-digit verification code is:
{reset_code}

This code is valid for 15 minutes. It can only be used once.

If you did not request this password reset, please ignore this email. Your password will remain unchanged.

Best regards,
The Offline Study AI Team
"""

    html_body = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Password Reset Code</title>
</head>
<body style="margin: 0; padding: 24px; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <div style="max-width: 520px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
    <div style="background-color: #0f172a; padding: 24px 28px; text-align: center;">
      <h1 style="color: #ffffff; font-size: 20px; margin: 0; font-weight: 700; letter-spacing: -0.02em;">Offline Study AI</h1>
      <p style="color: #94a3b8; font-size: 13px; margin: 4px 0 0 0;">Password Reset Verification</p>
    </div>
    <div style="padding: 28px;">
      <p style="color: #334155; font-size: 15px; margin: 0 0 16px 0;">Hello <strong>{display_name}</strong>,</p>
      <p style="color: #475569; font-size: 14px; line-height: 1.6; margin: 0 0 24px 0;">
        We received a request to reset your password. Use the 6-digit verification code below to complete your reset:
      </p>
      <div style="text-align: center; margin: 24px 0;">
        <div style="display: inline-block; background-color: #f1f5f9; border: 2px dashed #2563eb; border-radius: 10px; padding: 14px 28px; font-size: 32px; font-weight: 800; letter-spacing: 6px; color: #1e3a8a; user-select: all;">
          {reset_code}
        </div>
      </div>
      <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px; margin: 20px 0;">
        <p style="color: #475569; font-size: 13px; margin: 0; line-height: 1.5;">
          ⏱️ <strong>Valid for 15 minutes.</strong> This code can only be used once.
        </p>
      </div>
      <p style="color: #64748b; font-size: 13px; line-height: 1.5; margin: 20px 0 0 0;">
        If you did not request this password reset, you can safely ignore this email. Your account remains secure.
      </p>
    </div>
    <div style="background-color: #f8fafc; padding: 16px 28px; text-align: center; border-top: 1px solid #f1f5f9;">
      <p style="color: #94a3b8; font-size: 12px; margin: 0;">Offline Study AI — Private AI Learning Assistant with Isolated ChromaDB Vaults</p>
    </div>
  </div>
</body>
</html>
"""

    msg = EmailMessage()
    msg["Subject"] = subject
    from_header = f"{config['from_name']} <{config['from_email']}>" if config["from_name"] else config["from_email"]
    msg["From"] = from_header
    msg["To"] = clean_to
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    host = config["host"]
    port = config["port"]
    timeout = config["timeout"]
    user = config["user"]
    password = config["password"]

    if config["use_ssl"] or port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=timeout) as server:
            if user and password:
                server.login(user, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as server:
            if config["use_tls"]:
                context = ssl.create_default_context()
                server.starttls(context=context)
            if user and password:
                server.login(user, password)
            server.send_message(msg)

    return True
