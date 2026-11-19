"""Unit tests for SMTP Email Delivery Service in Offline Study AI.

Validates:
1. Retrieval and normalization of SMTP configuration from environment variables.
2. Enforcement of missing configuration errors (safe failure when credentials not set).
3. Plain text and HTML multipart email composition with 6-digit verification code.
4. Secure SMTP delivery handling with STARTTLS (port 587) and SSL (port 465).
5. Error propagation when SMTP server connection or authentication fails.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from backend.email_service import get_smtp_config, send_password_reset_email


class TestEmailService(unittest.TestCase):
    """Test suite for SMTP email service configuration, message construction, and delivery."""

    def test_unconfigured_smtp_raises_value_error(self):
        """Verifies that attempting to send without SMTP_HOST raises a clear descriptive ValueError."""
        with patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": ""}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                send_password_reset_email("student@example.com", "123456")
            self.assertIn("SMTP email service is not configured", str(ctx.exception))

    def test_invalid_recipient_raises_value_error(self):
        """Verifies that an invalid email address raises a ValueError."""
        with patch.dict(os.environ, {"SMTP_HOST": "smtp.gmail.com", "SMTP_USER": "test@gmail.com"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                send_password_reset_email("invalid-email-format", "123456")
            self.assertIn("Invalid recipient email address", str(ctx.exception))

    @patch("smtplib.SMTP")
    def test_smtp_starttls_delivery_success(self, mock_smtp_cls):
        """Tests successful email delivery using standard STARTTLS on port 587 (Gmail default)."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        env_vars = {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "app.study.ai@gmail.com",
            "SMTP_PASSWORD": "abcd1234efgh5678",
            "SMTP_FROM_EMAIL": "app.study.ai@gmail.com",
            "SMTP_FROM_NAME": "Offline Study AI",
            "SMTP_USE_TLS": "true",
            "SMTP_USE_SSL": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            result = send_password_reset_email("student@gmail.com", "583921", username="John Doe")
            self.assertTrue(result)

            # Verify connection host and port
            mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=10)

            # Verify starttls and login calls
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("app.study.ai@gmail.com", "abcd1234efgh5678")

            # Verify message was sent
            mock_server.send_message.assert_called_once()
            sent_msg = mock_server.send_message.call_args[0][0]

            self.assertEqual(sent_msg["To"], "student@gmail.com")
            self.assertIn("Offline Study AI", sent_msg["From"])
            self.assertIn("583921", sent_msg["Subject"])

            # Verify message payload contains the verification code
            payload_str = str(sent_msg)
            self.assertIn("583921", payload_str)
            self.assertIn("15 minutes", payload_str)

    @patch("smtplib.SMTP_SSL")
    def test_smtp_ssl_delivery_success(self, mock_ssl_cls):
        """Tests successful email delivery using direct SSL on port 465."""
        mock_server = MagicMock()
        mock_ssl_cls.return_value.__enter__.return_value = mock_server

        env_vars = {
            "SMTP_HOST": "smtp.mailserver.com",
            "SMTP_PORT": "465",
            "SMTP_USER": "mailer@study.org",
            "SMTP_PASSWORD": "secret_smtp_password",
            "SMTP_USE_SSL": "true",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            result = send_password_reset_email("target@example.com", "849201")
            self.assertTrue(result)

            mock_ssl_cls.assert_called_once_with("smtp.mailserver.com", 465, context=unittest.mock.ANY, timeout=10)
            mock_server.login.assert_called_once_with("mailer@study.org", "secret_smtp_password")
            mock_server.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    def test_smtp_server_exception_propagates(self, mock_smtp_cls):
        """Verifies that smtplib errors are not swallowed and propagate cleanly."""
        mock_smtp_cls.side_effect = ConnectionRefusedError("Connection refused by SMTP host")

        env_vars = {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USER": "user@gmail.com",
            "SMTP_PASSWORD": "password",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with self.assertRaises(ConnectionRefusedError):
                send_password_reset_email("user@example.com", "123456")


if __name__ == "__main__":
    unittest.main()
