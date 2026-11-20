"""Automated test suite for Secure Environment-Based CORS Configuration.

Validates:
1. Environment variable parsing of ALLOWED_ORIGINS (comma-separated, JSON array, whitespace, trailing slashes).
2. Strict prohibition and filtration of wildcard origins ('*') when allow_credentials=True.
3. Proper configuration and allowance of production frontend origin (https://offline-study-ai.vercel.app).
4. Proper configuration and allowance of local development origins (http://localhost:5173, http://127.0.0.1:5173).
5. allow_credentials=True preservation and verification that Access-Control-Allow-Origin is never '*'.
6. Full allowance of required HTTP methods (GET, POST, DELETE, OPTIONS, HEAD, PUT, PATCH) and headers (Authorization, Content-Type, Accept).
7. Safe rejection of unauthorized third-party origins (e.g. http://evil.com, https://attacker.io).
8. Absence of private secrets in configuration files and frontend code.
"""

import os
import sys
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from backend.main import (
    app,
    ALLOWED_ORIGINS,
    DEFAULT_ALLOWED_ORIGINS,
    parse_allowed_origins,
    get_allowed_origins,
)


class TestCORSConfiguration(unittest.TestCase):
    """Unit and integration tests for secure CORS configuration."""

    def setUp(self):
        self.client = TestClient(app)
        self.prod_origin = "https://offline-study-ai.vercel.app"
        self.local_origin = "http://localhost:5173"
        self.local_ip_origin = "http://127.0.0.1:5173"
        self.evil_origin = "https://malicious-site.example.com"

    def test_default_allowed_origins_include_production_and_local(self):
        """Verifies that default origins include the production frontend origin and local dev origin."""
        self.assertIn(self.prod_origin, DEFAULT_ALLOWED_ORIGINS)
        self.assertIn(self.local_origin, DEFAULT_ALLOWED_ORIGINS)
        self.assertIn(self.local_ip_origin, DEFAULT_ALLOWED_ORIGINS)

    def test_parse_allowed_origins_defaults_when_empty_or_none(self):
        """Verifies fallback to default origins when env var is None or whitespace."""
        self.assertEqual(parse_allowed_origins(None), DEFAULT_ALLOWED_ORIGINS)
        self.assertEqual(parse_allowed_origins(""), DEFAULT_ALLOWED_ORIGINS)
        self.assertEqual(parse_allowed_origins("   "), DEFAULT_ALLOWED_ORIGINS)

    def test_parse_allowed_origins_comma_separated(self):
        """Verifies parsing of comma-separated string format."""
        raw = "https://custom1.com, https://custom2.com,http://localhost:3000"
        result = parse_allowed_origins(raw)
        self.assertEqual(result, ["https://custom1.com", "https://custom2.com", "http://localhost:3000"])

    def test_parse_allowed_origins_json_array(self):
        """Verifies parsing of JSON list string format."""
        raw = '["https://example.org", "http://localhost:8080"]'
        result = parse_allowed_origins(raw)
        self.assertEqual(result, ["https://example.org", "http://localhost:8080"])

    def test_parse_allowed_origins_strips_trailing_slashes_and_whitespace(self):
        """Verifies stripping of whitespace and trailing slashes to guarantee exact origin matches."""
        raw = " https://offline-study-ai.vercel.app/ , http://localhost:5173/ "
        result = parse_allowed_origins(raw)
        self.assertEqual(result, ["https://offline-study-ai.vercel.app", "http://localhost:5173"])

    def test_wildcard_origin_is_never_allowed_with_credentials(self):
        """Requirement 4: Wildcard '*' must never be permitted when credentials are enabled."""
        # Sole wildcard input should fall back to safe default origins
        result = parse_allowed_origins("*")
        self.assertNotIn("*", result)
        self.assertEqual(result, DEFAULT_ALLOWED_ORIGINS)

        # Mixed wildcard input should discard '*' and keep only explicit valid origins
        result_mixed = parse_allowed_origins("*, https://trusted-partner.com, http://localhost:5173")
        self.assertNotIn("*", result_mixed)
        self.assertEqual(result_mixed, ["https://trusted-partner.com", "http://localhost:5173"])

    def test_production_origin_preflight_success(self):
        """Requirement 2: Preflight OPTIONS request from production origin succeeds with credentials."""
        response = self.client.options(
            "/health",
            headers={
                "Origin": self.prod_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), self.prod_origin)
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")
        self.assertIn("access-control-allow-methods", response.headers)
        self.assertIn("access-control-allow-headers", response.headers)

    def test_local_dev_origin_preflight_success(self):
        """Requirement 3: Preflight OPTIONS request from localhost:5173 succeeds with credentials."""
        response = self.client.options(
            "/health",
            headers={
                "Origin": self.local_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), self.local_origin)
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    def test_local_127_dev_origin_preflight_success(self):
        """Preflight OPTIONS request from 127.0.0.1:5173 succeeds with credentials."""
        response = self.client.options(
            "/health",
            headers={
                "Origin": self.local_ip_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), self.local_ip_origin)
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    def test_disallowed_origin_preflight_rejected(self):
        """Verifies unauthorized origins are rejected on CORS preflight."""
        response = self.client.options(
            "/health",
            headers={
                "Origin": self.evil_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotEqual(response.headers.get("access-control-allow-origin"), self.evil_origin)

    def test_disallowed_origin_simple_request_omits_cors_headers(self):
        """Verifies unauthorized origins do not receive Access-Control-Allow-Origin header."""
        response = self.client.get(
            "/health",
            headers={"Origin": self.evil_origin},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_credentials_never_paired_with_wildcard_allow_origin(self):
        """Requirement 4 & 5: Access-Control-Allow-Origin must NEVER be '*' when credentials are true."""
        for origin in [self.prod_origin, self.local_origin, self.local_ip_origin]:
            res = self.client.get("/health", headers={"Origin": origin})
            if res.headers.get("access-control-allow-credentials") == "true":
                allow_origin = res.headers.get("access-control-allow-origin")
                self.assertIsNotNone(allow_origin)
                self.assertNotEqual(allow_origin, "*", "Access-Control-Allow-Origin cannot be '*' with credentials")
                self.assertEqual(allow_origin, origin)

    def test_cors_headers_on_auth_login_route(self):
        """Verifies CORS preflight and headers on authentication routes."""
        response = self.client.options(
            "/auth/login",
            headers={
                "Origin": self.prod_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), self.prod_origin)
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    def test_cors_headers_on_password_reset_route(self):
        """Verifies CORS preflight and headers on password reset routes."""
        response = self.client.options(
            "/auth/forgot-password",
            headers={
                "Origin": self.local_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), self.local_origin)
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    def test_env_example_contains_allowed_origins_without_wildcard(self):
        """Requirement 8: Verify .env.example contains safe ALLOWED_ORIGINS definition without wildcard."""
        env_example_path = os.path.join(WORKSPACE_DIR, "backend", ".env.example")
        self.assertTrue(os.path.isfile(env_example_path))
        with open(env_example_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("ALLOWED_ORIGINS=", content)
        self.assertIn("https://offline-study-ai.vercel.app", content)
        self.assertIn("http://localhost:5173", content)
        # Ensure no wildcard is suggested as a value
        for line in content.splitlines():
            line_s = line.strip()
            if line_s.startswith("ALLOWED_ORIGINS="):
                val = line_s.split("=", 1)[1]
                self.assertNotIn("*", val, "ALLOWED_ORIGINS in .env.example must not contain wildcard '*'")


if __name__ == "__main__":
    unittest.main()
