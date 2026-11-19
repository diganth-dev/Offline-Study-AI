"""Runtime validation test for PWA assets and HTTP serving.

Validates:
1. Static HTTP serving of the application shell
2. Web app manifest delivery with valid JSON
3. Service worker script delivery
4. Icon asset delivery (192, 512, maskable, apple-touch-icon)
5. Offline shell availability simulation
"""

import http.server
import json
import os
import socketserver
import threading
import time
import urllib.request
import unittest

DIST_DIR = os.path.abspath(os.path.join("frontend", "dist"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIST_DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress console logging during tests


class TestPWARuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 41735
        cls.httpd = socketserver.TCPServer(("127.0.0.1", cls.port), Handler)
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _fetch(self, path):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "PWA-Test-Client"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.read(), response.headers

    def test_app_shell_html_served(self):
        """Verifies root URL serves index.html shell."""
        status, body, headers = self._fetch("/index.html")
        self.assertEqual(status, 200)
        content = body.decode("utf-8")
        self.assertIn("<!doctype html>", content.lower())
        self.assertIn('<div id="root"></div>', content)
        self.assertIn("registerSW.js", content)
        self.assertIn("manifest.webmanifest", content)

    def test_manifest_webmanifest_served(self):
        """Verifies manifest is valid JSON and contains required PWA metadata."""
        status, body, headers = self._fetch("/manifest.webmanifest")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data.get("name"), "Offline Study AI")
        self.assertEqual(data.get("short_name"), "Study AI")
        self.assertEqual(data.get("display"), "standalone")
        self.assertEqual(data.get("theme_color"), "#0f172a")
        self.assertEqual(data.get("background_color"), "#f8fafc")
        self.assertGreaterEqual(len(data.get("icons", [])), 3)

    def test_service_worker_script_served(self):
        """Verifies sw.js and registerSW.js are served correctly."""
        status, body, _ = self._fetch("/sw.js")
        self.assertEqual(status, 200)
        self.assertIn(b"precacheAndRoute", body)

        status, body, _ = self._fetch("/registerSW.js")
        self.assertEqual(status, 200)
        self.assertIn(b"navigator.serviceWorker.register", body)

    def test_icons_served(self):
        """Verifies all required PWA icons are served as valid PNG files."""
        icon_paths = [
            "/icons/icon-192x192.png",
            "/icons/icon-512x512.png",
            "/icons/maskable-icon-192x192.png",
            "/icons/maskable-icon-512x512.png",
            "/apple-touch-icon.png",
            "/icons/favicon-32x32.png",
            "/favicon.svg",
        ]
        png_magic = b"\x89PNG\r\n\x1a\n"

        for path in icon_paths:
            status, body, headers = self._fetch(path)
            self.assertEqual(status, 200, f"Failed to fetch {path}")
            self.assertGreater(len(body), 50, f"Icon {path} is suspiciously small")
            if path.endswith(".png"):
                self.assertTrue(body.startswith(png_magic), f"Icon {path} lacks PNG magic bytes")

    def test_offline_shell_completeness(self):
        """Verifies all assets referenced in sw.js precache list exist and are non-empty."""
        sw_path = os.path.join(DIST_DIR, "sw.js")
        with open(sw_path, "r", encoding="utf-8") as f:
            sw_text = f.read()

        import re
        urls = re.findall(r'url:"([^"]+)"', sw_text)
        self.assertGreater(len(urls), 5, "Precache list is too small")

        for rel_url in urls:
            full_path = os.path.join(DIST_DIR, rel_url.replace("/", os.sep))
            self.assertTrue(
                os.path.isfile(full_path),
                f"Precached file does not exist in dist: {rel_url} ({full_path})",
            )
            self.assertGreater(
                os.path.getsize(full_path),
                0,
                f"Precached file is empty: {rel_url}",
            )


if __name__ == "__main__":
    unittest.main()
