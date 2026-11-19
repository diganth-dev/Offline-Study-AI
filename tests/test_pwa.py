"""Test suite for Progressive Web App (PWA) configuration and assets.

Validates:
1. Production build output
2. Web App Manifest (name, short_name, icons, display, theme colors, shortcuts)
3. Service worker precaching and caching rules
4. API exclusion from static caching (NetworkOnly)
5. PWA HTML metadata and icon tags
"""

import os
import json
import re
import unittest

DIST_DIR = os.path.join("frontend", "dist")
PUBLIC_DIR = os.path.join("frontend", "public")


class TestPWAConfiguration(unittest.TestCase):

    def test_dist_build_exists(self):
        """Verifies that the frontend production build was generated."""
        self.assertTrue(os.path.isdir(DIST_DIR), "frontend/dist directory does not exist")
        self.assertTrue(os.path.isfile(os.path.join(DIST_DIR, "index.html")), "dist/index.html missing")
        self.assertTrue(os.path.isfile(os.path.join(DIST_DIR, "sw.js")), "dist/sw.js missing")
        self.assertTrue(os.path.isfile(os.path.join(DIST_DIR, "manifest.webmanifest")), "dist/manifest.webmanifest missing")
        self.assertTrue(os.path.isfile(os.path.join(DIST_DIR, "registerSW.js")), "dist/registerSW.js missing")

    def test_web_app_manifest(self):
        """Verifies manifest requirements: name, short_name, standalone, theme_color, icons."""
        manifest_path = os.path.join(DIST_DIR, "manifest.webmanifest")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("name"), "Offline Study AI")
        self.assertEqual(manifest.get("short_name"), "Study AI")
        self.assertEqual(manifest.get("display"), "standalone")
        self.assertEqual(manifest.get("theme_color"), "#0f172a")
        self.assertEqual(manifest.get("background_color"), "#f8fafc")
        self.assertEqual(manifest.get("start_url"), "/")
        self.assertEqual(manifest.get("scope"), "/")

        # Check icons
        icons = manifest.get("icons", [])
        self.assertGreaterEqual(len(icons), 3, "Manifest should contain at least 3 icon definitions")

        sizes = [icon.get("sizes") for icon in icons]
        self.assertIn("192x192", sizes, "192x192 icon missing from manifest")
        self.assertIn("512x512", sizes, "512x512 icon missing from manifest")

        # Check for maskable icon
        purposes = [icon.get("purpose") for icon in icons if "purpose" in icon]
        self.assertIn("maskable", purposes, "Maskable icon missing from manifest")

        # Check icon files actually exist on disk
        for icon in icons:
            src = icon.get("src", "").lstrip("/")
            icon_file_dist = os.path.join(DIST_DIR, src)
            icon_file_pub = os.path.join(PUBLIC_DIR, src)
            self.assertTrue(
                os.path.isfile(icon_file_dist) or os.path.isfile(icon_file_pub),
                f"Icon file referenced in manifest does not exist: {src}",
            )

    def test_shortcuts(self):
        """Verifies PWA manifest shortcuts for quick access."""
        manifest_path = os.path.join(DIST_DIR, "manifest.webmanifest")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        shortcuts = manifest.get("shortcuts", [])
        self.assertGreaterEqual(len(shortcuts), 2, "Shortcuts should be configured in manifest")
        shortcut_names = [s.get("name") for s in shortcuts]
        self.assertIn("AI Study Chat", shortcut_names)
        self.assertIn("AI Summarizer", shortcut_names)

    def test_service_worker_configuration(self):
        """Verifies service worker precache and NetworkOnly routing for backend APIs."""
        sw_path = os.path.join(DIST_DIR, "sw.js")
        with open(sw_path, "r", encoding="utf-8") as f:
            sw_content = f.read()

        # Check that SW precaches HTML, JS, CSS, and icons
        self.assertIn("index.html", sw_content)
        self.assertIn("precacheAndRoute", sw_content)

        # Check that SW denies caching on backend API endpoints
        self.assertIn("/ask", sw_content)
        self.assertIn("/documents", sw_content)
        self.assertIn("/summarize", sw_content)
        self.assertIn("/important-questions", sw_content)
        self.assertIn("NetworkOnly", sw_content)

    def test_html_meta_tags(self):
        """Verifies index.html has PWA meta tags, apple-touch-icon, and manifest link."""
        html_path = os.path.join(DIST_DIR, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn('<meta name="theme-color" content="#0f172a"', html)
        self.assertIn('<link rel="manifest" href="/manifest.webmanifest">', html)
        self.assertIn('apple-touch-icon', html)
        self.assertIn('registerSW.js', html)


if __name__ == "__main__":
    unittest.main()
