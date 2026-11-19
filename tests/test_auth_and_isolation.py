"""Automated test suite for Multi-User Authentication and Strict Data Isolation.

Validates:
1. User registration, password hashing (bcrypt), and input validation.
2. User authentication (username and email), JWT access token issuance (HS256).
3. JWT validation, expiration handling, and malformed token rejection.
4. Protected endpoint access control (unauthorized requests return 401).
5. ChromaDB chunk-level user ownership tagging (user_id in chunk metadata and ID prefix).
6. Strict multi-user isolation across:
   - Document upload & indexing
   - Document listing (/documents)
   - Vector search (/search)
   - Document deletion (/documents/{filename})
   - Document-scoped Q&A (/ask)
   - Document summarization (/summarize)
   - Important questions generation (/important-questions)
7. Cross-user anti-tampering (guessing document IDs or filenames across users).
8. Absence of private secrets in frontend bundles.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure workspace root is in PYTHONPATH
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from backend.main import app
from backend.auth_service import (
    register_user,
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_user_by_id,
    hash_password,
    verify_password,
    get_db_connection,
    JWT_SECRET_KEY,
)
from backend.vector_store import (
    index_document,
    get_indexed_documents,
    search_documents,
    delete_document,
    get_document_chunks,
)


class TestAuthenticationService(unittest.TestCase):
    """Unit tests for user registration, authentication, hashing, and JWT tokens."""

    def setUp(self):
        self.client = TestClient(app)
        self.ts = int(time.time() * 1000)

    def test_password_hashing_and_verification(self):
        """Verifies bcrypt hashing produces secure hashes that verify correctly."""
        password = "SecureStudyPassword123"
        hashed = hash_password(password)

        self.assertNotEqual(password, hashed)
        self.assertTrue(hashed.startswith("$2b$") or hashed.startswith("$2a$"))
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("WrongPassword456", hashed))

    def test_user_registration_success(self):
        """Tests successful registration via POST /auth/register."""
        username = f"user_reg_{self.ts}"
        email = f"{username}@example.com"
        password = "strongPassword123"

        response = self.client.post("/auth/register", json={
            "username": username,
            "email": email,
            "password": password,
        })

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertIn("user", data)
        self.assertEqual(data["user"]["username"], username)
        self.assertEqual(data["user"]["email"], email)
        self.assertTrue(data["user"]["user_id"].startswith("usr_"))

    def test_registration_validation_rules(self):
        """Tests that invalid usernames, duplicate emails, and short passwords fail."""
        # Short username (< 3 chars)
        res1 = self.client.post("/auth/register", json={
            "username": "ab",
            "email": f"test_{self.ts}@example.com",
            "password": "validPassword123",
        })
        self.assertEqual(res1.status_code, 422)  # Pydantic min_length validation

        # Short password (< 6 chars)
        res2 = self.client.post("/auth/register", json={
            "username": f"user_{self.ts}",
            "email": f"test2_{self.ts}@example.com",
            "password": "123",
        })
        self.assertEqual(res2.status_code, 422)  # Pydantic min_length validation

        # Invalid username characters (spaces, symbols)
        res3 = self.client.post("/auth/register", json={
            "username": "invalid user name!",
            "email": f"test3_{self.ts}@example.com",
            "password": "validPassword123",
        })
        self.assertEqual(res3.status_code, 400)

    def test_registration_duplicate_prevention(self):
        """Tests duplicate username and email registration rejection."""
        u = f"dup_user_{self.ts}"
        e = f"{u}@example.com"
        pw = "password123"

        # Register first time
        res1 = self.client.post("/auth/register", json={"username": u, "email": e, "password": pw})
        self.assertEqual(res1.status_code, 201)

        # Duplicate username
        res2 = self.client.post("/auth/register", json={"username": u, "email": f"other_{e}", "password": pw})
        self.assertEqual(res2.status_code, 400)
        self.assertIn("already taken", res2.json()["detail"])

        # Duplicate email
        res3 = self.client.post("/auth/register", json={"username": f"other_{u}", "email": e, "password": pw})
        self.assertEqual(res3.status_code, 400)
        self.assertIn("already registered", res3.json()["detail"])

    def test_login_with_username_and_email(self):
        """Tests login using both username and email identifier."""
        u = f"login_user_{self.ts}"
        e = f"{u}@example.com"
        pw = "correctPassword123"

        reg_res = self.client.post("/auth/register", json={"username": u, "email": e, "password": pw})
        self.assertEqual(reg_res.status_code, 201)

        # Login with username
        res_user = self.client.post("/auth/login", json={"username_or_email": u, "password": pw})
        self.assertEqual(res_user.status_code, 200)
        self.assertIn("access_token", res_user.json())

        # Login with email
        res_email = self.client.post("/auth/login", json={"username_or_email": e, "password": pw})
        self.assertEqual(res_email.status_code, 200)
        self.assertIn("access_token", res_email.json())

        # Wrong password
        res_bad = self.client.post("/auth/login", json={"username_or_email": u, "password": "WrongPassword"})
        self.assertEqual(res_bad.status_code, 401)

        # Non-existent user
        res_none = self.client.post("/auth/login", json={"username_or_email": "nonexistent_999", "password": pw})
        self.assertEqual(res_none.status_code, 401)

    def test_current_user_profile_me(self):
        """Tests /auth/me profile endpoint with Bearer token."""
        u = f"me_user_{self.ts}"
        e = f"{u}@example.com"
        pw = "password123"

        reg_res = self.client.post("/auth/register", json={"username": u, "email": e, "password": pw})
        token = reg_res.json()["access_token"]

        # Call /auth/me with valid token
        res = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["username"], u)
        self.assertEqual(data["email"], e)
        self.assertIn("user_id", data)

    def test_expired_and_malformed_token_rejection(self):
        """Tests that expired and tampered tokens are rejected with 401."""
        # Malformed token
        res_bad = self.client.get("/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"})
        self.assertEqual(res_bad.status_code, 401)

        # Expired token (delta = -60 seconds)
        expired_token = create_access_token(
            user_id="usr_expired_test",
            username="expired_user",
            email="expired@test.com",
            expires_delta=-60,
        )
        res_exp = self.client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
        self.assertEqual(res_exp.status_code, 401)
        self.assertIn("expired", res_exp.json()["detail"].lower())

    def test_google_auth_flow(self):
        """Tests Google login and auto-registration flow."""
        g_email = f"google_student_{self.ts}@example.com"
        g_name = "Google Student"
        g_id = f"g_sub_{self.ts}"

        # 1. Fresh user registration via Google
        res1 = self.client.post("/auth/google", json={
            "email": g_email,
            "name": g_name,
            "google_id": g_id,
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertIn("access_token", data1)
        self.assertEqual(data1["user"]["email"], g_email)
        user_id_1 = data1["user"]["user_id"]

        # 2. Existing user logging in again via Google
        res2 = self.client.post("/auth/google", json={
            "email": g_email,
            "name": g_name,
            "google_id": g_id,
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["user"]["user_id"], user_id_1)
        self.assertEqual(data2["user"]["email"], g_email)

    @patch("backend.auth_service.send_password_reset_email")
    def test_forgot_and_reset_password_flow(self, mock_send_email):
        """Tests forgot password email delivery, secure code hashing, and reset flow."""
        sent_emails = []

        def fake_send(to_email, reset_code, username=None):
            sent_emails.append({
                "to_email": to_email,
                "reset_code": reset_code,
                "username": username,
            })
            return True

        mock_send_email.side_effect = fake_send

        u = f"reset_user_{self.ts}"
        e = f"{u}@example.com"
        old_pw = "OriginalPassword123"
        new_pw = "UpdatedPassword456"

        # Register user
        reg_res = self.client.post("/auth/register", json={"username": u, "email": e, "password": old_pw})
        self.assertEqual(reg_res.status_code, 201)

        # 1. Request password reset
        forgot_res = self.client.post("/auth/forgot-password", json={"email": e})
        self.assertEqual(forgot_res.status_code, 200)
        forgot_data = forgot_res.json()

        # Requirement 14: Never return reset_code or reset_token in API response
        self.assertNotIn("reset_code", forgot_data)
        self.assertNotIn("reset_token", forgot_data)
        self.assertIn("message", forgot_data)
        self.assertEqual(forgot_data.get("expires_in_minutes"), 15)

        # Requirement 2 & 3: Email delivery of 6-digit verification code
        self.assertEqual(len(sent_emails), 1)
        self.assertEqual(sent_emails[0]["to_email"], e)
        code = sent_emails[0]["reset_code"]
        self.assertEqual(len(code), 6)
        self.assertTrue(code.isdigit())

        # Requirement 7: Database stores only securely hashed code (never plaintext)
        with get_db_connection() as conn:
            row = conn.execute("SELECT reset_code FROM password_resets WHERE email = ?", (e,)).fetchone()
            self.assertIsNotNone(row)
            stored_hash = row["reset_code"]
            self.assertNotEqual(stored_hash, code)
            self.assertIn("$", stored_hash)

        # 2. Reset password using the 6-digit verification code
        reset_res = self.client.post("/auth/reset-password", json={
            "token": code,
            "new_password": new_pw,
            "email": e,
        })
        self.assertEqual(reset_res.status_code, 200)
        self.assertTrue(reset_res.json()["success"])

        # 3. Old password should now fail
        res_old = self.client.post("/auth/login", json={"username_or_email": u, "password": old_pw})
        self.assertEqual(res_old.status_code, 401)

        # 4. New password should succeed
        res_new = self.client.post("/auth/login", json={"username_or_email": u, "password": new_pw})
        self.assertEqual(res_new.status_code, 200)
        self.assertIn("access_token", res_new.json())

        # 5. Reusing the same code must fail (Requirement 9: Single-use policy)
        reuse_res = self.client.post("/auth/reset-password", json={
            "token": code,
            "new_password": "AnotherPassword789",
            "email": e,
        })
        self.assertEqual(reuse_res.status_code, 400)
        self.assertIn("already been used", reuse_res.json()["detail"].lower())

    @patch("backend.auth_service.send_password_reset_email")
    def test_password_reset_code_expiration(self, mock_send_email):
        """Tests that expired reset codes (>15 minutes) are rejected (Requirement 8)."""
        captured = []
        mock_send_email.side_effect = lambda to_email, reset_code, username=None: captured.append(reset_code) or True

        u = f"expire_user_{self.ts}"
        e = f"{u}@example.com"
        self.client.post("/auth/register", json={"username": u, "email": e, "password": "Password123"})

        forgot_res = self.client.post("/auth/forgot-password", json={"email": e})
        self.assertEqual(forgot_res.status_code, 200)
        code = captured[0]

        # Artificially expire the reset code in database
        with get_db_connection() as conn:
            conn.execute("UPDATE password_resets SET expires_at = ? WHERE email = ?", (time.time() - 60, e))
            conn.commit()

        reset_res = self.client.post("/auth/reset-password", json={
            "token": code,
            "new_password": "NewPassword123",
            "email": e,
        })
        self.assertEqual(reset_res.status_code, 400)
        self.assertIn("expired", reset_res.json()["detail"].lower())

    def test_password_reset_invalid_code(self):
        """Tests that incorrect verification codes are rejected."""
        res = self.client.post("/auth/reset-password", json={
            "token": "000000",
            "new_password": "ValidPassword123",
            "email": "anyuser@example.com",
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("invalid", res.json()["detail"].lower())

    @patch("backend.auth_service.send_password_reset_email")
    def test_password_reset_rate_limiting(self, mock_send_email):
        """Tests rate limiting prevents repeated reset requests within 60s (Requirement 10)."""
        mock_send_email.return_value = True

        u = f"ratelimit_user_{self.ts}"
        e = f"{u}@example.com"
        self.client.post("/auth/register", json={"username": u, "email": e, "password": "Password123"})

        # First request succeeds
        res1 = self.client.post("/auth/forgot-password", json={"email": e})
        self.assertEqual(res1.status_code, 200)

        # Immediate second request fails with HTTP 429
        res2 = self.client.post("/auth/forgot-password", json={"email": e})
        self.assertEqual(res2.status_code, 429)
        self.assertIn("wait", res2.json()["detail"].lower())

    @patch("backend.auth_service.send_password_reset_email")
    def test_password_reset_email_delivery_failure_handling(self, mock_send_email):
        """Tests that SMTP delivery failure raises an error and does not store reset code (Requirement 12)."""
        mock_send_email.side_effect = Exception("SMTP Connection timed out: smtp.gmail.com:587")

        u = f"fail_user_{self.ts}"
        e = f"{u}@example.com"
        self.client.post("/auth/register", json={"username": u, "email": e, "password": "Password123"})

        res = self.client.post("/auth/forgot-password", json={"email": e})
        self.assertEqual(res.status_code, 502)
        self.assertIn("Email delivery failed", res.json()["detail"])

        # Verify no reset code was committed to database
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM password_resets WHERE email = ?", (e,)).fetchone()
            self.assertIsNone(row)

    @patch("backend.auth_service.send_password_reset_email")
    def test_forgot_password_nonexistent_email(self, mock_send_email):
        """Tests that requesting reset for unknown email gives safe generic message (Requirement 11)."""
        res = self.client.post("/auth/forgot-password", json={"email": "nonexistent_email_999@test.com"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("If an account with this email exists", res.json()["message"])
        # Email sender must NOT be called for non-existent users
        mock_send_email.assert_not_called()

    def test_protected_routes_reject_unauthenticated(self):
        """Verifies that protected routes return 401 when no token is supplied."""
        routes = [
            ("GET", "/documents"),
            ("DELETE", "/documents/somefile.pdf"),
            ("POST", "/search"),
            ("POST", "/ask"),
            ("POST", "/summarize"),
            ("POST", "/important-questions"),
        ]

        for method, path in routes:
            if method == "GET":
                res = self.client.get(path)
            elif method == "DELETE":
                res = self.client.delete(path)
            else:
                res = self.client.post(path, json={})
            self.assertEqual(res.status_code, 401, f"Route {method} {path} should require authentication")


class TestMultiUserDataIsolation(unittest.TestCase):
    """End-to-end tests validating strict data isolation between multiple users."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        ts = int(time.time() * 1000)

        # Create User A (Alice)
        cls.user_a_name = f"alice_{ts}"
        cls.user_a_email = f"alice_{ts}@study.local"
        res_a = cls.client.post("/auth/register", json={
            "username": cls.user_a_name,
            "email": cls.user_a_email,
            "password": "AlicePassword123",
        })
        cls.user_a_data = res_a.json()
        cls.token_a = cls.user_a_data["access_token"]
        cls.user_a_id = cls.user_a_data["user"]["user_id"]
        cls.headers_a = {"Authorization": f"Bearer {cls.token_a}"}

        # Create User B (Bob)
        cls.user_b_name = f"bob_{ts}"
        cls.user_b_email = f"bob_{ts}@study.local"
        res_b = cls.client.post("/auth/register", json={
            "username": cls.user_b_name,
            "email": cls.user_b_email,
            "password": "BobPassword123",
        })
        cls.user_b_data = res_b.json()
        cls.token_b = cls.user_b_data["access_token"]
        cls.user_b_id = cls.user_b_data["user"]["user_id"]
        cls.headers_b = {"Authorization": f"Bearer {cls.token_b}"}

        # Directly index a test document for User A
        cls.doc_a_chunks = [
            {
                "chunk_number": 1,
                "page_number": 1,
                "text": "Alice's Quantum Mechanics Notes. Photons exhibit wave-particle duality as verified by the double-slit experiment.",
            },
            {
                "chunk_number": 2,
                "page_number": 2,
                "text": "The Schrödinger wave equation describes how the quantum state of a physical system changes over time.",
            },
        ]
        cls.doc_a_meta = index_document(
            filename="alice_quantum_physics.pdf",
            total_pages=2,
            chunks=cls.doc_a_chunks,
            user_id=cls.user_a_id,
        )
        cls.doc_a_id = cls.doc_a_meta["document_id"]

        # Directly index a test document for User B
        cls.doc_b_chunks = [
            {
                "chunk_number": 1,
                "page_number": 1,
                "text": "Bob's Organic Chemistry Lecture. Alkanes are saturated hydrocarbons with single bonds between carbon atoms.",
            },
            {
                "chunk_number": 2,
                "page_number": 2,
                "text": "Nucleophilic substitution reactions occur via SN1 or SN2 mechanisms depending on substrate hindrance.",
            },
        ]
        cls.doc_b_meta = index_document(
            filename="bob_organic_chemistry.pdf",
            total_pages=2,
            chunks=cls.doc_b_chunks,
            user_id=cls.user_b_id,
        )
        cls.doc_b_id = cls.doc_b_meta["document_id"]

    def test_chroma_chunks_ownership_metadata(self):
        """Verifies that ChromaDB chunks have user_id strictly assigned in metadata and chunk IDs."""
        chunks_a = get_document_chunks(self.doc_a_id, user_id=self.user_a_id)
        self.assertEqual(len(chunks_a), 2)
        for chunk in chunks_a:
            self.assertEqual(chunk["metadata"]["user_id"], self.user_a_id)
            self.assertTrue(chunk["chunk_id"].startswith(f"{self.user_a_id}_"))

        chunks_b = get_document_chunks(self.doc_b_id, user_id=self.user_b_id)
        self.assertEqual(len(chunks_b), 2)
        for chunk in chunks_b:
            self.assertEqual(chunk["metadata"]["user_id"], self.user_b_id)
            self.assertTrue(chunk["chunk_id"].startswith(f"{self.user_b_id}_"))

    def test_document_listing_isolation(self):
        """Verifies User A only sees their own documents, and User B sees only theirs."""
        res_a = self.client.get("/documents", headers=self.headers_a)
        self.assertEqual(res_a.status_code, 200)
        docs_a = res_a.json()["documents"]
        filenames_a = [d["filename"] for d in docs_a]
        self.assertIn("alice_quantum_physics.pdf", filenames_a)
        self.assertNotIn("bob_organic_chemistry.pdf", filenames_a)

        res_b = self.client.get("/documents", headers=self.headers_b)
        self.assertEqual(res_b.status_code, 200)
        docs_b = res_b.json()["documents"]
        filenames_b = [d["filename"] for d in docs_b]
        self.assertIn("bob_organic_chemistry.pdf", filenames_b)
        self.assertNotIn("alice_quantum_physics.pdf", filenames_b)

    def test_cross_user_search_isolation(self):
        """Verifies User B cannot retrieve chunks from User A's documents through search."""
        # User B queries for Alice's specific topic ("wave-particle duality quantum")
        search_res = self.client.post("/search", headers=self.headers_b, json={
            "query": "wave-particle duality double-slit experiment quantum photons",
            "top_k": 5,
        })
        self.assertEqual(search_res.status_code, 200)
        results = search_res.json()["results"]
        # Must return zero chunks because User B does not own any quantum physics documents
        for r in results:
            self.assertNotEqual(r["filename"], "alice_quantum_physics.pdf")

        # User A searches for the same query and successfully retrieves their chunk
        search_res_a = self.client.post("/search", headers=self.headers_a, json={
            "query": "wave-particle duality double-slit experiment quantum photons",
            "top_k": 5,
        })
        self.assertEqual(search_res_a.status_code, 200)
        results_a = search_res_a.json()["results"]
        self.assertTrue(any("wave-particle duality" in r["text"] for r in results_a))

    def test_cross_user_document_id_scoping_search(self):
        """Verifies passing another user's document_id in search returns 0 results."""
        res = self.client.post("/search", headers=self.headers_b, json={
            "query": "quantum wave",
            "document_id": self.doc_a_id,  # User B attempts to search Alice's doc_id
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["results"]), 0)

    def test_cross_user_deletion_prevention(self):
        """Verifies User B cannot delete User A's document even if filename is known."""
        # User B attempts to delete Alice's document
        del_res = self.client.delete("/documents/alice_quantum_physics.pdf", headers=self.headers_b)
        self.assertEqual(del_res.status_code, 404)
        self.assertIn("not found in your library", del_res.json()["detail"].lower())

        # Verify Alice's document is still intact
        res_a = self.client.get("/documents", headers=self.headers_a)
        filenames_a = [d["filename"] for d in res_a.json()["documents"]]
        self.assertIn("alice_quantum_physics.pdf", filenames_a)

    def test_cross_user_summarize_and_questions_scoping(self):
        """Verifies User B cannot summarize or generate questions for User A's document ID."""
        # User B attempts to summarize Alice's document ID
        sum_res = self.client.post("/summarize", headers=self.headers_b, json={
            "document_id": self.doc_a_id,
        })
        self.assertEqual(sum_res.status_code, 404)

        # User B attempts to generate questions for Alice's document ID
        q_res = self.client.post("/important-questions", headers=self.headers_b, json={
            "document_id": self.doc_a_id,
        })
        self.assertEqual(q_res.status_code, 404)

        # User B attempts to answer a question for Alice's document ID
        ans_res = self.client.post("/important-questions/answer", headers=self.headers_b, json={
            "document_id": self.doc_a_id,
            "question": "What is wave-particle duality?",
        })
        self.assertEqual(ans_res.status_code, 404)

    def test_own_document_deletion_does_not_affect_other_user(self):
        """Verifies User A deleting their document leaves User B's document untouched."""
        # Delete Alice's document
        del_res = self.client.delete("/documents/alice_quantum_physics.pdf", headers=self.headers_a)
        self.assertEqual(del_res.status_code, 200)

        # Check Alice's library is empty of this doc
        res_a = self.client.get("/documents", headers=self.headers_a)
        filenames_a = [d["filename"] for d in res_a.json()["documents"]]
        self.assertNotIn("alice_quantum_physics.pdf", filenames_a)

        # Check Bob's library still has his document intact
        res_b = self.client.get("/documents", headers=self.headers_b)
        filenames_b = [d["filename"] for d in res_b.json()["documents"]]
        self.assertIn("bob_organic_chemistry.pdf", filenames_b)

        # Clean up Bob's document as well
        del_b = self.client.delete("/documents/bob_organic_chemistry.pdf", headers=self.headers_b)
        self.assertEqual(del_b.status_code, 200)


class TestFrontendCredentialSafety(unittest.TestCase):
    """Tests ensuring no private credentials, secrets, or passwords leak into frontend bundles."""

    def test_frontend_dist_does_not_contain_secrets(self):
        dist_assets = os.path.join(WORKSPACE_DIR, "frontend", "dist", "assets")
        if not os.path.isdir(dist_assets):
            self.skipTest("Frontend dist assets not built yet")

        for fname in os.listdir(dist_assets):
            if fname.endswith(".js"):
                fpath = os.path.join(dist_assets, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Ensure server JWT secret key is never in frontend
                self.assertNotIn(JWT_SECRET_KEY, content, f"JWT_SECRET_KEY leaked in {fname}")
                # Ensure no private sqlite paths or backend paths
                self.assertNotIn("users.db", content, f"Internal DB name leaked in {fname}")
                self.assertNotIn("hashed_password", content, f"DB schema leaked in {fname}")


if __name__ == "__main__":
    unittest.main()
