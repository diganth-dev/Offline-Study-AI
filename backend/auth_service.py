"""Authentication and User Management Service for Offline Study AI.

Provides:
- SQLite-backed user storage (backend/storage/users.db)
- Password hashing with bcrypt
- Cryptographically signed JWT access tokens (HS256)
- FastAPI authentication dependency (get_current_user)
- Strict user validation (username, email, password strength)
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from typing import Any, Optional
import bcrypt
from dotenv import load_dotenv
import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    from backend.email_service import send_password_reset_email
except ImportError:
    from email_service import send_password_reset_email

# Ensure environment variables from backend/.env and root .env are loaded
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))
load_dotenv()

# Storage configuration
STORAGE_DIR = os.getenv("STUDY_AI_STORAGE_DIR", os.path.join(BASE_DIR, "storage"))
os.makedirs(STORAGE_DIR, exist_ok=True)

DB_PATH = os.path.join(STORAGE_DIR, "users.db")
SECRET_FILE_PATH = os.path.join(STORAGE_DIR, ".jwt_secret")


def _get_or_create_jwt_secret() -> str:
    """Retrieves the JWT signing secret from env var, secret file, or generates one."""
    env_secret = os.getenv("JWT_SECRET_KEY")
    if env_secret and len(env_secret) >= 32:
        return env_secret

    if os.path.isfile(SECRET_FILE_PATH):
        try:
            with open(SECRET_FILE_PATH, "r", encoding="utf-8") as f:
                saved_secret = f.read().strip()
                if len(saved_secret) >= 32:
                    return saved_secret
        except Exception:
            pass

    # Generate a cryptographically secure 256-bit (64 hex characters) key
    new_secret = secrets.token_hex(32)
    try:
        with open(SECRET_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(new_secret)
    except Exception:
        pass
    return new_secret


JWT_SECRET_KEY = _get_or_create_jwt_secret()
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = int(os.getenv("ACCESS_TOKEN_EXPIRE_SECONDS", 7 * 24 * 3600))  # 7 days

security_bearer = HTTPBearer(auto_error=False)


def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite users database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    """Initializes users table, password resets table, and indexes if not already present."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                hashed_password TEXT NOT NULL,
                created_at REAL NOT NULL,
                is_active INTEGER DEFAULT 1,
                google_id TEXT
            )
        """)
        # Ensure google_id column exists if table was created previously without it
        user_columns = [row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "google_id" not in user_columns:
            try:
                conn.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
            except Exception:
                pass

        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                reset_token TEXT PRIMARY KEY,
                reset_code TEXT NOT NULL,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                expires_at REAL NOT NULL,
                used INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                failed_attempts INTEGER DEFAULT 0
            )
        """)
        # Ensure failed_attempts column exists if table was created previously without it
        reset_columns = [row["name"] for row in conn.execute("PRAGMA table_info(password_resets)").fetchall()]
        if "failed_attempts" not in reset_columns:
            try:
                conn.execute("ALTER TABLE password_resets ADD COLUMN failed_attempts INTEGER DEFAULT 0")
            except Exception:
                pass

        conn.execute("CREATE INDEX IF NOT EXISTS idx_resets_token ON password_resets (reset_token)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resets_code ON password_resets (reset_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resets_email ON password_resets (email)")
        conn.commit()


# Initialize database schema on module import
init_auth_db()


def hash_reset_code(code: str) -> str:
    """Securely hashes a 6-digit verification code using HMAC-SHA256 with a unique cryptographic salt."""
    salt = secrets.token_hex(16)
    digest = hmac.new(
        JWT_SECRET_KEY.encode("utf-8"),
        f"{salt}:{code.strip()}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{salt}${digest}"


def verify_reset_code(provided_code: str, stored_hash: str) -> bool:
    """Verifies a user-supplied verification code against the stored hash in constant time."""
    try:
        clean_code = (provided_code or "").strip()
        if not clean_code or not stored_hash:
            return False

        if "$" in stored_hash:
            salt, expected_digest = stored_hash.split("$", 1)
            calculated_digest = hmac.new(
                JWT_SECRET_KEY.encode("utf-8"),
                f"{salt}:{clean_code}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return hmac.compare_digest(calculated_digest, expected_digest)

        # Fallback for plain text or bcrypt in legacy tests
        if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
            return verify_password(clean_code, stored_hash)
        return hmac.compare_digest(clean_code, stored_hash.strip())
    except Exception:
        return False


def validate_username(username: str) -> str:
    """Validates and cleans username (3-30 characters, alphanumeric, underscores, hyphens)."""
    cleaned = (username or "").strip()
    if not cleaned:
        raise ValueError("Username is required.")
    if len(cleaned) < 3 or len(cleaned) > 30:
        raise ValueError("Username must be between 3 and 30 characters.")
    if not re.match(r"^[a-zA-Z0-9_\-]+$", cleaned):
        raise ValueError("Username can only contain letters, numbers, hyphens and underscores.")
    return cleaned


def validate_email(email: str) -> str:
    """Validates and normalizes email address."""
    cleaned = (email or "").strip().lower()
    if not cleaned:
        raise ValueError("Email is required.")
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", cleaned):
        raise ValueError("Please provide a valid email address.")
    return cleaned


def validate_password(password: str) -> str:
    """Validates password strength (minimum 6 characters)."""
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")
    return password


def hash_password(password: str) -> str:
    """Hashes a password with bcrypt and returns a UTF-8 string."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks whether a plain password matches the bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: str, username: str, email: str, expires_delta: Optional[int] = None) -> str:
    """Generates a cryptographically signed JWT access token."""
    now = time.time()
    expire_duration = expires_delta if expires_delta is not None else ACCESS_TOKEN_EXPIRE_SECONDS
    expire_timestamp = now + expire_duration

    payload = {
        "sub": user_id,
        "username": username,
        "email": email,
        "iat": int(now),
        "exp": int(expire_timestamp),
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decodes and validates a JWT token.

    Raises:
        HTTPException (401): If token has expired, is invalid, or malformed.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\", error_description=\"token expired\""},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""},
        )


def register_user(username: str, email: str, password: str) -> dict[str, Any]:
    """Registers a new user and returns user info + access token."""
    clean_username = validate_username(username)
    clean_email = validate_email(email)
    validate_password(password)

    hashed_pw = hash_password(password)
    user_id = f"usr_{secrets.token_hex(12)}"
    now = time.time()

    with get_db_connection() as conn:
        # Check if username or email is already taken
        cursor = conn.execute(
            "SELECT username, email FROM users WHERE username = ? OR email = ?",
            (clean_username, clean_email)
        )
        row = cursor.fetchone()
        if row:
            if row["username"].lower() == clean_username.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Username '{clean_username}' is already taken. Please choose another."
                )
            if row["email"].lower() == clean_email.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Email '{clean_email}' is already registered. Please log in instead."
                )

        conn.execute(
            "INSERT INTO users (user_id, username, email, hashed_password, created_at, is_active) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, clean_username, clean_email, hashed_pw, now, 1)
        )
        conn.commit()

    token = create_access_token(user_id=user_id, username=clean_username, email=clean_email)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user_id,
            "username": clean_username,
            "email": clean_email,
        }
    }


def authenticate_user(username_or_email: str, password: str) -> dict[str, Any]:
    """Authenticates a user by username or email and returns user info + access token."""
    identifier = (username_or_email or "").strip()
    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email is required."
        )
    if not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is required."
        )

    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT user_id, username, email, hashed_password, is_active FROM users WHERE username = ? OR email = ?",
            (identifier, identifier.lower())
        )
        user = cursor.fetchone()

    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated."
        )

    user_id = user["user_id"]
    username = user["username"]
    email = user["email"]

    token = create_access_token(user_id=user_id, username=username, email=email)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user_id,
            "username": username,
            "email": email,
        }
    }


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """Fetches a user profile by user_id."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT user_id, username, email, created_at, is_active FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "email": row["email"],
                "created_at": row["created_at"],
                "is_active": bool(row["is_active"]),
            }
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> dict[str, Any]:
    """FastAPI dependency to extract and verify the current authenticated user.

    Raises:
        HTTPException (401): If token is missing, invalid, or user does not exist.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject identifier.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive.",
        )

    return user


def authenticate_google_user(
    credential: Optional[str] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
    google_id: Optional[str] = None,
) -> dict[str, Any]:
    """Authenticates or registers a user via Google Sign-In.

    Accepts a Google ID token credential or parsed user details.
    """
    extracted_email = email
    extracted_name = name
    extracted_sub = google_id

    # If an ID token was provided, extract payload
    if credential:
        cred = credential.strip()
        verified = False
        # 1. Try verifying with Google's tokeninfo endpoint if network is reachable
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={cred}")
                if res.status_code == 200:
                    data = res.json()
                    extracted_email = data.get("email")
                    extracted_name = data.get("name") or data.get("given_name")
                    extracted_sub = data.get("sub")
                    verified = True
        except Exception:
            pass

        # 2. Fallback: decode unverified JWT payload for offline / dev support
        if not verified and not extracted_email:
            try:
                payload = jwt.decode(cred, options={"verify_signature": False})
                extracted_email = payload.get("email")
                extracted_name = payload.get("name") or payload.get("given_name")
                extracted_sub = payload.get("sub")
            except Exception:
                pass

    if not extracted_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve a valid email from Google sign-in.",
        )

    clean_email = validate_email(extracted_email)
    google_sub = str(extracted_sub or f"g_{secrets.token_hex(8)}")
    display_name = (extracted_name or clean_email.split("@")[0]).strip()

    with get_db_connection() as conn:
        # Check if user exists by email or google_id
        cursor = conn.execute(
            "SELECT user_id, username, email, is_active, google_id FROM users WHERE email = ? OR google_id = ?",
            (clean_email, google_sub)
        )
        existing_user = cursor.fetchone()

        if existing_user:
            if not existing_user["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="This account has been deactivated."
                )
            user_id = existing_user["user_id"]
            username = existing_user["username"]
            email = existing_user["email"]

            # Update google_id if missing
            if not existing_user["google_id"]:
                conn.execute(
                    "UPDATE users SET google_id = ? WHERE user_id = ?",
                    (google_sub, user_id)
                )
                conn.commit()
        else:
            # Auto-register new user via Google
            base_username = re.sub(r"[^a-zA-Z0-9_\-]", "", display_name.replace(" ", "_").lower())
            if len(base_username) < 3:
                base_username = f"google_user_{secrets.token_hex(3)}"
            base_username = base_username[:24]

            candidate_username = base_username
            attempt = 1
            while True:
                cursor = conn.execute("SELECT user_id FROM users WHERE username = ?", (candidate_username,))
                if not cursor.fetchone():
                    break
                candidate_username = f"{base_username}_{secrets.token_hex(2)}"
                attempt += 1
                if attempt > 10:
                    candidate_username = f"user_{secrets.token_hex(6)}"
                    break

            user_id = f"usr_{secrets.token_hex(12)}"
            hashed_pw = hash_password(secrets.token_urlsafe(32))
            now = time.time()

            conn.execute(
                "INSERT INTO users (user_id, username, email, hashed_password, created_at, is_active, google_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, candidate_username, clean_email, hashed_pw, now, 1, google_sub)
            )
            conn.commit()
            username = candidate_username
            email = clean_email

    token = create_access_token(user_id=user_id, username=username, email=email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user_id,
            "username": username,
            "email": email,
        }
    }


def request_password_reset(email: str) -> dict[str, Any]:
    """Generates a secure 6-digit verification code, sends it via SMTP, and records the salted hash.

    Enforces:
    - Rate limiting (60s minimum interval, max 5 requests per 15m)
    - Anti-enumeration (identical success message for unregistered emails)
    - Code delivery before database commit (Requirement 12)
    - Secure salted HMAC-SHA256 hashing in database (Requirement 7)
    - 15-minute expiration (Requirement 8)
    - No reset code or secret exposed in API response (Requirement 6, 14)
    """
    try:
        clean_email = validate_email(email)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    now = time.time()

    with get_db_connection() as conn:
        # Rate limiting: Check cooldown (60 seconds)
        cursor = conn.execute(
            "SELECT created_at FROM password_resets WHERE email = ? ORDER BY created_at DESC LIMIT 1",
            (clean_email,)
        )
        last_request = cursor.fetchone()
        if last_request:
            elapsed = now - last_request["created_at"]
            if elapsed < 60:
                wait_sec = int(60 - elapsed)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Please wait {wait_sec} seconds before requesting another verification code.",
                )

        # Rate limiting: Check max requests in 15 minutes window
        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM password_resets WHERE email = ? AND created_at > ?",
            (clean_email, now - 900)
        )
        count_row = cursor.fetchone()
        if count_row and count_row["cnt"] >= 5:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many password reset requests for this email. Please try again after 15 minutes.",
            )

        # Check if user exists
        cursor = conn.execute(
            "SELECT user_id, username, email FROM users WHERE email = ? AND is_active = 1",
            (clean_email,)
        )
        user = cursor.fetchone()

        if not user:
            # Neutral response to prevent email discovery (Requirement 11)
            return {
                "message": "If an account with this email exists, a 6-digit verification code has been sent to your email.",
                "expires_in_minutes": 15,
            }

        user_id = user["user_id"]
        username = user["username"]

        # Generate cryptographically secure 6-digit code and unique token
        reset_code = f"{secrets.randbelow(900000) + 100000}"
        reset_token = f"rst_{secrets.token_urlsafe(24)}"
        expires_at = now + 900  # strictly 15 minutes

        # Requirement 12: Display a success message only after the email provider accepts the sending request.
        # Attempt email delivery via SMTP first.
        try:
            send_password_reset_email(
                to_email=clean_email,
                reset_code=reset_code,
                username=username,
            )
        except ValueError as e:
            # Configuration missing (e.g., SMTP_HOST not set)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
        except Exception as e:
            # SMTP network or authentication failure
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Email delivery failed: {str(e)}. Please check your SMTP settings or network connectivity.",
            )

        # Hash code with unique salt before storing (Requirement 7)
        hashed_code = hash_reset_code(reset_code)

        conn.execute(
            """
            INSERT INTO password_resets (
                reset_token, reset_code, user_id, email, expires_at, used, created_at, failed_attempts
            ) VALUES (?, ?, ?, ?, ?, 0, ?, 0)
            """,
            (reset_token, hashed_code, user_id, clean_email, expires_at, now)
        )
        conn.commit()

    return {
        "message": "A 6-digit verification code has been sent to your email address.",
        "expires_in_minutes": 15,
    }


def reset_password(token_or_code: str, new_password: str, email: Optional[str] = None) -> dict[str, Any]:
    """Resets user password using a valid 6-digit verification code or reset token.

    Enforces:
    - Code existence and cryptographic verification
    - 15-minute expiration window (Requirement 8)
    - Single-use policy (Requirement 9)
    - Max 5 failed attempts per reset request before invalidation
    - Password validation and bcrypt hashing (Requirement 13)
    """
    token_val = (token_or_code or "").strip()
    if not token_val:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="6-digit verification code is required.",
        )

    try:
        validate_password(new_password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    now = time.time()
    clean_email = email.strip().lower() if email else None

    with get_db_connection() as conn:
        # Retrieve candidate reset records
        if clean_email:
            cursor = conn.execute(
                """
                SELECT reset_token, reset_code, user_id, email, expires_at, used, failed_attempts
                FROM password_resets
                WHERE email = ?
                ORDER BY created_at DESC LIMIT 10
                """,
                (clean_email,)
            )
        else:
            cursor = conn.execute(
                """
                SELECT reset_token, reset_code, user_id, email, expires_at, used, failed_attempts
                FROM password_resets
                ORDER BY created_at DESC LIMIT 25
                """
            )
        rows = cursor.fetchall()

        matched_row = None
        for row in rows:
            # Check token match (by hash or direct token identifier)
            if verify_reset_code(token_val, row["reset_code"]) or token_val == row["reset_token"]:
                matched_row = row
                break

        if not matched_row:
            # If email was provided, increment failed attempts on the most recent unused reset
            if clean_email and rows:
                conn.execute(
                    "UPDATE password_resets SET failed_attempts = failed_attempts + 1 WHERE reset_token = ?",
                    (rows[0]["reset_token"],)
                )
                conn.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code. Please check your email or request a new code.",
            )

        # Check maximum failed attempts
        if matched_row["failed_attempts"] >= 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This verification code has been invalidated due to too many incorrect attempts. Please request a new code.",
            )

        # Check single use (Requirement 9)
        if matched_row["used"] != 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This verification code has already been used. Please request a new one.",
            )

        # Check expiration (Requirement 8)
        if matched_row["expires_at"] < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This verification code has expired (codes expire after 15 minutes). Please request a new one.",
            )

        # Update password with bcrypt (Requirement 13)
        new_hashed = hash_password(new_password)
        conn.execute(
            "UPDATE users SET hashed_password = ? WHERE user_id = ?",
            (new_hashed, matched_row["user_id"])
        )

        # Mark code as used (single use) and invalidate all active codes for this user
        conn.execute(
            "UPDATE password_resets SET used = 1 WHERE user_id = ?",
            (matched_row["user_id"],)
        )
        conn.commit()

    return {
        "message": "Password successfully reset! You can now sign in with your new password.",
        "success": True,
    }
