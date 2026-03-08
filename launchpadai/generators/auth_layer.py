"""Generate the authentication layer for the test UI and API.

Provides three levels:
- simple: Single shared password from env var
- multi_user: Username + password pairs from env var
- oauth: OAuth/SSO via NextAuth.js (Next.js only)
"""
from pathlib import Path


def generate_auth_layer(config: dict, project_path: Path):
    """Generate authentication files based on config."""
    auth = config.get("auth", "none")
    if auth == "none":
        return

    base = project_path / "auth"
    base.mkdir(parents=True, exist_ok=True)
    _write(base / "__init__.py", "")

    # Core auth module (used by API and UIs)
    _generate_auth_core(config, base)

    # API middleware
    _generate_api_middleware(config, base)

    # Update .env.example with auth vars
    _append_auth_env(config, project_path)


def _generate_auth_core(config: dict, base: Path):
    """Generate the core authentication logic."""
    auth = config.get("auth", "none")

    _write(base / "provider.py", '''"""Authentication Provider — validates credentials.

Supports:
- Simple: Single shared password (APP_PASSWORD env var)
- Multi-user: Username:password pairs (APP_USERS env var)

For production, replace with a proper identity provider
(Auth0, Clerk, Supabase Auth, AWS Cognito, etc.)

Security notes:
- Uses hmac.compare_digest for constant-time comparison (timing attack resistant)
- Session tokens generated via secrets.token_urlsafe (cryptographically secure)
- Sessions expire after configurable TTL (default: 24 hours)
- For production: migrate sessions to Redis/DB, use bcrypt/argon2 for password hashing,
  enforce HTTPS, and consider a proper identity provider.
"""
import hmac
import os
import secrets
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class AuthProvider:
    """Authenticate users against configured credentials."""

    # Maximum failed auth attempts before temporary lockout
    MAX_FAILED_ATTEMPTS = 10
    LOCKOUT_SECONDS = 300  # 5 minutes

    def __init__(self):
        self.mode = os.getenv("AUTH_MODE", "''' + auth + '''")
        self._sessions: dict[str, dict] = {}  # token -> {user, expires}
        self._failed_attempts: dict[str, dict] = {}  # identifier -> {count, locked_until}

    def authenticate(self, username: str = None, password: str = None) -> dict:
        """Validate credentials and return result.

        Returns:
            {"authenticated": bool, "user": str, "token": str, "error": str}
        """
        # Rate limiting on failed attempts
        identifier = username or "simple"
        if self._is_locked_out(identifier):
            logger.warning(f"Auth lockout active for identifier: {identifier}")
            return {"authenticated": False, "error": "Too many failed attempts. Try again later."}

        if self.mode == "simple":
            result = self._auth_simple(password)
        elif self.mode == "multi_user":
            result = self._auth_multi_user(username, password)
        else:
            return {"authenticated": True, "user": "anonymous", "token": ""}

        if not result.get("authenticated"):
            self._record_failed_attempt(identifier)
        else:
            self._clear_failed_attempts(identifier)

        return result

    def validate_token(self, token: str) -> dict | None:
        """Validate a session token. Returns user info or None."""
        if not token or not isinstance(token, str):
            return None
        session = self._sessions.get(token)
        if not session:
            return None
        if time.time() > session["expires"]:
            del self._sessions[token]
            return None
        return {"user": session["user"]}

    def _auth_simple(self, password: str) -> dict:
        """Single password auth — good for demos and shared environments."""
        expected = os.getenv("APP_PASSWORD")
        if not expected:
            return {
                "authenticated": False,
                "error": "APP_PASSWORD not set in .env. Add it to enable auth.",
            }

        if hmac.compare_digest((password or "").encode(), expected.encode()):
            token = self._create_session("user")
            return {"authenticated": True, "user": "user", "token": token}
        return {"authenticated": False, "error": "Invalid password"}

    def _auth_multi_user(self, username: str, password: str) -> dict:
        """Multi-user auth — user:password pairs from env var.

        APP_USERS format: "alice:pass123,bob:pass456,charlie:pass789"
        """
        users_str = os.getenv("APP_USERS", "")
        if not users_str:
            return {
                "authenticated": False,
                "error": "APP_USERS not set in .env. Format: user1:pass1,user2:pass2",
            }

        users = {}
        for pair in users_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                u, p = pair.split(":", 1)
                users[u.strip()] = p.strip()

        # Always perform comparison even if user not found (prevent user enumeration)
        expected = users.get(username, "")
        dummy_check = hmac.compare_digest((password or "").encode(), expected.encode())

        if username in users and dummy_check:
            token = self._create_session(username)
            return {"authenticated": True, "user": username, "token": token}

        return {"authenticated": False, "error": "Invalid username or password"}

    def _create_session(self, user: str, ttl_hours: int = 24) -> str:
        """Create a session token."""
        token = secrets.token_urlsafe(32)
        self._sessions[token] = {
            "user": user,
            "expires": time.time() + ttl_hours * 3600,
        }
        return token

    def _is_locked_out(self, identifier: str) -> bool:
        """Check if an identifier is locked out due to failed attempts."""
        info = self._failed_attempts.get(identifier)
        if not info:
            return False
        if info.get("locked_until") and time.time() < info["locked_until"]:
            return True
        if info.get("locked_until") and time.time() >= info["locked_until"]:
            del self._failed_attempts[identifier]
            return False
        return False

    def _record_failed_attempt(self, identifier: str):
        """Record a failed authentication attempt."""
        info = self._failed_attempts.setdefault(identifier, {"count": 0})
        info["count"] += 1
        if info["count"] >= self.MAX_FAILED_ATTEMPTS:
            info["locked_until"] = time.time() + self.LOCKOUT_SECONDS
            logger.warning(f"Auth lockout triggered for: {identifier}")

    def _clear_failed_attempts(self, identifier: str):
        """Clear failed attempts after successful auth."""
        self._failed_attempts.pop(identifier, None)

    def logout(self, token: str):
        """Invalidate a session token."""
        self._sessions.pop(token, None)


# Singleton
auth_provider = AuthProvider()
''')


def _generate_api_middleware(config: dict, base: Path):
    """Generate FastAPI auth middleware."""

    _write(base / "middleware.py", '''"""FastAPI Authentication Middleware.

Protects API endpoints with token-based auth.
The /auth/login endpoint is always open.

Usage in api/routes.py:
    from auth.middleware import require_auth

    @app.post("/chat")
    async def chat(request: ChatRequest, user: dict = Depends(require_auth)):
        # user = {"user": "alice"}
        ...
"""
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth.provider import auth_provider

security = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Dependency that requires a valid auth token.

    Add to any endpoint: user = Depends(require_auth)
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    user = auth_provider.validate_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user


async def optional_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict | None:
    """Dependency that optionally validates auth.

    Returns user dict if authenticated, None otherwise.
    """
    if not credentials:
        return None
    return auth_provider.validate_token(credentials.credentials)
''')

    # Auth routes
    _write(base / "routes.py", '''"""Authentication API routes.

Provides login/logout endpoints for the test UI.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from auth.provider import auth_provider

router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: Optional[str] = None
    password: str


class LoginResponse(BaseModel):
    authenticated: bool
    user: Optional[str] = None
    token: Optional[str] = None
    error: Optional[str] = None


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate and receive a session token."""
    result = auth_provider.authenticate(
        username=request.username,
        password=request.password,
    )
    return LoginResponse(**result)


@router.post("/logout")
async def logout(token: str):
    """Invalidate a session token."""
    auth_provider.logout(token)
    return {"status": "logged out"}


@router.get("/check")
async def check_auth():
    """Check if authentication is enabled."""
    import os
    mode = os.getenv("AUTH_MODE", "none")
    return {
        "auth_enabled": mode != "none",
        "mode": mode,
        "needs_username": mode == "multi_user",
    }
''')


def _append_auth_env(config: dict, project_path: Path):
    """Add auth env vars to .env.example."""
    auth = config.get("auth", "none")
    env_file = project_path / ".env.example"

    auth_block = "\n# === Authentication ===\n"
    auth_block += f'AUTH_MODE={auth}\n'

    if auth == "simple":
        auth_block += '# Set a shared password for the test UI\n'
        auth_block += 'APP_PASSWORD=change-me-to-a-secure-password\n'
    elif auth == "multi_user":
        auth_block += '# Comma-separated username:password pairs\n'
        auth_block += 'APP_USERS=admin:change-me,demo:demo-password\n'
    elif auth == "oauth":
        auth_block += '# OAuth provider credentials\n'
        auth_block += 'NEXTAUTH_SECRET=generate-with-openssl-rand-base64-32\n'
        auth_block += 'NEXTAUTH_URL=http://localhost:3000\n'
        auth_block += '# Google OAuth (optional)\n'
        auth_block += 'GOOGLE_CLIENT_ID=your-client-id\n'
        auth_block += 'GOOGLE_CLIENT_SECRET=your-client-secret\n'
        auth_block += '# GitHub OAuth (optional)\n'
        auth_block += 'GITHUB_CLIENT_ID=your-client-id\n'
        auth_block += 'GITHUB_CLIENT_SECRET=your-client-secret\n'

    with open(env_file, "a") as f:
        f.write(auth_block)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
