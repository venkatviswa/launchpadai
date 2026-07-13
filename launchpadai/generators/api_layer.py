"""Generate the API layer — FastAPI serving endpoints."""
from pathlib import Path


def generate_api_layer(config, project_path: Path):
    base = project_path / "api"
    _write(base / "__init__.py", "")

    # Every framework adapter exposes the same entrypoint (agents/__init__.py)
    agent_import = "from agents import agent"
    agent_call = 'agent.run(request.message, session_id=request.session_id)'

    # Token auth wires into the API for the env-var-backed modes. OAuth is
    # handled by the Next.js frontend (NextAuth), not the API layer.
    auth_enabled = config.get("auth", "none") in ("simple", "multi_user")
    auth_imports = ""
    auth_router = ""
    chat_auth_param = ""
    if auth_enabled:
        auth_imports = (
            "from auth.middleware import require_auth\n"
            "from auth.routes import router as auth_router\n"
        )
        auth_router = "\n# Login/logout endpoints (/auth/login is always open)\napp.include_router(auth_router)\n"
        chat_auth_param = ", user: dict = Depends(require_auth)"

    fastapi_imports = "Depends, FastAPI, Request" if auth_enabled else "FastAPI, Request"

    _write(base / "routes.py", f'''"""FastAPI routes — HTTP API for the agent."""
import os
from fastapi import {fastapi_imports}
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChatResponse
{agent_import}
{auth_imports}
app = FastAPI(title="{config['project_name']} API", version="0.1.0")

# CORS — restrict origins for security
# Configure ALLOWED_ORIGINS in .env (comma-separated), defaults to localhost only
_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

{auth_router}
# --- Security headers middleware ---
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest{chat_auth_param}):
    """Send a message to the agent."""
    result = {agent_call}
    return ChatResponse(
        response=result["response"] if isinstance(result, dict) else str(result),
        session_id=request.session_id or "default",
    )


@app.get("/health")
async def health():
    return {{"status": "ok", "project": "{config['project_name']}"}}
''')

    _write(base / "schemas.py", '''"""Pydantic schemas for API request/response models.

Single source of truth — api/routes.py imports these; keep validation
constraints here so every consumer gets them.
"""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[str] = Field("default", max_length=128)
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list[str] = []
    metadata: Optional[dict] = None
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
