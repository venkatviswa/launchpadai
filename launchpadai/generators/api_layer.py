"""Generate the API layer — FastAPI serving endpoints."""
from pathlib import Path


def generate_api_layer(config: dict, project_path: Path):
    base = project_path / "api"
    _write(base / "__init__.py", "")

    agent_import = {
        "plain": "from agents.base import agent",
        "langchain": "from agents.graph import agent",
        "llamaindex": "from agents.agent import agent",
        "crewai": "from agents.crew import run_crew",
    }.get(config["framework"], "from agents.base import agent")

    agent_call = "run_crew(request.message)" if config["framework"] == "crewai" else 'agent.run(request.message, session_id=request.session_id)'

    _write(base / "routes.py", f'''"""FastAPI routes — HTTP API for the agent."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

{agent_import}

app = FastAPI(title="{config['project_name']} API", version="0.1.0")

# CORS — allow the test UI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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

    _write(base / "schemas.py", '''"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
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
