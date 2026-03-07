"""Generate configuration files (.env, settings, etc.)."""
from pathlib import Path


ENV_VARS = {
    "openai": 'OPENAI_API_KEY=your-key-here',
    "anthropic": 'ANTHROPIC_API_KEY=your-key-here',
    "google": 'GOOGLE_API_KEY=your-key-here',
    "ollama": 'OLLAMA_BASE_URL=http://localhost:11434',
    "multiple": 'OPENAI_API_KEY=your-key-here\nANTHROPIC_API_KEY=your-key-here',
}

VECTORDB_ENV = {
    "chroma": '# ChromaDB — local by default, no key needed\nCHROMA_PERSIST_DIR=./data/chroma',
    "pinecone": 'PINECONE_API_KEY=your-key-here\nPINECONE_INDEX=your-index-name',
    "weaviate": 'WEAVIATE_URL=http://localhost:8080\nWEAVIATE_API_KEY=your-key-here',
    "qdrant": 'QDRANT_URL=http://localhost:6333\nQDRANT_API_KEY=your-key-here',
    "pgvector": 'POSTGRES_URL=postgresql://user:pass@localhost:5432/vectors',
}

EMBEDDING_ENV = {
    "openai-small": "",
    "openai-large": "",
    "cohere": "COHERE_API_KEY=your-key-here",
    "bge-m3": "# BGE-M3 runs locally — no API key needed",
    "gte-qwen2": "# GTE-Qwen2 runs locally — no API key needed",
    "nomic": "# Nomic runs locally — no API key needed",
    "ollama": "# Ollama embeddings — uses OLLAMA_BASE_URL above",
}

OBS_ENV = {
    "langfuse": 'LANGFUSE_PUBLIC_KEY=your-key-here\nLANGFUSE_SECRET_KEY=your-key-here\nLANGFUSE_HOST=https://cloud.langfuse.com',
    "langsmith": 'LANGSMITH_API_KEY=your-key-here\nLANGSMITH_PROJECT=your-project',
    "opentelemetry": 'OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317',
    "none": "",
}


def generate_config_files(config: dict, project_path: Path):
    """Generate .env.example, config/settings.py, .gitignore, README."""

    # .env.example
    env_lines = [
        f"# LaunchpadAI Project: {config['project_name']}",
        f"# Generated configuration\n",
        "# === LLM Provider ===",
        ENV_VARS.get(config["llm_provider"], ""),
        "",
        "# === Embedding Model ===",
        EMBEDDING_ENV.get(config["embedding_model"], ""),
        "",
        "# === Vector Database ===",
        VECTORDB_ENV.get(config["vector_db"], ""),
        "",
    ]

    if config["observability"] != "none":
        env_lines.extend([
            "# === Observability ===",
            OBS_ENV.get(config["observability"], ""),
            "",
        ])

    _write(project_path / ".env.example", "\n".join(env_lines))

    # config/settings.py
    settings = f'''"""Project settings — loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for the AI agent."""

    # Project
    PROJECT_NAME = "{config['project_name']}"
    AGENT_DESCRIPTION = "{config['agent_description']}"

    # LLM
    LLM_PROVIDER = "{config['llm_provider']}"
'''

    if config["llm_provider"] == "openai":
        settings += '''    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 0.7
'''
    elif config["llm_provider"] == "anthropic":
        settings += '''    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL = "claude-sonnet-4-20250514"
    LLM_TEMPERATURE = 0.7
'''
    elif config["llm_provider"] == "google":
        settings += '''    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLM_MODEL = "gemini-1.5-pro"
    LLM_TEMPERATURE = 0.7
'''
    elif config["llm_provider"] == "ollama":
        settings += '''    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL = "llama3.1"
    LLM_TEMPERATURE = 0.7
'''

    settings += f'''
    # Embedding
    EMBEDDING_PROVIDER = "{config['embedding_model']}"

    # Vector Database
    VECTOR_DB = "{config['vector_db']}"

    # RAG
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5

    # Observability
    OBSERVABILITY = "{config['observability']}"


settings = Settings()
'''

    _write(project_path / "config" / "__init__.py", "")
    _write(project_path / "config" / "settings.py", settings)

    # .gitignore
    gitignore = """# LaunchpadAI project
.env
__pycache__/
*.pyc
.pytest_cache/
data/chroma/
*.egg-info/
dist/
build/
.venv/
venv/
node_modules/
.next/
"""
    _write(project_path / ".gitignore", gitignore)

    # README.md (with architecture diagram)
    from launchpadai.generators.architecture_viz import (
        generate_architecture_diagram,
        generate_stack_summary,
        generate_request_flow_description,
    )

    mermaid_diagram = generate_architecture_diagram(config)
    stack_table = generate_stack_summary(config)
    request_flow = generate_request_flow_description(config)

    # Build dynamic project structure listing
    structure_lines = [
        f"{config['project_name']}/",
        "├── config/          # Settings, environment config",
        "├── models/          # LLM and embedding model abstractions",
        "│   ├── llm/         # Language model provider",
        "│   └── embeddings/  # Embedding model provider",
    ]
    if config.get("include_rag"):
        structure_lines.append("├── knowledge/       # RAG pipeline (ingestion, retrieval)")
    structure_lines.append("├── prompts/         # System prompts and templates")
    structure_lines.append("├── tools/           # External tool integrations")
    structure_lines.append("├── agents/          # Core agent logic and orchestration")
    if config.get("include_guardrails"):
        structure_lines.append("├── guardrails/      # Input/output safety filters")
    structure_lines.append("├── memory/          # Conversation and state management")
    structure_lines.append("├── api/             # FastAPI serving layer")
    if config.get("auth", "none") != "none":
        structure_lines.append("├── auth/            # Authentication layer")
    if config.get("ui", "none") != "none":
        structure_lines.append(f"├── ui/              # Test UI ({config['ui']})")
    if config.get("include_eval"):
        structure_lines.append("├── eval/            # Evaluation framework")
    if config.get("include_data_layer"):
        structure_lines.append("├── data/            # Raw, processed, and interim datasets")
        structure_lines.append("├── data_processing/ # Data loading and transformation")
    if config.get("include_notebooks"):
        structure_lines.append("├── notebooks/       # Jupyter notebooks for EDA")
    if config.get("include_ml_pipeline"):
        structure_lines.append("├── ml_models/       # Training, inference, and model registry")
    if config.get("observability", "none") != "none":
        structure_lines.append("├── observability/   # Tracing, monitoring, and cost tracking")
    structure_lines.append("├── scripts/         # Utility scripts")
    structure_lines.append("└── tests/           # Test suite")
    project_structure = "\n".join(structure_lines)

    readme = f"""# {config['project_name']}

{config['agent_description']}

> Generated with [LaunchpadAI](https://github.com/launchpadai/launchpad)

---

## Architecture Diagram

```mermaid
{mermaid_diagram}
```

> 💡 **View this diagram**: Paste the mermaid code into [mermaid.live](https://mermaid.live) or view directly on GitHub (GitHub renders Mermaid natively).

---

## Technology Stack

{stack_table}

---

## How It Works — Request Flow

{request_flow}

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys
"""

    if config.get("include_rag"):
        readme += """
# 3. Add documents and ingest (if using RAG)
#    Place files in data/documents/
launchpad ingest --source data/documents/
"""

    if config.get("include_ml_pipeline"):
        readme += """
# 4. Train ML model (if using ML pipeline)
python ml_models/training/train.py
"""

    readme += """
# Run the agent
launchpad run
```

---

## Project Structure

```
""" + project_structure + """
```

---

## Commands

| Command | Description |
|---------|-------------|
| `launchpad run` | Start the agent with test UI |
| `launchpad ingest` | Ingest documents into vector store |
| `launchpad evaluate` | Run evaluation suite |

---

*Generated by LaunchpadAI — the architecture-first scaffolding tool for AI engineering*
"""
    _write(project_path / "README.md", readme)

    # Also save standalone mermaid file
    _write(project_path / "docs" / "architecture.mermaid", mermaid_diagram)
    _write(project_path / "docs" / "architecture.md", f"""# Architecture — {config['project_name']}

## System Diagram

```mermaid
{mermaid_diagram}
```

## Technology Stack

{stack_table}

## Request Flow

{request_flow}
""")


def _write(path: Path, content: str):
    """Write content to file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
