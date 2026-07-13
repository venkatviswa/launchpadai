# LaunchpadAI

[![CI](https://github.com/venkatviswa/launchpadai/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatviswa/launchpadai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

LaunchpadAI is an interactive CLI that scaffolds runnable, multi-agent AI projects in seconds. Answer a few questions (or pass a few flags) and get a working, well-architected, production-oriented foundation — ready for customization and hardening.

**Quick links:** [Quick Start](#quick-start) · [CLI Commands](#cli-commands) · [Multi-Agent Teams](#multi-agent-teams) · [Configuration Options](#configuration-options) · [Project Structure](#generated-project-structure) · [Development](#development)

---

## Why LaunchpadAI?

Building an AI agent today means making dozens of decisions before writing a single line of business logic: which framework, which LLM provider, one agent or a team, how to structure RAG, tool calling, memory, guardrails, evaluation, observability, auth, and deployment.

**LaunchpadAI eliminates this boilerplate.** It generates a complete, opinionated project with clean separation of concerns and a uniform agent interface — every layer independently configurable.

### What You Get

- **Multi-agent by design** — define a team of agents (each with its own prompt and tools) and pick an orchestration mode: single, sequential pipeline, or supervisor routing
- A **framework-adapter architecture** — the same project layout whether you pick plain Python, LangGraph, or CrewAI; only the orchestration files differ
- A **uniform entrypoint** — `from agents import agent; agent.run(msg, session_id)` works identically across every framework, with guardrails and tracing wired at ingress/egress
- **Tests that pass on day one** — every project ships a pytest suite that runs fully offline against a built-in mock LLM provider (`LLM_MOCK=1`), so `pytest tests` works before you add a single API key
- **Auto-generated architecture diagrams** (Mermaid) showing your request flow
- **Evaluation framework** with test cases to measure agent quality from day one
- **Docker setup** for one-command deployment

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/venkatviswa/launchpadai.git
cd launchpadai

# With uv (recommended)
uv sync && uv run launchpad --help

# Or with pip
pip install -e .
```

### Create Your First Project

```bash
# Interactive wizard
launchpad init my-agent

# Fully non-interactive (defaults + flags) — great for scripts and AI agents
launchpad init my-agent --defaults

# A multi-agent research pipeline on LangGraph, no wizard
launchpad init research-crew \
  --framework langgraph \
  --agent "researcher:Research Analyst:Find accurate information" \
  --agent "writer:Response Writer:Write clear, grounded answers" \
  --orchestration sequential \
  --llm anthropic --ui streamlit
```

### Run Your Agent

```bash
cd my-agent
pip install -r requirements.txt -r requirements-dev.txt
pytest tests            # runs offline against the mock LLM — no keys needed
cp .env.example .env    # add your API keys (or set LLM_MOCK=1 to stay offline)
launchpad run
```

---

## CLI Commands

### `launchpad init [project_name]`

Creates a new agentic AI project. Run with no flags for the interactive wizard, or pass `--defaults` and/or any option flags for a fully non-interactive run (any flag implies non-interactive mode).

Key flags (see `launchpad init --help` for all):

| Flag | Values |
|------|--------|
| `--framework, -f` | `plain`, `langgraph`, `crewai`, `agentscript` |
| `--llm` | `openai`, `anthropic`, `ollama` |
| `--agent` | Repeatable `"name:role:goal"` spec for multi-agent teams |
| `--orchestration` | `single`, `sequential`, `supervisor` |
| `--retrieval` | `custom`, `llamaindex` |
| `--rag/--no-rag`, `--guardrails/--no-guardrails`, `--eval/--no-eval`, `--mcp/--no-mcp` | feature toggles |
| `--observability` | `langfuse`, `langsmith`, `opentelemetry`, `none` |
| `--ui` | `streamlit`, `gradio`, `nextjs`, `none` |
| `--defaults, -y` | Use defaults for anything not passed |
| `--force` | Overwrite an existing directory |

After generation, a `launchpad.yaml` file saves your configuration for reproducibility.

### `launchpad run [project_dir]`

Runs the agent locally with the configured UI (Streamlit, Gradio, Next.js, or CLI).

### `launchpad ingest [project_dir]`

Ingests documents into the retrieval layer for RAG-enabled projects (`--source` to point at a directory).

### `launchpad evaluate [project_dir]`

Runs the evaluation suite (`eval/run_eval.py`) against the test cases in `eval/datasets/test_cases.yaml`.

---

## Multi-Agent Teams

Every project is built from **agent slices**. Each agent you define gets its own vertical slice:

```
agents/
├── __init__.py          # Uniform entrypoint: agent.run(...) / agent.reset(...)
├── researcher/
│   ├── prompts/system.md   # This agent's system prompt (versioned artifact)
│   └── tools.py            # Tools only this agent can use
├── writer/
│   ├── prompts/system.md
│   └── tools.py
└── ...                  # Framework orchestration files (see below)
```

Orchestration modes:

| Mode | Behavior |
|------|----------|
| `single` | One agent handles every request |
| `sequential` | Agents run in order; each receives the original request plus the previous agent's output |
| `supervisor` | A routing step picks the best agent for each request |

How each framework implements them:

| Framework | Files | single | sequential | supervisor |
|-----------|-------|--------|------------|------------|
| `plain` | `agents/base.py`, `agents/pipeline.py` | ✅ | ✅ pipeline loop | ✅ LLM router |
| `langgraph` | `agents/graph.py` | ✅ | ✅ chained nodes | ✅ conditional edges |
| `crewai` | `agents/crew.py` | ✅ | ✅ `Process.sequential` | ✅ `Process.hierarchical` |
| `agentscript` | `agents/client.py` + DX bundle | ✅ | — (topics live in Salesforce) | — |

Guardrails and observability tracing are applied in the shared entrypoint (`agents/__init__.py`), so ingress/egress behavior is identical regardless of framework.

---

## Configuration Options

### Agent Framework

| Framework | Tier | Description |
|-----------|------|-------------|
| **Plain Python** | 1 | No framework — full control over the agent loop |
| **LangGraph** | 1 | Graph-based orchestration with checkpointed state |
| **CrewAI** | 1 | Role-based multi-agent crews |
| **Salesforce AgentScript** | 2 | Agentforce DX — declarative agents deployed to Salesforce |

Tier 1 adapters are fully supported and smoke-tested in CI (render → install → import). Tier 2 adapters are maintained with a reduced feature surface.

### LLM Provider

| Provider | Default model | Notes |
|----------|--------------|-------|
| **Anthropic** | `claude-opus-4-8` | Any Claude model via the `LLM_MODEL` env var |
| **OpenAI** | `gpt-4o` | Any OpenAI chat model via `LLM_MODEL` |
| **Ollama** | `llama3.1` | Local models (Llama, Mistral, etc.) |

### Retrieval Layer (RAG)

| Option | Description |
|--------|-------------|
| **Custom pipeline** | Generated chunkers, loaders, and vector-store client — full control |
| **LlamaIndex** | Managed loading, chunking, and indexing behind the same retriever interface |

The retrieval layer pairs with **any** framework — LlamaIndex is a retrieval option here, not an orchestration framework.

### Embedding Models

| Model | Type |
|-------|------|
| OpenAI text-embedding-3-small / -large | Cloud (API) |
| BGE-M3, GTE-Qwen2 | Local (HuggingFace) |
| Nomic embed-text-v1.5 | Local |

### Vector Databases

| Database | Type |
|----------|------|
| **ChromaDB** | Local, great for development |
| **Pinecone** | Managed cloud |

### UI, Auth, Observability, Data & ML

| Dimension | Choices |
|-----------|---------|
| **UI** | Streamlit, Gradio, Next.js, CLI only |
| **Auth** | none, simple password, multi-user, OAuth/SSO (Next.js) |
| **Observability** | LangFuse, LangSmith, OpenTelemetry, none |
| **Data format** | CSV/TSV, Parquet, JSON/JSONL, SQL |
| **ML framework** | scikit-learn, PyTorch, XGBoost, HuggingFace Transformers |
| **Extras** | RAG, guardrails, evaluation, MCP tools, notebooks, Docker |

---

## Generated Project Structure

```
my-ai-agent/
├── agents/              # Agent slices + framework orchestration + entrypoint
├── api/                 # FastAPI routes & Pydantic schemas
├── auth/                # Authentication middleware (if enabled)
├── config/              # settings.py, application constants
├── data/                # Documents, datasets
├── docs/                # Architecture docs + Mermaid diagram
├── eval/                # Evaluation framework (if enabled)
├── guardrails/          # Input/output safety filters (if enabled)
├── knowledge/           # Retrieval layer (custom or LlamaIndex)
├── memory/              # Conversation history management
├── ml_models/           # ML training/inference (if enabled)
├── models/              # LLM & embedding provider abstractions
├── notebooks/           # Jupyter notebooks (if enabled)
├── observability/       # Tracing + cost tracking (if enabled)
├── prompts/             # Project-level prompts and few-shot examples
├── scripts/             # Utility scripts (ingestion, etc.)
├── tools/               # Shared tool registry and MCP servers
├── ui/                  # Chat interface (if enabled)
├── .env.example         # Environment variable template
├── docker-compose.yml   # Container orchestration (if enabled)
├── Dockerfile
├── launchpad.yaml       # Project configuration (reproducible)
└── requirements.txt
```

Only the layers you enable are generated. **AgentScript projects** additionally generate `force-app/main/aiAuthoringBundles/` (`.agent` DSL) and `sfdx-project.json`.

---

## How a Request Flows

```
UI / API  →  auth  →  agents/__init__.py (uniform entrypoint)
              →  input guardrails  →  orchestrator (framework + mode)
              →  per-agent loop: RAG context + LLM calls + tools
              →  output guardrails  →  tracing  →  response
```

Every generated project follows this shape regardless of framework — only the orchestrator differs.

---

## Development

### Repository Architecture

The CLI is built around a small plugin architecture:

- `launchpadai/config.py` — `ProjectConfig` / `AgentSpec` (Pydantic v2): option values, the project-name slug, and cross-field rules are validated before generation; framework-specific rules live in the adapter registry
- `launchpadai/frameworks/` — one adapter module per framework, each exposing a frozen `FrameworkAdapter` and a `generate()` hook; `registry.py` is the single registry that CLI choices and generation dispatch derive from
- `launchpadai/generators/` — framework-agnostic layer generators (API, guardrails, memory, knowledge, ...)

Adding a framework = one adapter module + a registry entry + dependency mappings + tests. Core generators never branch on framework names.

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
git clone https://github.com/venkatviswa/launchpadai.git
cd launchpadai
uv sync                      # or: pip install -e ".[dev]"
```

### Running Tests

```bash
uv run pytest                          # fast tests (unit + integration + validation)
uv run pytest tests/unit -x           # generator unit tests
uv run pytest tests/smoke -m slow     # render → install → import smoke tests (slow)
```

| Layer | What It Validates |
|-------|-------------------|
| **Unit** | Each generator in isolation with parametrized configs |
| **Integration** | Full project generation, edge cases, cross-generator consistency (pairwise via allpairspy) |
| **Validation** | All generated `.py`, `.yaml`, `.json`, `.ipynb` files parse correctly |
| **Smoke** | Each adapter's generated project installs its requirements, imports its entrypoint, and passes its own generated test suite against real, current framework releases |

CI runs the fast suite on every push, plus one smoke lane per framework adapter — an upstream breaking release fails only its adapter's lane.

---

## Tech Stack

The CLI itself is deliberately small: [Typer](https://typer.tiangolo.com/) (CLI), [Rich](https://rich.readthedocs.io/) (terminal output), [Pydantic v2](https://docs.pydantic.dev/) (typed configuration), and [PyYAML](https://pyyaml.org/) (config persistence). Generated projects pull in only the dependencies for the options you select — see [Configuration Options](#configuration-options).

---

## License

MIT
