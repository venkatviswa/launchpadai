# LaunchpadAI

**The Spring Initializr for Agentic AI Applications**

LaunchpadAI is an interactive CLI that scaffolds production-ready agentic AI projects in seconds. Instead of spending days wiring together LLMs, vector databases, frameworks, and UIs, answer a few questions and get a fully functional, well-architected project — ready to customize and deploy.

---

## Why LaunchpadAI?

Building an AI agent today means making dozens of decisions before writing a single line of business logic:

- Which LLM provider? Which framework? Which vector database?
- How do I structure RAG pipelines, tool calling, and memory?
- What about guardrails, evaluation, observability, and auth?
- How do I containerize and deploy this thing?

**LaunchpadAI eliminates this boilerplate.** It generates a complete, opinionated project structure with clean separation of concerns — every layer is independently configurable and production-ready out of the box.

### What You Get

- A **modular, layered architecture** following AI engineering best practices
- **Working code** — not stubs. Every generated file is syntactically valid and import-ready
- **Auto-generated architecture diagrams** (Mermaid) showing your request flow
- **Evaluation framework** with test cases to measure agent quality from day one
- **Docker setup** for one-command deployment
- **Jupyter notebooks** for data exploration and experimentation

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/venkatviswa/launchpadai.git
cd launchpadai

# Install in development mode
pip install -e .
```

### Create Your First Project

```bash
launchpad init my-agent
```

The interactive wizard walks you through every configuration choice with sensible defaults. When it's done, you'll have a fully scaffolded project.

### Run Your Agent

```bash
cd my-agent

# Copy the environment template and add your API keys
cp .env.example .env
# Edit .env with your keys (OPENAI_API_KEY, etc.)

# Start the agent
launchpad run
```

---

## CLI Commands

### `launchpad init [project_name]`

Creates a new agentic AI project through an interactive setup wizard.

```bash
# Create in current directory
launchpad init my-agent

# Specify output directory
launchpad init my-agent --output ~/projects
```

The wizard guides you through selecting:
- Agent framework (LangChain, LlamaIndex, CrewAI, Haystack, or plain Python)
- LLM provider and embedding model
- Vector database for RAG
- UI framework and authentication
- Observability, guardrails, evaluation, and more

After generation, a `launchpad.yaml` file saves your configuration for reproducibility.

### `launchpad run [project_dir]`

Runs the agent locally with the configured UI.

```bash
# Run from project directory
launchpad run

# Run from another location
launchpad run ./my-agent
```

Automatically installs dependencies and launches the appropriate interface (Streamlit, Gradio, Next.js, or CLI).

### `launchpad ingest [project_dir]`

Ingests documents into the vector store for RAG-enabled projects.

```bash
# Ingest from default data/documents/ directory
launchpad ingest

# Ingest from a specific source
launchpad ingest --source ~/my-documents
```

### `launchpad evaluate [project_dir]`

Runs the evaluation suite against your agent using predefined test cases.

```bash
launchpad evaluate
```

Executes `eval/run_eval.py` with the test cases defined in `eval/datasets/test_cases.yaml`.

---

## Configuration Options

LaunchpadAI supports 19 configuration dimensions. Here's what you can choose:

### Agent Framework

| Framework | Description |
|-----------|-------------|
| **Plain Python** | No framework — full control over the agent loop |
| **LangChain / LangGraph** | Graph-based agent orchestration with LangGraph |
| **LlamaIndex** | Data-centric agent framework |
| **CrewAI** | Multi-agent collaboration framework |
| **Haystack** | Pipeline-based NLP framework |

### LLM Provider

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4o, GPT-4o-mini |
| **Anthropic** | Claude |
| **Google** | Gemini |
| **Ollama** | Local models (Llama, Mistral, etc.) |
| **Multiple** | Configure multiple providers |

### Embedding Models

| Model | Type |
|-------|------|
| OpenAI text-embedding-3-small | Cloud (API) |
| OpenAI text-embedding-3-large | Cloud (API) |
| Cohere embed-v4 | Cloud (API) |
| BGE-M3 | Local (HuggingFace) |
| GTE-Qwen2 | Local (HuggingFace) |
| Nomic embed-text-v1.5 | Local |
| Ollama | Local |

### Vector Databases

| Database | Type |
|----------|------|
| **ChromaDB** | Local, great for development |
| **Pinecone** | Managed cloud |
| **Weaviate** | Managed or self-hosted |
| **Qdrant** | Managed or self-hosted |
| **pgvector** | PostgreSQL extension |

### UI Options

| UI | Best For |
|----|----------|
| **Streamlit** | Fastest to get running, great for prototyping |
| **Gradio** | Demos and sharing with others |
| **Next.js** | Production-grade, separate frontend/backend |
| **CLI only** | No web UI, terminal interaction |

### Authentication

| Auth | Description |
|------|-------------|
| **None** | Open access (local dev) |
| **Simple password** | Single shared password via env var |
| **Multi-user** | Username + password pairs |
| **OAuth / SSO** | Google, GitHub (Next.js only) |

### Observability

| Platform | Type |
|----------|------|
| **LangFuse** | Open-source tracing |
| **LangSmith** | LangChain ecosystem |
| **OpenTelemetry** | Vendor-neutral standard |
| **None** | Add later |

### Data & ML

| Option | Choices |
|--------|---------|
| **Data format** | CSV/TSV, Parquet, JSON/JSONL, SQL |
| **ML framework** | scikit-learn, PyTorch, XGBoost/LightGBM, HuggingFace Transformers |
| **Notebooks** | Jupyter notebooks for EDA (yes/no) |
| **Data layer** | Structured data processing (yes/no) |
| **ML pipeline** | Model training & inference (yes/no) |

### Additional Features

| Feature | Description |
|---------|-------------|
| **RAG pipeline** | Document ingestion, chunking, retrieval |
| **Guardrails** | Input/output safety, PII detection |
| **Evaluation** | Test case framework for agent quality |
| **MCP tools** | Model Context Protocol server integration |
| **Docker** | Dockerfile + docker-compose for deployment |

---

## Generated Project Structure

After running `launchpad init`, your project looks like this:

```
my-ai-agent/
├── agents/              # Agent orchestration (framework-specific)
├── api/                 # FastAPI routes & Pydantic schemas
├── auth/                # Authentication middleware
├── config/              # settings.py, application constants
├── data/                # Raw, processed, and external data
│   ├── documents/       # Documents for RAG ingestion
│   ├── raw/             # Raw datasets
│   ├── processed/       # Cleaned/transformed data
│   └── external/        # External data sources
├── data_processing/     # Data loading and transformation
├── docs/                # Architecture documentation
├── eval/                # Evaluation framework
│   ├── datasets/        # Test case definitions (YAML)
│   └── run_eval.py      # Evaluation runner
├── guardrails/          # Input/output safety filters
├── knowledge/           # RAG pipeline
│   ├── ingestion/       # Document loaders and chunkers
│   ├── vectorstore/     # Vector database client
│   └── retrieval/       # Retriever implementation
├── memory/              # Conversation history management
├── ml_models/           # ML training, inference, registry
│   ├── training/        # Model training scripts
│   ├── inference/       # Prediction serving
│   ├── configs/         # Training configurations
│   └── artifacts/       # Saved model checkpoints
├── models/              # LLM & embedding provider abstractions
│   ├── llm/             # LLM provider (OpenAI, Anthropic, etc.)
│   └── embeddings/      # Embedding model provider
├── notebooks/           # Jupyter notebooks for exploration
├── observability/       # Tracing, cost tracking, dashboards
├── prompts/             # System prompts, few-shot examples
├── scripts/             # Utility scripts (ingestion, etc.)
├── tests/               # Test suite
├── tools/               # Tool registry and MCP servers
├── ui/                  # Chat interface (Streamlit/Gradio/Next.js)
├── .env.example         # Environment variable template
├── .gitignore
├── docker-compose.yml   # Container orchestration
├── Dockerfile           # Application container
├── launchpad.yaml       # Project configuration (reproducible)
├── README.md            # Auto-generated project README
└── requirements.txt     # Python dependencies
```

Only the layers you enable are generated — a minimal project (plain Python, no RAG, no UI) produces just the core agent, models, and API layers.

---

## Architecture

Every generated project follows a layered architecture with clean separation of concerns:

```
┌─────────────────────────────────────────────┐
│                  User Interface              │
│          (Streamlit / Gradio / Next.js)       │
├─────────────────────────────────────────────┤
│              Authentication Layer            │
├─────────────────────────────────────────────┤
│               API Layer (FastAPI)             │
├─────────────────────────────────────────────┤
│                Agent Core                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Prompts  │ │Guardrails│ │   Memory     │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
├────────────┬────────────┬───────────────────┤
│   Tools    │ Knowledge  │    ML Pipeline    │
│  Registry  │   (RAG)    │  Train/Inference  │
├────────────┴────────────┴───────────────────┤
│              Models Layer                    │
│         (LLM + Embedding Providers)          │
├─────────────────────────────────────────────┤
│           Observability & Eval               │
└─────────────────────────────────────────────┘
```

A Mermaid architecture diagram is auto-generated in each project's README, customized to your specific configuration choices.

---

## Request Flow

Here's how a typical request flows through a generated project:

1. **User sends a message** through the UI (Streamlit, Gradio, Next.js, or CLI)
2. **Authentication** validates the request (if auth is enabled)
3. **FastAPI** receives the request and routes it to the agent
4. **Input guardrails** check for safety (PII, injection, blocked topics)
5. **Agent orchestrator** determines the plan (using the selected framework)
6. **RAG retrieval** fetches relevant context from the vector store (if enabled)
7. **Prompt assembly** combines system prompt, context, conversation history, and user query
8. **LLM call** generates a response through the configured provider
9. **Tool execution** handles any function calls the LLM requests
10. **Output guardrails** validate the response (hallucination check, safety)
11. **Observability** logs traces, tokens, and costs
12. **Response** is returned to the user through the UI

---

## Development

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/venkatviswa/launchpadai.git
cd launchpadai
pip install -e ".[dev]"
```

### Running Tests

The test suite includes 365+ tests across unit, integration, and validation layers:

```bash
# Run all tests
pytest tests/

# Fast unit tests (~10s)
pytest tests/unit/ -x

# Validation tests — every generated file parses correctly
pytest tests/validation/

# Integration tests — edge cases and cross-generator consistency
pytest tests/integration/

# Run in parallel
pytest tests/ -n auto

# Run by marker
pytest -m unit
pytest -m validation
pytest -m integration
```

### Test Architecture

| Layer | Tests | What It Validates |
|-------|-------|-------------------|
| **Unit** | ~244 | Each generator in isolation with parametrized configs |
| **Integration** | ~28 | Full project generation, edge cases, cross-generator consistency |
| **Validation** | ~82 | All generated `.py`, `.yaml`, `.json`, `.ipynb` files parse correctly |
| **Pairwise** | ~39 | Combinatorial coverage of config interactions (allpairspy) |

The pairwise suite uses [allpairspy](https://github.com/thombashi/allpairspy) to reduce ~12.9 million possible configuration combinations down to ~39 representative test cases that cover every two-way interaction between config options.

---

## Tech Stack

**LaunchpadAI CLI:**
- [Typer](https://typer.tiangolo.com/) — CLI framework
- [Rich](https://rich.readthedocs.io/) — Terminal formatting and progress indicators
- [Jinja2](https://jinja.palletsprojects.com/) — Template rendering
- [PyYAML](https://pyyaml.org/) — Configuration parsing

**Generated projects use (based on your selections):**
- FastAPI, Pydantic (API layer)
- LangChain/LangGraph, LlamaIndex, CrewAI, Haystack (frameworks)
- OpenAI, Anthropic, Google, Ollama SDKs (LLM providers)
- ChromaDB, Pinecone, Weaviate, Qdrant, pgvector (vector stores)
- Streamlit, Gradio, Next.js (UIs)
- Presidio (guardrails/PII detection)
- LangFuse, LangSmith, OpenTelemetry (observability)
- scikit-learn, PyTorch, XGBoost, HuggingFace Transformers (ML)
- Jupyter, Pandas, NumPy (data science)
- Docker, docker-compose (deployment)

---

## Project Stats

- **22 generator modules** producing different architectural layers
- **5** agent framework options
- **5** LLM providers
- **7** embedding model options
- **5** vector database options
- **4** UI framework choices
- **4** authentication levels
- **3** observability platforms
- **4** ML framework options
- **~12.9 million** possible configuration combinations
- **365+** automated tests

---

## License

MIT
