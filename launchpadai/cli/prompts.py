"""Interactive prompts for gathering project configuration."""
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()


def _select(question: str, options: list[str], default: str = None) -> str:
    """Display numbered options and return selection."""
    console.print(f"[bold green]? {question}[/]")
    for i, opt in enumerate(options, 1):
        marker = "[cyan]>[/]" if opt == default else " "
        console.print(f"  {marker} [bold]{i}[/]) {opt}")

    while True:
        choice = Prompt.ask("  [dim]Enter number[/]", default=str(options.index(default) + 1) if default else "1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                console.print(f"  [dim]Selected: {options[idx]}[/]")
                console.print()
                return options[idx]
        except ValueError:
            pass
        console.print(f"  [red]Please enter a number between 1 and {len(options)}[/]")


def _multi_select(question: str, options: list[str]) -> list[str]:
    """Display options and allow multiple selections."""
    console.print(f"[bold green]? {question}[/] [dim](comma-separated numbers)[/]")
    for i, opt in enumerate(options, 1):
        console.print(f"    [bold]{i}[/]) {opt}")

    while True:
        choices = Prompt.ask("  [dim]Enter numbers[/]", default="1")
        try:
            indices = [int(c.strip()) - 1 for c in choices.split(",")]
            if all(0 <= idx < len(options) for idx in indices):
                selected = [options[idx] for idx in indices]
                console.print(f"  [dim]Selected: {', '.join(selected)}[/]")
                console.print()
                return selected
        except ValueError:
            pass
        console.print(f"  [red]Please enter comma-separated numbers between 1 and {len(options)}[/]")


def gather_project_config(project_name: str) -> dict:
    """Gather all project configuration through interactive prompts."""

    # 1. Framework
    framework = _select(
        "Agent framework",
        [
            "Plain Python (no framework, full control)",
            "LangChain / LangGraph",
            "LlamaIndex",
            "CrewAI",
            "Haystack",
            "Salesforce AgentScript (Agentforce DX)",
        ],
        default="Plain Python (no framework, full control)",
    )

    # Normalize framework name
    framework_key = {
        "Plain Python (no framework, full control)": "plain",
        "LangChain / LangGraph": "langchain",
        "LlamaIndex": "llamaindex",
        "CrewAI": "crewai",
        "Haystack": "haystack",
        "Salesforce AgentScript (Agentforce DX)": "agentscript",
    }[framework]

    # 2. LLM Provider
    llm_provider = _select(
        "LLM provider",
        [
            "OpenAI (GPT-4o, GPT-4o-mini)",
            "Anthropic (Claude)",
            "Google (Gemini)",
            "Local (Ollama)",
            "Multiple (configure later)",
        ],
        default="Anthropic (Claude)",
    )

    llm_key = {
        "OpenAI (GPT-4o, GPT-4o-mini)": "openai",
        "Anthropic (Claude)": "anthropic",
        "Google (Gemini)": "google",
        "Local (Ollama)": "ollama",
        "Multiple (configure later)": "multiple",
    }[llm_provider]

    # 3. Embedding Model
    embedding = _select(
        "Embedding model",
        [
            "OpenAI (text-embedding-3-small)",
            "OpenAI (text-embedding-3-large)",
            "Cohere (embed-v4)",
            "HuggingFace — BGE-M3 (local/self-hosted)",
            "HuggingFace — GTE-Qwen2 (local/self-hosted)",
            "Nomic (nomic-embed-text-v1.5, local)",
            "Ollama (local)",
        ],
        default="OpenAI (text-embedding-3-small)",
    )

    embedding_key = {
        "OpenAI (text-embedding-3-small)": "openai-small",
        "OpenAI (text-embedding-3-large)": "openai-large",
        "Cohere (embed-v4)": "cohere",
        "HuggingFace — BGE-M3 (local/self-hosted)": "bge-m3",
        "HuggingFace — GTE-Qwen2 (local/self-hosted)": "gte-qwen2",
        "Nomic (nomic-embed-text-v1.5, local)": "nomic",
        "Ollama (local)": "ollama",
    }[embedding]

    # 4. Vector Database
    vector_db = _select(
        "Vector database",
        [
            "ChromaDB (local, great for dev)",
            "Pinecone (managed cloud)",
            "Weaviate (managed or self-hosted)",
            "Qdrant (managed or self-hosted)",
            "pgvector (PostgreSQL extension)",
        ],
        default="ChromaDB (local, great for dev)",
    )

    vectordb_key = {
        "ChromaDB (local, great for dev)": "chroma",
        "Pinecone (managed cloud)": "pinecone",
        "Weaviate (managed or self-hosted)": "weaviate",
        "Qdrant (managed or self-hosted)": "qdrant",
        "pgvector (PostgreSQL extension)": "pgvector",
    }[vector_db]

    # 5. Features
    console.print("[bold green]? Which features to include?[/]")
    console.print()
    include_rag = Confirm.ask("  [bold]Include RAG pipeline?[/]", default=True)
    include_guardrails = Confirm.ask("  [bold]Include guardrails (input/output safety)?[/]", default=True)
    include_eval = Confirm.ask("  [bold]Include evaluation framework?[/]", default=True)
    include_mcp = Confirm.ask("  [bold]Include MCP tool integration?[/]", default=True)
    console.print()

    # 6. Observability
    observability = _select(
        "Observability / tracing",
        [
            "LangFuse (open-source)",
            "LangSmith (LangChain)",
            "OpenTelemetry",
            "None (add later)",
        ],
        default="LangFuse (open-source)",
    )

    obs_key = {
        "LangFuse (open-source)": "langfuse",
        "LangSmith (LangChain)": "langsmith",
        "OpenTelemetry": "opentelemetry",
        "None (add later)": "none",
    }[observability]

    # 7. Test UI
    ui = _select(
        "Test UI for the agent",
        [
            "Streamlit (fastest to get running)",
            "Gradio (good for demos & sharing)",
            "Next.js (production-grade, separate frontend)",
            "CLI only (no web UI)",
        ],
        default="Streamlit (fastest to get running)",
    )

    ui_key = {
        "Streamlit (fastest to get running)": "streamlit",
        "Gradio (good for demos & sharing)": "gradio",
        "Next.js (production-grade, separate frontend)": "nextjs",
        "CLI only (no web UI)": "none",
    }[ui]

    # 7b. Authentication (only if UI is selected)
    auth = "none"
    if ui_key != "none":
        auth_choice = _select(
            "Authentication for the test UI",
            [
                "None (open access, local dev only)",
                "Simple password (single shared password via env var)",
                "Multi-user (username + password pairs via env var)",
                "OAuth / SSO (Google, GitHub — Next.js only)",
            ],
            default="None (open access, local dev only)",
        )
        auth = {
            "None (open access, local dev only)": "none",
            "Simple password (single shared password via env var)": "simple",
            "Multi-user (username + password pairs via env var)": "multi_user",
            "OAuth / SSO (Google, GitHub — Next.js only)": "oauth",
        }[auth_choice]

        # OAuth only works with Next.js
        if auth == "oauth" and ui_key != "nextjs":
            console.print("  [yellow]Note: OAuth/SSO requires Next.js. Falling back to multi-user auth.[/]")
            auth = "multi_user"

    # 8. Data & ML Pipeline
    console.print("[bold green]? Data & ML capabilities[/]")
    console.print()
    include_notebooks = Confirm.ask("  [bold]Include Jupyter notebooks for EDA?[/]", default=True)

    include_data_layer = False
    data_format = "csv"
    if include_notebooks:
        include_data_layer = Confirm.ask("  [bold]Include structured data layer (datasets, versioning)?[/]", default=True)
        if include_data_layer:
            data_format_choice = _select(
                "Primary data format",
                [
                    "CSV / TSV files",
                    "Parquet (columnar, faster)",
                    "JSON / JSONL",
                    "Database (SQL)",
                ],
                default="CSV / TSV files",
            )
            data_format = {
                "CSV / TSV files": "csv",
                "Parquet (columnar, faster)": "parquet",
                "JSON / JSONL": "json",
                "Database (SQL)": "sql",
            }[data_format_choice]

    include_ml_pipeline = Confirm.ask("  [bold]Include ML model training & inference pipeline?[/]", default=False)

    ml_framework = "sklearn"
    if include_ml_pipeline:
        ml_framework_choice = _select(
            "ML framework",
            [
                "scikit-learn (classical ML, tabular data)",
                "PyTorch (deep learning, custom models)",
                "XGBoost / LightGBM (gradient boosting)",
                "HuggingFace Transformers (NLP fine-tuning)",
            ],
            default="scikit-learn (classical ML, tabular data)",
        )
        ml_framework = {
            "scikit-learn (classical ML, tabular data)": "sklearn",
            "PyTorch (deep learning, custom models)": "pytorch",
            "XGBoost / LightGBM (gradient boosting)": "xgboost",
            "HuggingFace Transformers (NLP fine-tuning)": "transformers",
        }[ml_framework_choice]

    console.print()

    # 9. Docker
    include_docker = Confirm.ask(
        "[bold green]? Include Docker / docker-compose setup?[/]",
        default=True,
    )

    # 10. Agent description (optional)
    console.print()
    agent_desc = Prompt.ask(
        "[bold green]? Describe your agent in one line[/] [dim](optional, used in README)[/]",
        default="An AI-powered assistant",
    )

    return {
        "project_name": project_name,
        "framework": framework_key,
        "llm_provider": llm_key,
        "embedding_model": embedding_key,
        "vector_db": vectordb_key,
        "include_rag": include_rag,
        "include_guardrails": include_guardrails,
        "include_eval": include_eval,
        "include_mcp": include_mcp,
        "observability": obs_key,
        "ui": ui_key,
        "auth": auth,
        "include_notebooks": include_notebooks,
        "include_data_layer": include_data_layer,
        "data_format": data_format,
        "include_ml_pipeline": include_ml_pipeline,
        "ml_framework": ml_framework,
        "include_docker": include_docker,
        "agent_description": agent_desc,
    }
