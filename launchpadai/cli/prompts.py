"""Interactive prompts for gathering project configuration."""
import re

from rich.console import Console
from rich.prompt import Confirm, Prompt

from launchpadai.config import AgentSpec, ProjectConfig
from launchpadai.frameworks.registry import ADAPTERS

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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")
    return slug or "agent"


def _gather_agents(orchestrations: tuple[str, ...], agent_description: str) -> tuple[list[AgentSpec], str]:
    """Ask for the agent team layout and return (agents, orchestration)."""
    multi_supported = len(orchestrations) > 1
    if not multi_supported:
        console.print("[dim]This framework supports a single agent per project.[/]")
        console.print()
        return (
            [AgentSpec(name="assistant", role="Assistant", goal=agent_description)],
            "single",
        )

    team_options = ["Single agent"]
    if "sequential" in orchestrations:
        team_options.append("Multi-agent pipeline (agents run in order)")
    if "supervisor" in orchestrations:
        team_options.append("Multi-agent with supervisor routing")

    team = _select("Agent team", team_options, default="Single agent")
    if team == "Single agent":
        return (
            [AgentSpec(name="assistant", role="Assistant", goal=agent_description)],
            "single",
        )

    orchestration = "sequential" if "pipeline" in team else "supervisor"

    agents: list[AgentSpec] = []
    console.print("[bold green]? Define your agents[/] [dim](at least 2; empty name to finish)[/]")
    suggestions = [("researcher", "Research Analyst"), ("writer", "Response Writer")]
    i = 0
    while True:
        default_name, default_role = suggestions[i] if i < len(suggestions) else ("", "")
        name = Prompt.ask(
            f"  [bold]Agent {i + 1} name[/]",
            default=default_name if default_name else None,
        )
        if not name or not name.strip():
            if len(agents) >= 2:
                break
            console.print("  [red]Define at least two agents for multi-agent orchestration.[/]")
            continue
        name = _slugify(name)
        if any(a.name == name for a in agents):
            console.print(f"  [red]Agent '{name}' already exists.[/]")
            continue
        role = Prompt.ask("    [bold]Role[/]", default=default_role or name.replace("_", " ").title())
        goal = Prompt.ask("    [bold]Goal[/] [dim](one line)[/]", default=f"Handle {role.lower()} work")
        agents.append(AgentSpec(name=name, role=role, goal=goal))
        i += 1
        if len(agents) >= 2:
            more = Confirm.ask("  [bold]Add another agent?[/]", default=False)
            if not more:
                break

    console.print()
    return agents, orchestration


def gather_project_config(project_name: str) -> ProjectConfig:
    """Gather all project configuration through interactive prompts."""

    # 1. Framework — choices derived from the adapter registry
    adapters = list(ADAPTERS.values())
    labels = {}
    for adapter in adapters:
        tier_note = "" if adapter.tier == 1 else " [Tier 2]"
        labels[f"{adapter.display_name} ({adapter.description}){tier_note}"] = adapter.name

    framework_label = _select(
        "Agent framework",
        list(labels.keys()),
        default=next(iter(labels.keys())),
    )
    framework_key = labels[framework_label]
    adapter = ADAPTERS[framework_key]

    # 2. LLM Provider
    llm_provider = _select(
        "LLM provider",
        [
            "Anthropic (Claude)",
            "OpenAI (GPT-4o, GPT-4o-mini)",
            "Local (Ollama)",
        ],
        default="Anthropic (Claude)",
    )

    llm_key = {
        "Anthropic (Claude)": "anthropic",
        "OpenAI (GPT-4o, GPT-4o-mini)": "openai",
        "Local (Ollama)": "ollama",
    }[llm_provider]

    # 3. Agent description + team
    agent_desc = Prompt.ask(
        "[bold green]? Describe your agent in one line[/] [dim](used in prompts and README)[/]",
        default="An AI-powered assistant",
    )
    console.print()
    agents, orchestration = _gather_agents(adapter.orchestrations, agent_desc)

    # 4. RAG + retrieval stack
    include_rag = Confirm.ask("[bold green]? Include RAG pipeline?[/]", default=True)
    console.print()

    embedding_key = "openai-small"
    vectordb_key = "chroma"
    retrieval_key = "custom"
    if include_rag:
        retrieval = _select(
            "Retrieval layer",
            [
                "Custom pipeline (chunkers + vector store client, full control)",
                "LlamaIndex (managed loading, chunking, and indexing)",
            ],
            default="Custom pipeline (chunkers + vector store client, full control)",
        )
        retrieval_key = "llamaindex" if retrieval.startswith("LlamaIndex") else "custom"

        embedding = _select(
            "Embedding model",
            [
                "OpenAI (text-embedding-3-small)",
                "OpenAI (text-embedding-3-large)",
                "HuggingFace — BGE-M3 (local/self-hosted)",
                "HuggingFace — GTE-Qwen2 (local/self-hosted)",
                "Nomic (nomic-embed-text-v1.5, local)",
            ],
            default="OpenAI (text-embedding-3-small)",
        )
        embedding_key = {
            "OpenAI (text-embedding-3-small)": "openai-small",
            "OpenAI (text-embedding-3-large)": "openai-large",
            "HuggingFace — BGE-M3 (local/self-hosted)": "bge-m3",
            "HuggingFace — GTE-Qwen2 (local/self-hosted)": "gte-qwen2",
            "Nomic (nomic-embed-text-v1.5, local)": "nomic",
        }[embedding]

        vector_db = _select(
            "Vector database",
            [
                "ChromaDB (local, great for dev)",
                "Pinecone (managed cloud)",
            ],
            default="ChromaDB (local, great for dev)",
        )
        vectordb_key = {
            "ChromaDB (local, great for dev)": "chroma",
            "Pinecone (managed cloud)": "pinecone",
        }[vector_db]

    # 5. Features
    console.print("[bold green]? Which features to include?[/]")
    console.print()
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

    return ProjectConfig(
        project_name=project_name,
        framework=framework_key,
        llm_provider=llm_key,
        embedding_model=embedding_key,
        vector_db=vectordb_key,
        retrieval=retrieval_key,
        include_rag=include_rag,
        include_guardrails=include_guardrails,
        include_eval=include_eval,
        include_mcp=include_mcp,
        observability=obs_key,
        ui=ui_key,
        auth=auth,
        include_notebooks=include_notebooks,
        include_data_layer=include_data_layer,
        data_format=data_format,
        include_ml_pipeline=include_ml_pipeline,
        ml_framework=ml_framework,
        include_docker=include_docker,
        agent_description=agent_desc,
        agents=agents,
        orchestration=orchestration,
    )
