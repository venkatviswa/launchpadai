"""LaunchpadAI CLI — The Spring Initializr for Agentic AI Applications."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from launchpadai.cli.prompts import gather_project_config
from launchpadai.config import AgentSpec, ProjectConfig
from launchpadai.frameworks.registry import framework_names, validate_config
from launchpadai.generators.project import ProjectGenerator

app = typer.Typer(
    name="launchpad",
    help="CLI scaffolding tool for agentic AI applications",
    no_args_is_help=True,
)
console = Console()


BANNER = r"""
  _                            _                    _
 | |    __ _ _   _ _ __   ___| |__  _ __   __ _  __| |
 | |   / _` | | | | '_ \ / __| '_ \| '_ \ / _` |/ _` |
 | |__| (_| | |_| | | | | (__| | | | |_) | (_| | (_| |
 |_____\__,_|\__,_|_| |_|\___|_| |_| .__/ \__,_|\__,_|
                                    |_|
  The Spring Initializr for AI Engineering
"""


def _parse_agent_specs(specs: list[str]) -> list[AgentSpec]:
    """Parse repeatable --agent flags of the form "name:role:goal"."""
    agents = []
    for raw in specs:
        parts = [p.strip() for p in raw.split(":", 2)]
        name = parts[0].lower().replace("-", "_").replace(" ", "_")
        role = parts[1] if len(parts) > 1 and parts[1] else name.replace("_", " ").title()
        goal = parts[2] if len(parts) > 2 and parts[2] else f"Handle {role.lower()} work"
        agents.append(AgentSpec(name=name, role=role, goal=goal))
    return agents


@app.command()
def init(
    project_name: str = typer.Argument(None, help="Name of the project to create"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
    framework: str = typer.Option(None, "--framework", "-f", help=f"Agent framework: {', '.join(framework_names())}"),
    llm: str = typer.Option(None, "--llm", help="LLM provider: openai, anthropic, ollama"),
    embedding: str = typer.Option(None, "--embedding", help="Embedding model: openai-small, openai-large, bge-m3, gte-qwen2, nomic"),
    vector_db: str = typer.Option(None, "--vector-db", help="Vector database: chroma, pinecone"),
    retrieval: str = typer.Option(None, "--retrieval", help="Retrieval layer: custom, llamaindex"),
    rag: bool = typer.Option(None, "--rag/--no-rag", help="Include RAG pipeline"),
    guardrails: bool = typer.Option(None, "--guardrails/--no-guardrails", help="Include input/output guardrails"),
    include_eval: bool = typer.Option(None, "--eval/--no-eval", help="Include evaluation framework"),
    mcp: bool = typer.Option(None, "--mcp/--no-mcp", help="Include MCP tool integration"),
    observability: str = typer.Option(None, "--observability", help="Observability: langfuse, langsmith, opentelemetry, none"),
    ui: str = typer.Option(None, "--ui", help="Test UI: streamlit, gradio, nextjs, none"),
    auth: str = typer.Option(None, "--auth", help="UI authentication: none, simple, multi_user, oauth"),
    notebooks: bool = typer.Option(None, "--notebooks/--no-notebooks", help="Include Jupyter notebooks"),
    data_layer: bool = typer.Option(None, "--data-layer/--no-data-layer", help="Include structured data layer"),
    data_format: str = typer.Option(None, "--data-format", help="Data format: csv, parquet, json, sql"),
    ml: bool = typer.Option(None, "--ml/--no-ml", help="Include ML training/inference pipeline"),
    ml_framework: str = typer.Option(None, "--ml-framework", help="ML framework: sklearn, pytorch, xgboost, transformers"),
    docker: bool = typer.Option(None, "--docker/--no-docker", help="Include Docker setup"),
    description: str = typer.Option(None, "--description", "-d", help="One-line agent description"),
    agent: list[str] = typer.Option(None, "--agent", help='Agent spec "name:role:goal" (repeat for multi-agent)'),
    orchestration: str = typer.Option(None, "--orchestration", help="Multi-agent orchestration: single, sequential, supervisor"),
    defaults: bool = typer.Option(False, "--defaults", "-y", help="Non-interactive: use defaults for anything not passed as a flag"),
    force: bool = typer.Option(False, "--force", help="Replace an existing LaunchpadAI project directory (deletes it, then regenerates)"),
):
    """Create a new agentic AI project (interactive wizard, or fully via flags)."""
    overrides = {
        "framework": framework,
        "llm_provider": llm,
        "embedding_model": embedding,
        "vector_db": vector_db,
        "retrieval": retrieval,
        "include_rag": rag,
        "include_guardrails": guardrails,
        "include_eval": include_eval,
        "include_mcp": mcp,
        "observability": observability,
        "ui": ui,
        "auth": auth,
        "include_notebooks": notebooks,
        "include_data_layer": data_layer,
        "data_format": data_format,
        "include_ml_pipeline": ml,
        "ml_framework": ml_framework,
        "include_docker": docker,
        "agent_description": description,
        "orchestration": orchestration,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if agent:
        overrides["agents"] = _parse_agent_specs(agent)

    non_interactive = defaults or bool(overrides)

    console.print(Panel(BANNER, style="bold cyan", expand=False))
    console.print()

    # If project name not provided as argument, ask for it (or default it)
    if not project_name:
        if non_interactive:
            project_name = "my-ai-agent"
        else:
            project_name = Prompt.ask(
                "[bold green]? Project name[/]",
                default="my-ai-agent",
            )

    # Validate project name — strict slug because the name is embedded in
    # generated source files and filesystem paths.
    project_name = project_name.strip().replace(" ", "-").lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", project_name):
        console.print(
            f"[red]Invalid project name '{project_name}'.[/] "
            "Use lowercase letters, digits, and hyphens, starting with a letter "
            "(max 63 characters)."
        )
        raise typer.Exit(1)
    project_path = Path(output_dir) / project_name

    if project_path.exists():
        if not force:
            if non_interactive:
                console.print(
                    f"[red]Directory '{project_name}' already exists. "
                    "Use --force to replace it.[/]"
                )
                raise typer.Exit(1)
            overwrite = Confirm.ask(
                f"[yellow]Directory '{project_name}' already exists. "
                "Delete it and regenerate?[/]",
                default=False,
            )
            if not overwrite:
                console.print("[red]Aborted.[/]")
                raise typer.Exit(1)

        # Clean regeneration — leftover files from a previous generation must
        # not survive (they would no longer match launchpad.yaml). Only ever
        # delete directories that are LaunchpadAI projects.
        if any(project_path.iterdir()) and not (project_path / "launchpad.yaml").exists():
            console.print(
                f"[red]'{project_path}' is not empty and has no launchpad.yaml — "
                "refusing to delete it. Remove it manually if you really want to.[/]"
            )
            raise typer.Exit(1)
        shutil.rmtree(project_path)

    if non_interactive:
        try:
            config = ProjectConfig(project_name=project_name, **overrides)
            validate_config(config)
        except (ValueError, KeyError) as e:
            console.print(f"[red]Invalid configuration:[/] {e}")
            raise typer.Exit(1)
        _show_summary(config)
    else:
        console.print()
        console.print("[bold]Let's configure your AI agent project![/]")
        console.print()

        # Gather all configuration via interactive prompts
        config = gather_project_config(project_name)
        try:
            validate_config(config)
        except (ValueError, KeyError) as e:
            console.print(f"[red]Invalid configuration:[/] {e}")
            raise typer.Exit(1)

        _show_summary(config)

        proceed = Confirm.ask("\n[bold green]? Generate project with this configuration?[/]", default=True)
        if not proceed:
            console.print("[red]Aborted.[/]")
            raise typer.Exit(1)

    # Generate the project
    console.print()
    generator = ProjectGenerator(config, project_path)
    generator.generate()

    # Show next steps
    _show_next_steps(config, project_path)


@app.command()
def run(
    project_dir: str = typer.Argument(".", help="Path to the LaunchpadAI project"),
):
    """Run the agent locally with the test UI."""
    project_path = Path(project_dir)
    config_file = project_path / "launchpad.yaml"

    if not config_file.exists():
        console.print("[red]Error: Not a LaunchpadAI project. Run 'launchpad init' first.[/]")
        raise typer.Exit(1)

    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)

    ui = config.get("ui", "none")
    console.print(Panel(f"[bold cyan]Starting LaunchpadAI Agent — UI: {ui}[/]", expand=False))

    env_file = project_path / ".env"
    if not env_file.exists():
        console.print("[yellow]Warning: No .env file found. Copy .env.example to .env and add your API keys.[/]")
        console.print(f"  [dim]cp {project_path}/.env.example {project_path}/.env[/]")
        raise typer.Exit(1)

    # Dependencies are installed explicitly, never as a side effect of run:
    #   pip install -r requirements.txt
    console.print("[dim]If startup fails with ImportError: pip install -r requirements.txt[/]")

    # Launch based on UI type; propagate the process's exit code
    if ui == "streamlit":
        console.print("[green]Launching Streamlit UI...[/]")
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port=8501"],
            cwd=str(project_path),
        )
        if result.returncode:
            raise typer.Exit(result.returncode)
    elif ui == "gradio":
        console.print("[green]Launching Gradio UI...[/]")
        result = subprocess.run(
            [sys.executable, "ui/app.py"],
            cwd=str(project_path),
        )
        if result.returncode:
            raise typer.Exit(result.returncode)
    elif ui == "nextjs":
        console.print("[green]Launching FastAPI backend + Next.js frontend...[/]")
        console.print("[dim]Starting API server on :8000...[/]")
        # Start FastAPI in background
        api_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.routes:app", "--port=8000", "--reload"],
            cwd=str(project_path),
        )
        console.print("[dim]Starting Next.js on :3000...[/]")
        console.print("[yellow]Run 'cd ui && npm install && npm run dev' in another terminal[/]")
        try:
            api_proc.wait()
        except KeyboardInterrupt:
            api_proc.terminate()
    else:
        # No UI — run as CLI
        console.print("[green]Running agent in CLI mode...[/]")
        result = subprocess.run(
            [sys.executable, "-m", "agents.cli_runner"],
            cwd=str(project_path),
        )
        if result.returncode:
            raise typer.Exit(result.returncode)


def _require_project(project_dir: str) -> Path:
    """Resolve and validate a LaunchpadAI project directory."""
    project_path = Path(project_dir)
    if not (project_path / "launchpad.yaml").exists():
        console.print(
            f"[red]Error: '{project_path}' is not a LaunchpadAI project "
            "(no launchpad.yaml). Run 'launchpad init' first.[/]"
        )
        raise typer.Exit(1)
    return project_path


@app.command()
def ingest(
    project_dir: str = typer.Argument(".", help="Path to the LaunchpadAI project"),
    source: str = typer.Option(None, "--source", "-s", help="Path to documents to ingest"),
):
    """Ingest documents into the vector store."""
    project_path = _require_project(project_dir)
    console.print(Panel("[bold cyan]LaunchpadAI Document Ingestion[/]", expand=False))
    result = subprocess.run(
        [sys.executable, "scripts/ingest.py"] + (["--source", source] if source else []),
        cwd=str(project_path),
    )
    if result.returncode:
        raise typer.Exit(result.returncode)


@app.command()
def evaluate(
    project_dir: str = typer.Argument(".", help="Path to the LaunchpadAI project"),
):
    """Run evaluation suite against the agent."""
    project_path = _require_project(project_dir)
    console.print(Panel("[bold cyan]LaunchpadAI Agent Evaluation[/]", expand=False))
    result = subprocess.run(
        [sys.executable, "eval/run_eval.py"],
        cwd=str(project_path),
    )
    if result.returncode:
        raise typer.Exit(result.returncode)


def _show_summary(config):
    """Display a summary table of selected configuration."""
    table = Table(title="Project Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="green")

    agents_desc = ", ".join(a.name for a in config["agents"])
    table.add_row("Project Name", config["project_name"])
    table.add_row("Framework", config["framework"])
    table.add_row("Agents", f"{agents_desc} ({config['orchestration']})")
    table.add_row("LLM Provider", config["llm_provider"])
    table.add_row("RAG Pipeline", "Yes" if config["include_rag"] else "No")
    if config["include_rag"]:
        table.add_row("Retrieval Layer", config["retrieval"])
        table.add_row("Embedding Model", config["embedding_model"])
        table.add_row("Vector Database", config["vector_db"])
    table.add_row("Guardrails", "Yes" if config["include_guardrails"] else "No")
    table.add_row("Eval Framework", "Yes" if config["include_eval"] else "No")
    table.add_row("Observability", config["observability"])
    table.add_row("Test UI", config["ui"])
    table.add_row("Authentication", config.get("auth", "none"))
    table.add_row("Jupyter Notebooks", "Yes" if config.get("include_notebooks") else "No")
    table.add_row("Data Layer", config.get("data_format", "n/a") if config.get("include_data_layer") else "No")
    table.add_row("ML Pipeline", config.get("ml_framework", "n/a") if config.get("include_ml_pipeline") else "No")
    table.add_row("Docker Setup", "Yes" if config["include_docker"] else "No")

    console.print()
    console.print(table)


def _show_next_steps(config: dict, project_path: Path):
    """Show post-generation instructions."""
    steps = [
        f"cd {project_path}",
        "pip install -r requirements.txt -r requirements-dev.txt",
        "pytest tests                   # Runs offline against the mock LLM",
        "cp .env.example .env           # Add your API keys",
    ]

    if config["include_rag"]:
        steps.append("# Add documents to data/documents/")
        steps.append("launchpad ingest               # Load documents into vector store")

    if config["ui"] == "nextjs":
        steps.append("cd ui && npm install && cd ..   # Install frontend deps")

    steps.append("launchpad run                  # Launch the agent")

    console.print()
    console.print(Panel(
        "\n".join([f"  [green]{s}[/]" if not s.startswith("#") else f"  [dim]{s}[/]" for s in steps]),
        title="[bold]Next Steps",
        expand=False,
    ))
    console.print()
    console.print("[bold green]Happy building! [/][dim]Report issues at github.com/venkatviswa/launchpadai[/]")


if __name__ == "__main__":
    app()
