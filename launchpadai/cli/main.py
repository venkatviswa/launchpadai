"""LaunchpadAI CLI — The Spring Initializr for Agentic AI Applications."""
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from pathlib import Path
import subprocess
import sys

from launchpadai.cli.prompts import gather_project_config
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


@app.command()
def init(
    project_name: str = typer.Argument(None, help="Name of the project to create"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
):
    """Create a new agentic AI project with interactive setup."""
    console.print(Panel(BANNER, style="bold cyan", expand=False))
    console.print()

    # If project name not provided as argument, ask for it
    if not project_name:
        project_name = Prompt.ask(
            "[bold green]? Project name[/]",
            default="my-ai-agent",
        )

    # Validate project name
    project_name = project_name.strip().replace(" ", "-").lower()
    project_path = Path(output_dir) / project_name

    if project_path.exists():
        overwrite = Confirm.ask(
            f"[yellow]Directory '{project_name}' already exists. Overwrite?[/]",
            default=False,
        )
        if not overwrite:
            console.print("[red]Aborted.[/]")
            raise typer.Exit(1)

    console.print()
    console.print("[bold]Let's configure your AI agent project![/]")
    console.print("[dim]Use arrow keys to select, Enter to confirm[/]")
    console.print()

    # Gather all configuration via interactive prompts
    config = gather_project_config(project_name)

    # Show summary
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

    # Install dependencies if needed
    req_file = project_path / "requirements.txt"
    if req_file.exists():
        console.print("[dim]Checking dependencies...[/]")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            cwd=str(project_path),
        )

    # Launch based on UI type
    if ui == "streamlit":
        console.print("[green]Launching Streamlit UI...[/]")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port=8501"],
            cwd=str(project_path),
        )
    elif ui == "gradio":
        console.print("[green]Launching Gradio UI...[/]")
        subprocess.run(
            [sys.executable, "ui/app.py"],
            cwd=str(project_path),
        )
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
        subprocess.run(
            [sys.executable, "-m", "agents.cli_runner"],
            cwd=str(project_path),
        )


@app.command()
def ingest(
    project_dir: str = typer.Argument(".", help="Path to the LaunchpadAI project"),
    source: str = typer.Option(None, "--source", "-s", help="Path to documents to ingest"),
):
    """Ingest documents into the vector store."""
    project_path = Path(project_dir)
    console.print(Panel("[bold cyan]LaunchpadAI Document Ingestion[/]", expand=False))
    subprocess.run(
        [sys.executable, "scripts/ingest.py"] + (["--source", source] if source else []),
        cwd=str(project_path),
    )


@app.command()
def evaluate(
    project_dir: str = typer.Argument(".", help="Path to the LaunchpadAI project"),
):
    """Run evaluation suite against the agent."""
    project_path = Path(project_dir)
    console.print(Panel("[bold cyan]LaunchpadAI Agent Evaluation[/]", expand=False))
    subprocess.run(
        [sys.executable, "eval/run_eval.py"],
        cwd=str(project_path),
    )


def _show_summary(config: dict):
    """Display a summary table of selected configuration."""
    table = Table(title="Project Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Project Name", config["project_name"])
    table.add_row("Framework", config["framework"])
    table.add_row("LLM Provider", config["llm_provider"])
    table.add_row("Embedding Model", config["embedding_model"])
    table.add_row("Vector Database", config["vector_db"])
    table.add_row("RAG Pipeline", "Yes" if config["include_rag"] else "No")
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
        "cp .env.example .env          # Add your API keys",
        "pip install -r requirements.txt",
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
    console.print("[bold green]Happy building! [/][dim]Report issues at github.com/launchpadai/launchpadai[/]")


if __name__ == "__main__":
    app()
