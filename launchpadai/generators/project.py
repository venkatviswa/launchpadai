"""Project generator — creates folder structure and starter files based on config."""
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

from launchpadai.generators.requirements import generate_requirements
from launchpadai.generators.config_files import generate_config_files
from launchpadai.generators.models_layer import generate_models_layer
from launchpadai.generators.knowledge_layer import generate_knowledge_layer
from launchpadai.generators.tools_layer import generate_tools_layer
from launchpadai.generators.prompts_layer import generate_prompts_layer
from launchpadai.generators.agents_layer import generate_agents_layer
from launchpadai.generators.guardrails_layer import generate_guardrails_layer
from launchpadai.generators.memory_layer import generate_memory_layer
from launchpadai.generators.api_layer import generate_api_layer
from launchpadai.generators.eval_layer import generate_eval_layer
from launchpadai.generators.ui_layer import generate_ui_layer
from launchpadai.generators.docker_layer import generate_docker_layer
from launchpadai.generators.scripts_layer import generate_scripts_layer
from launchpadai.generators.cli_runner import generate_cli_runner
from launchpadai.generators.notebooks_layer import generate_notebooks_layer
from launchpadai.generators.data_layer import generate_data_layer
from launchpadai.generators.ml_pipeline import generate_ml_pipeline
from launchpadai.generators.observability_layer import generate_observability_layer
from launchpadai.generators.auth_layer import generate_auth_layer

console = Console()


class ProjectGenerator:
    """Generates the full project based on collected configuration."""

    def __init__(self, config: dict, project_path: Path):
        self.config = config
        self.path = project_path

    def generate(self):
        """Generate the full project structure."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            steps = [
                ("Creating project structure...", self._create_dirs),
                ("Generating configuration...", lambda: generate_config_files(self.config, self.path)),
                ("Setting up models layer...", lambda: generate_models_layer(self.config, self.path)),
                ("Building knowledge/RAG layer...", lambda: generate_knowledge_layer(self.config, self.path)),
                ("Creating tools layer...", lambda: generate_tools_layer(self.config, self.path)),
                ("Writing prompt templates...", lambda: generate_prompts_layer(self.config, self.path)),
                ("Generating agent logic...", lambda: generate_agents_layer(self.config, self.path)),
                ("Adding guardrails...", lambda: generate_guardrails_layer(self.config, self.path)),
                ("Setting up memory layer...", lambda: generate_memory_layer(self.config, self.path)),
                ("Creating API layer...", lambda: generate_api_layer(self.config, self.path)),
                ("Adding evaluation framework...", lambda: generate_eval_layer(self.config, self.path)),
                ("Setting up data layer...", lambda: generate_data_layer(self.config, self.path)),
                ("Creating Jupyter notebooks...", lambda: generate_notebooks_layer(self.config, self.path)),
                ("Building ML pipeline...", lambda: generate_ml_pipeline(self.config, self.path)),
                ("Setting up observability...", lambda: generate_observability_layer(self.config, self.path)),
                ("Adding authentication...", lambda: generate_auth_layer(self.config, self.path)),
                ("Generating test UI...", lambda: generate_ui_layer(self.config, self.path)),
                ("Creating Docker setup...", lambda: generate_docker_layer(self.config, self.path)),
                ("Writing scripts...", lambda: generate_scripts_layer(self.config, self.path)),
                ("Generating CLI runner...", lambda: generate_cli_runner(self.config, self.path)),
                ("Writing requirements.txt...", lambda: generate_requirements(self.config, self.path)),
                ("Saving project config...", self._save_config),
            ]

            for desc, func in steps:
                task = progress.add_task(desc, total=None)
                func()
                progress.update(task, completed=True)

        console.print(f"\n[bold green]✓ Project '{self.config['project_name']}' created at {self.path}[/]")

    def _create_dirs(self):
        """Create the directory structure."""
        dirs = [
            "config",
            "models/llm",
            "models/embeddings",
            "prompts/system",
            "prompts/few_shot",
            "agents",
            "api",
            "memory",
            "scripts",
            "tests",
            "data/documents",
            "docs",
        ]

        if self.config["include_rag"]:
            dirs.extend([
                "knowledge/ingestion",
                "knowledge/vectorstore",
                "knowledge/retrieval",
            ])

        if self.config["include_guardrails"]:
            dirs.append("guardrails")

        if self.config["include_eval"]:
            dirs.extend(["eval", "eval/datasets"])

        if self.config["include_mcp"]:
            dirs.append("tools/mcp")
        else:
            dirs.append("tools")

        if self.config["ui"] != "none":
            dirs.append("ui")

        if self.config.get("include_notebooks"):
            dirs.append("notebooks")

        if self.config.get("include_data_layer"):
            dirs.extend(["data/raw", "data/processed", "data/interim", "data/external", "data_processing"])

        if self.config.get("include_ml_pipeline"):
            dirs.extend(["ml_models/training", "ml_models/inference", "ml_models/artifacts", "ml_models/configs"])

        if self.config.get("observability", "none") != "none":
            dirs.append("observability")

        if self.config.get("auth", "none") != "none":
            dirs.append("auth")

        for d in dirs:
            (self.path / d).mkdir(parents=True, exist_ok=True)

    def _save_config(self):
        """Save the project configuration as launchpad.yaml."""
        with open(self.path / "launchpad.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
