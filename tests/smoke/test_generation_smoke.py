"""Render → install → import smoke tests, one per framework adapter.

These are the tests that catch what static checks can't: a generated project
whose requirements don't install, or whose entrypoint doesn't import against
the real, currently-published framework versions. They create a fresh venv
per case, so they are marked slow and run per-adapter in CI.
"""
import os
import subprocess
import sys
import venv

import pytest

from launchpadai.config import ProjectConfig
from launchpadai.generators.project import ProjectGenerator

pytestmark = pytest.mark.slow

# Minimal feature set: keeps the dependency install small while still
# exercising the models/tools/memory/prompts/API layers the entrypoint pulls in.
SMOKE_BASE = {
    "llm_provider": "openai",
    "include_rag": False,
    "include_guardrails": True,  # pure-regex filters; exercises entrypoint wiring
    "include_eval": False,
    "include_mcp": False,
    "observability": "none",
    "ui": "none",
    "include_notebooks": False,
    "include_data_layer": False,
    "include_ml_pipeline": False,
    "include_docker": False,
}

MULTI_AGENTS = [
    {"name": "researcher", "role": "Research Analyst", "goal": "Find accurate information"},
    {"name": "writer", "role": "Response Writer", "goal": "Write clear answers"},
]

CASES = {
    "plain": {"framework": "plain"},
    "plain-supervisor": {
        "framework": "plain",
        "agents": MULTI_AGENTS,
        "orchestration": "supervisor",
    },
    "langgraph": {
        "framework": "langgraph",
        "agents": MULTI_AGENTS,
        "orchestration": "sequential",
    },
    "crewai": {"framework": "crewai"},
    "agentscript": {"framework": "agentscript"},
}

IMPORT_CHECK = (
    "import agents; "
    "assert hasattr(agents, 'agent'), 'agents package must expose `agent`'; "
    "assert callable(agents.agent.run), 'agent.run must be callable'; "
    "assert callable(agents.agent.reset), 'agent.reset must be callable'; "
    "print('entrypoint OK')"
)


@pytest.mark.parametrize("case", CASES.keys())
def test_render_install_import(tmp_path, case):
    overrides = CASES[case]
    config = ProjectConfig(project_name=f"smoke-{case}", **{**SMOKE_BASE, **overrides})
    project = tmp_path / config.project_name
    project.mkdir()
    ProjectGenerator(config, project).generate()

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    py = str(venv_dir / bin_dir / "python")

    install = subprocess.run(
        [py, "-m", "pip", "install", "-q", "-r", str(project / "requirements.txt")],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert install.returncode == 0, f"pip install failed:\n{install.stderr[-3000:]}"

    env = {
        **os.environ,
        # Dummy credentials: SDK clients are constructed at import time but
        # make no network calls.
        "OPENAI_API_KEY": "sk-dummy-key-for-import-check",
        "ANTHROPIC_API_KEY": "sk-ant-dummy-key-for-import-check",
    }
    result = subprocess.run(
        [py, "-c", IMPORT_CHECK],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"entrypoint import failed for {case}:\n{result.stderr[-3000:]}"
    )
    assert "entrypoint OK" in result.stdout
