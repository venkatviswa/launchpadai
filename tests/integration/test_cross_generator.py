"""Integration tests — cross-generator consistency checks.

Every framework now exposes a uniform entrypoint: agents/__init__.py defines
``class AgentEntrypoint`` and ``agent = AgentEntrypoint()``, and every consumer
(api, eval, ui, cli_runner) imports ``from agents import agent``.
"""
import pytest
from launchpadai.generators.project import ProjectGenerator


FRAMEWORKS = ["plain", "langgraph", "crewai", "agentscript"]

# Framework-specific implementation modules under agents/
AGENT_FILES = {
    "plain": ["base.py", "pipeline.py"],
    "langgraph": ["graph.py"],
    "crewai": ["crew.py"],
    "agentscript": ["client.py"],
}


def _generate(make_config, tmp_path, name, **overrides):
    config = make_config(**overrides)
    project_path = tmp_path / name
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()
    return project_path


@pytest.mark.integration
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_agents_init_defines_uniform_entrypoint(tmp_path, make_config, framework):
    """agents/__init__.py must define the AgentEntrypoint and `agent` instance."""
    project_path = _generate(make_config, tmp_path, f"init-{framework}", framework=framework)

    init_file = project_path / "agents" / "__init__.py"
    assert init_file.exists(), f"agents/__init__.py missing for {framework}"
    content = init_file.read_text()
    assert "class AgentEntrypoint" in content
    assert "agent = AgentEntrypoint()" in content


@pytest.mark.integration
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_framework_agent_modules_exist(tmp_path, make_config, framework):
    """Each framework generates its implementation module(s) under agents/."""
    project_path = _generate(make_config, tmp_path, f"mod-{framework}", framework=framework)

    for agent_file in AGENT_FILES[framework]:
        assert (project_path / "agents" / agent_file).exists(), \
            f"agents/{agent_file} missing for {framework}"


@pytest.mark.integration
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_api_layer_uses_uniform_entrypoint(tmp_path, make_config, framework):
    """api/routes.py imports the uniform entrypoint for every framework."""
    project_path = _generate(make_config, tmp_path, f"api-{framework}", framework=framework)

    api_content = (project_path / "api" / "routes.py").read_text()
    assert "from agents import agent" in api_content, \
        f"api/routes.py must import the uniform entrypoint for {framework}"


@pytest.mark.integration
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_eval_layer_uses_uniform_entrypoint(tmp_path, make_config, framework):
    """eval/run_eval.py imports the uniform entrypoint for every framework."""
    project_path = _generate(
        make_config, tmp_path, f"eval-{framework}",
        framework=framework, include_eval=True,
    )

    eval_content = (project_path / "eval" / "run_eval.py").read_text()
    assert "from agents import agent" in eval_content, \
        f"eval/run_eval.py must import the uniform entrypoint for {framework}"


@pytest.mark.integration
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_cli_runner_uses_uniform_entrypoint(tmp_path, make_config, framework):
    """agents/cli_runner.py imports the uniform entrypoint for every framework."""
    project_path = _generate(make_config, tmp_path, f"cli-{framework}", framework=framework)

    cli_content = (project_path / "agents" / "cli_runner.py").read_text()
    assert "from agents import agent" in cli_content, \
        f"cli_runner must import the uniform entrypoint for {framework}"


@pytest.mark.integration
@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("ui", ["streamlit", "gradio"])
def test_ui_layer_uses_uniform_entrypoint(tmp_path, make_config, framework, ui):
    """ui/app.py imports the uniform entrypoint for every framework."""
    project_path = _generate(
        make_config, tmp_path, f"ui-{framework}-{ui}",
        framework=framework, ui=ui,
    )

    ui_content = (project_path / "ui" / "app.py").read_text()
    assert "from agents import agent" in ui_content, \
        f"{ui} UI must import the uniform entrypoint for {framework}"


@pytest.mark.integration
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_per_agent_slices_generated(tmp_path, make_config, framework):
    """Default single-agent config generates the agents/assistant/ slice."""
    project_path = _generate(make_config, tmp_path, f"slice-{framework}", framework=framework)

    slice_dir = project_path / "agents" / "assistant"
    assert (slice_dir / "__init__.py").exists()
    assert (slice_dir / "prompts" / "system.md").exists()
    assert (slice_dir / "tools.py").exists()


@pytest.mark.integration
@pytest.mark.parametrize("framework", ["plain", "langgraph", "crewai"])
@pytest.mark.parametrize("orchestration", ["sequential", "supervisor"])
def test_multi_agent_slices_generated(tmp_path, make_config, framework, orchestration):
    """Multi-agent configs generate one slice per agent."""
    project_path = _generate(
        make_config, tmp_path, f"multi-{framework}-{orchestration}",
        framework=framework, orchestration=orchestration,
    )

    for agent_name in ("researcher", "writer"):
        slice_dir = project_path / "agents" / agent_name
        assert (slice_dir / "__init__.py").exists(), \
            f"agents/{agent_name}/ slice missing for {framework}/{orchestration}"
        assert (slice_dir / "prompts" / "system.md").exists()
        assert (slice_dir / "tools.py").exists()


@pytest.mark.integration
def test_ml_predict_tool_generated_with_ml_pipeline(tmp_path, make_config):
    """ml_pipeline should generate tools/ml_predict.py that references tools/registry."""
    project_path = _generate(make_config, tmp_path, "ml-tools", include_ml_pipeline=True)

    assert (project_path / "tools" / "ml_predict.py").exists()
    assert (project_path / "tools" / "registry.py").exists()
