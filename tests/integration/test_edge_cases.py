"""Integration tests — edge case configurations."""
import ast
import json
import yaml
import pytest
from launchpadai.generators.project import ProjectGenerator


@pytest.mark.integration
def test_minimal_config(tmp_path, make_config):
    """Everything disabled — bare minimum project."""
    config = make_config(
        include_rag=False, include_guardrails=False, include_eval=False,
        include_mcp=False, observability="none", ui="none", auth="none",
        include_notebooks=False, include_data_layer=False,
        include_ml_pipeline=False, include_docker=False,
    )
    project_path = tmp_path / config["project_name"]
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    # Core files should exist
    assert (project_path / "requirements.txt").exists()
    assert (project_path / "launchpad.yaml").exists()
    assert (project_path / "agents").is_dir()
    assert (project_path / "models").is_dir()
    assert (project_path / "README.md").exists()

    # Optional dirs should NOT exist
    assert not (project_path / "knowledge").exists()
    assert not (project_path / "guardrails").exists()
    assert not (project_path / "eval").exists()
    assert not (project_path / "ui").exists()
    assert not (project_path / "observability").exists()
    assert not (project_path / "auth").exists()
    assert not (project_path / "ml_models").exists()
    assert not (project_path / "notebooks").exists()

    # All .py files must parse
    for py_file in project_path.rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.integration
def test_maximal_config(tmp_path, make_config):
    """Everything enabled — maximum feature set."""
    config = make_config(
        framework="langgraph", llm_provider="openai",
        embedding_model="bge-m3", vector_db="pinecone",
        retrieval="custom", orchestration="supervisor",
        include_rag=True, include_guardrails=True, include_eval=True,
        include_mcp=True, observability="opentelemetry",
        ui="nextjs", auth="oauth",
        include_notebooks=True, include_data_layer=True, data_format="sql",
        include_ml_pipeline=True, ml_framework="transformers",
        include_docker=True,
    )
    project_path = tmp_path / config["project_name"]
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    # All optional dirs should exist
    assert (project_path / "knowledge").is_dir()
    assert (project_path / "guardrails").is_dir()
    assert (project_path / "eval").is_dir()
    assert (project_path / "ui").is_dir()
    assert (project_path / "observability").is_dir()
    assert (project_path / "auth").is_dir()
    assert (project_path / "ml_models").is_dir()
    assert (project_path / "data_processing").is_dir()
    assert (project_path / "notebooks").is_dir()

    # Multi-agent slices should exist for the default supervisor agents
    assert (project_path / "agents" / "researcher").is_dir()
    assert (project_path / "agents" / "writer").is_dir()

    # All .py files must parse
    errors = []
    for py_file in project_path.rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{py_file.relative_to(project_path)}: {e}")
    assert not errors, "Syntax errors:\n" + "\n".join(errors)

    # All YAML files must parse (incl. docker-compose.yml)
    for pattern in ("*.yaml", "*.yml"):
        for yaml_file in project_path.rglob(pattern):
            yaml.safe_load(yaml_file.read_text())

    # All JSON files must parse
    for json_file in project_path.rglob("*.json"):
        json.loads(json_file.read_text())


@pytest.mark.integration
@pytest.mark.parametrize("framework", ["plain", "langgraph", "crewai", "agentscript"])
def test_every_framework_generates_cleanly(tmp_path, make_config, framework):
    """Each framework should produce a valid project."""
    config = make_config(framework=framework, include_rag=True)
    project_path = tmp_path / f"fw-{framework}"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    for py_file in project_path.rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.integration
def test_unknown_framework_rejected(tmp_path, make_config):
    """Removed/unknown frameworks are rejected by the adapter registry."""
    for framework in ["langchain", "llamaindex", "haystack", "nonsense"]:
        config = make_config(framework=framework)
        project_path = tmp_path / f"bad-{framework}"
        project_path.mkdir()
        with pytest.raises(KeyError):
            ProjectGenerator(config, project_path)


@pytest.mark.integration
@pytest.mark.parametrize("orchestration", ["sequential", "supervisor"])
def test_agentscript_rejects_multi_agent_orchestration(tmp_path, make_config, orchestration):
    """agentscript supports only single-agent orchestration."""
    config = make_config(framework="agentscript", orchestration=orchestration)
    project_path = tmp_path / f"as-{orchestration}"
    project_path.mkdir()
    with pytest.raises(ValueError):
        ProjectGenerator(config, project_path)


@pytest.mark.integration
def test_special_characters_in_project_name(tmp_path, make_config):
    """Slug names generate; unsafe names are rejected by the config model."""
    for name in ["my-ai-agent-2024", "a", "agent-2"]:
        config = make_config(project_name=name)
        project_path = tmp_path / name
        project_path.mkdir()
        ProjectGenerator(config, project_path).generate()

        for py_file in project_path.rglob("*.py"):
            content = py_file.read_text()
            if content.strip():
                ast.parse(content)

    # The name is interpolated into generated code and paths, so anything
    # outside the strict slug pattern must be rejected outright.
    for bad in ["test_project", "../escape", "/absolute", 'quote"name', "Name", "9starts-digit", "a" * 64]:
        with pytest.raises(ValueError):
            make_config(project_name=bad)


@pytest.mark.integration
def test_launchpad_yaml_saved_correctly(tmp_path, make_config):
    """launchpad.yaml should round-trip the config, including agents."""
    config = make_config(
        framework="crewai", llm_provider="anthropic",
        retrieval="llamaindex", orchestration="sequential",
    )
    project_path = tmp_path / config["project_name"]
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    saved = yaml.safe_load((project_path / "launchpad.yaml").read_text())
    assert saved["framework"] == "crewai"
    assert saved["llm_provider"] == "anthropic"
    assert saved["retrieval"] == "llamaindex"
    assert saved["orchestration"] == "sequential"
    # agents serialize as a list of dicts
    agent_names = [a["name"] for a in saved["agents"]]
    assert agent_names == ["researcher", "writer"]
