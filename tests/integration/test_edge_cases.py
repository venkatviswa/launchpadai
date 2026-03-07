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
        framework="langchain", llm_provider="openai",
        embedding_model="bge-m3", vector_db="pgvector",
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

    # All .py files must parse
    errors = []
    for py_file in project_path.rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{py_file.relative_to(project_path)}: {e}")
    assert not errors, f"Syntax errors:\n" + "\n".join(errors)

    # All YAML files must parse
    for yaml_file in project_path.rglob("*.yaml"):
        yaml.safe_load(yaml_file.read_text())

    # All JSON files must parse
    for json_file in project_path.rglob("*.json"):
        json.loads(json_file.read_text())


@pytest.mark.integration
@pytest.mark.parametrize("framework", ["plain", "langchain", "llamaindex", "crewai", "haystack"])
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
def test_special_characters_in_project_name(tmp_path, make_config):
    """Project names with hyphens and numbers should work."""
    for name in ["my-ai-agent-2024", "test_project", "a"]:
        config = make_config(project_name=name)
        project_path = tmp_path / name
        project_path.mkdir()
        ProjectGenerator(config, project_path).generate()

        for py_file in project_path.rglob("*.py"):
            content = py_file.read_text()
            if content.strip():
                ast.parse(content)


@pytest.mark.integration
def test_launchpad_yaml_saved_correctly(tmp_path, make_config):
    """launchpad.yaml should contain the config we passed in."""
    config = make_config(framework="crewai", llm_provider="anthropic")
    project_path = tmp_path / config["project_name"]
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    saved = yaml.safe_load((project_path / "launchpad.yaml").read_text())
    assert saved["framework"] == "crewai"
    assert saved["llm_provider"] == "anthropic"
