"""Unit tests for config_files generator."""
import ast
import pytest
from launchpadai.generators.config_files import generate_config_files


@pytest.mark.unit
@pytest.mark.parametrize("llm_provider", ["openai", "anthropic", "google", "ollama", "multiple"])
def test_env_example_contains_provider_vars(tmp_path, make_config, llm_provider):
    config = make_config(llm_provider=llm_provider)
    (tmp_path / "docs").mkdir(parents=True)
    generate_config_files(config, tmp_path)

    content = (tmp_path / ".env.example").read_text()
    if llm_provider == "openai":
        assert "OPENAI_API_KEY" in content
    elif llm_provider == "anthropic":
        assert "ANTHROPIC_API_KEY" in content


@pytest.mark.unit
@pytest.mark.parametrize("llm_provider", ["openai", "anthropic", "google", "ollama", "multiple"])
def test_settings_py_valid_python(tmp_path, make_config, llm_provider):
    config = make_config(llm_provider=llm_provider)
    (tmp_path / "docs").mkdir(parents=True)
    generate_config_files(config, tmp_path)

    content = (tmp_path / "config" / "settings.py").read_text()
    ast.parse(content)


@pytest.mark.unit
def test_gitignore_generated(tmp_path, make_config):
    config = make_config()
    (tmp_path / "docs").mkdir(parents=True)
    generate_config_files(config, tmp_path)

    content = (tmp_path / ".gitignore").read_text()
    assert ".env" in content
    assert "__pycache__" in content


@pytest.mark.unit
def test_readme_contains_project_name(tmp_path, make_config):
    config = make_config(project_name="my-awesome-agent")
    (tmp_path / "docs").mkdir(parents=True)
    generate_config_files(config, tmp_path)

    content = (tmp_path / "README.md").read_text()
    assert "my-awesome-agent" in content


@pytest.mark.unit
def test_readme_contains_mermaid_diagram(tmp_path, make_config):
    config = make_config()
    (tmp_path / "docs").mkdir(parents=True)
    generate_config_files(config, tmp_path)

    content = (tmp_path / "README.md").read_text()
    assert "```mermaid" in content


@pytest.mark.unit
def test_architecture_docs_generated(tmp_path, make_config):
    config = make_config()
    (tmp_path / "docs").mkdir(parents=True)
    generate_config_files(config, tmp_path)

    assert (tmp_path / "docs" / "architecture.mermaid").exists()
    assert (tmp_path / "docs" / "architecture.md").exists()
