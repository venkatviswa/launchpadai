"""Unit tests for cli_runner generator."""
import ast
import pytest
from launchpadai.generators.cli_runner import generate_cli_runner


@pytest.mark.unit
def test_generates_valid_python(tmp_path, make_config):
    config = make_config()
    (tmp_path / "agents").mkdir(parents=True)
    generate_cli_runner(config, tmp_path)

    content = (tmp_path / "agents" / "cli_runner.py").read_text()
    ast.parse(content)


@pytest.mark.unit
def test_contains_project_name(tmp_path, make_config):
    config = make_config(project_name="my-bot")
    (tmp_path / "agents").mkdir(parents=True)
    generate_cli_runner(config, tmp_path)

    content = (tmp_path / "agents" / "cli_runner.py").read_text()
    assert "my-bot" in content


@pytest.mark.unit
def test_contains_agent_description(tmp_path, make_config):
    config = make_config(agent_description="A helpful research assistant")
    (tmp_path / "agents").mkdir(parents=True)
    generate_cli_runner(config, tmp_path)

    content = (tmp_path / "agents" / "cli_runner.py").read_text()
    assert "A helpful research assistant" in content
