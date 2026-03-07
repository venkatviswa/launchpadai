"""Unit tests for api_layer generator."""
import ast
import pytest
from launchpadai.generators.api_layer import generate_api_layer


FRAMEWORKS = ["plain", "langchain", "llamaindex", "crewai", "haystack"]

EXPECTED_IMPORTS = {
    "plain": "from agents.base import agent",
    "langchain": "from agents.graph import agent",
    "llamaindex": "from agents.agent import agent",
    "crewai": "from agents.crew import run_crew",
    "haystack": "from agents.base import agent",
}


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generates_valid_python(tmp_path, make_config, framework):
    config = make_config(framework=framework)
    generate_api_layer(config, tmp_path)

    for py_file in (tmp_path / "api").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_correct_agent_import(tmp_path, make_config, framework):
    config = make_config(framework=framework)
    generate_api_layer(config, tmp_path)

    content = (tmp_path / "api" / "routes.py").read_text()
    assert EXPECTED_IMPORTS[framework] in content


@pytest.mark.unit
def test_crewai_uses_run_crew(tmp_path, make_config):
    config = make_config(framework="crewai")
    generate_api_layer(config, tmp_path)

    content = (tmp_path / "api" / "routes.py").read_text()
    assert "run_crew(request.message)" in content


@pytest.mark.unit
def test_project_name_in_api(tmp_path, make_config):
    config = make_config(project_name="my-cool-agent")
    generate_api_layer(config, tmp_path)

    content = (tmp_path / "api" / "routes.py").read_text()
    assert "my-cool-agent" in content


@pytest.mark.unit
def test_schemas_file_generated(tmp_path, make_config):
    config = make_config()
    generate_api_layer(config, tmp_path)

    assert (tmp_path / "api" / "schemas.py").exists()
    content = (tmp_path / "api" / "schemas.py").read_text()
    assert "class ChatRequest" in content
    assert "class ChatResponse" in content
