"""Unit tests for agents_layer generator."""
import ast
import pytest
from launchpadai.generators.agents_layer import generate_agents_layer


FRAMEWORKS = ["plain", "langchain", "llamaindex", "crewai", "haystack"]

EXPECTED_FILES = {
    "plain": "base.py",
    "langchain": "graph.py",
    "llamaindex": "agent.py",
    "crewai": "crew.py",
    "haystack": "base.py",  # falls back to plain
}


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("include_rag", [True, False])
@pytest.mark.parametrize("include_guardrails", [True, False])
def test_generates_valid_python(tmp_path, make_config, framework, include_rag, include_guardrails):
    config = make_config(framework=framework, include_rag=include_rag, include_guardrails=include_guardrails)
    generate_agents_layer(config, tmp_path)

    for py_file in (tmp_path / "agents").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generates_correct_file(tmp_path, make_config, framework):
    config = make_config(framework=framework)
    generate_agents_layer(config, tmp_path)

    expected = EXPECTED_FILES[framework]
    assert (tmp_path / "agents" / expected).exists(), f"Expected agents/{expected} for {framework}"
    assert (tmp_path / "agents" / "__init__.py").exists()


@pytest.mark.unit
def test_plain_includes_rag_imports_when_enabled(tmp_path, make_config):
    config = make_config(framework="plain", include_rag=True)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "base.py").read_text()
    assert "from knowledge.retrieval.retriever import retriever" in content
    assert "retriever.retrieve" in content


@pytest.mark.unit
def test_plain_excludes_rag_when_disabled(tmp_path, make_config):
    config = make_config(framework="plain", include_rag=False)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "base.py").read_text()
    assert "retriever" not in content


@pytest.mark.unit
def test_plain_includes_guardrails_when_enabled(tmp_path, make_config):
    config = make_config(framework="plain", include_guardrails=True)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "base.py").read_text()
    assert "from guardrails.input_filters import check_input" in content
    assert "from guardrails.output_filters import check_output" in content


@pytest.mark.unit
def test_plain_excludes_guardrails_when_disabled(tmp_path, make_config):
    config = make_config(framework="plain", include_guardrails=False)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "base.py").read_text()
    assert "check_input" not in content
    assert "check_output" not in content


@pytest.mark.unit
def test_langchain_has_stategraph(tmp_path, make_config):
    config = make_config(framework="langchain")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "graph.py").read_text()
    assert "StateGraph" in content
    assert "class AgentState" in content


@pytest.mark.unit
def test_crewai_has_crew_pattern(tmp_path, make_config):
    config = make_config(framework="crewai")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "crew.py").read_text()
    assert "from crewai import" in content
    assert "def run_crew" in content


@pytest.mark.unit
def test_llamaindex_has_react_agent(tmp_path, make_config):
    config = make_config(framework="llamaindex")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "agent.py").read_text()
    assert "ReActAgent" in content
