"""Validation tests — verify all generated Python files are syntactically valid.

This is the single highest-value test: every generated .py file must
pass ast.parse() across the framework x llm_provider matrix, plus
multi-agent orchestration and llamaindex-retrieval configs.
"""
import ast
import pytest
from launchpadai.generators.project import ProjectGenerator


FRAMEWORKS = ["plain", "langgraph", "crewai", "agentscript"]
LLM_PROVIDERS = ["openai", "anthropic", "ollama"]


def _assert_all_python_parses(project_path, label):
    errors = []
    for py_file in project_path.rglob("*.py"):
        content = py_file.read_text()
        if not content.strip():
            continue
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"{py_file.relative_to(project_path)}: {e}")
    assert not errors, f"Syntax errors ({label}):\n" + "\n".join(errors)


@pytest.mark.validation
@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("llm_provider", LLM_PROVIDERS)
def test_all_generated_python_is_valid(tmp_path, make_config, framework, llm_provider):
    """Every generated .py file must pass ast.parse (full feature set)."""
    config = make_config(
        framework=framework, llm_provider=llm_provider,
        include_rag=True, include_guardrails=True, include_eval=True,
        include_mcp=True, ui="streamlit", auth="multi_user",
        include_notebooks=True, include_data_layer=True,
        include_ml_pipeline=True, include_docker=True,
        observability="langfuse",
    )
    project_path = tmp_path / "syntax-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    _assert_all_python_parses(project_path, f"{framework}/{llm_provider}")


@pytest.mark.validation
@pytest.mark.parametrize("framework", ["plain", "langgraph", "crewai"])
@pytest.mark.parametrize("orchestration", ["sequential", "supervisor"])
def test_multi_agent_python_is_valid(tmp_path, make_config, framework, orchestration):
    """Multi-agent orchestration output must parse for every tier-1 framework."""
    config = make_config(
        framework=framework, orchestration=orchestration,
        include_rag=True, include_eval=True, ui="streamlit",
    )
    project_path = tmp_path / "multi-syntax-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    _assert_all_python_parses(project_path, f"{framework}/{orchestration}")


@pytest.mark.validation
def test_llamaindex_retrieval_python_is_valid(tmp_path, make_config):
    """retrieval='llamaindex' output must parse."""
    config = make_config(
        framework="plain", retrieval="llamaindex",
        include_rag=True, include_eval=True,
    )
    project_path = tmp_path / "llamaindex-syntax-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    _assert_all_python_parses(project_path, "plain/llamaindex-retrieval")
