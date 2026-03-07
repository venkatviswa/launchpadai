"""Validation tests — verify all generated Python files are syntactically valid.

This is the single highest-value test: every generated .py file must
pass ast.parse() across the framework x llm_provider matrix.
"""
import ast
import pytest
from launchpadai.generators.project import ProjectGenerator


@pytest.mark.validation
@pytest.mark.parametrize("framework", ["plain", "langchain", "llamaindex", "crewai", "haystack"])
@pytest.mark.parametrize("llm_provider", ["openai", "anthropic", "google", "ollama", "multiple"])
def test_all_generated_python_is_valid(tmp_path, make_config, framework, llm_provider):
    """Every generated .py file must pass ast.parse."""
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

    errors = []
    for py_file in project_path.rglob("*.py"):
        content = py_file.read_text()
        if not content.strip():
            continue
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"{py_file.relative_to(project_path)}: {e}")

    assert not errors, f"Syntax errors ({framework}/{llm_provider}):\n" + "\n".join(errors)
