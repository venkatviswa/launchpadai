"""Unit tests for guardrails_layer generator."""
import ast
import pytest
from launchpadai.generators.guardrails_layer import generate_guardrails_layer


@pytest.mark.unit
def test_skipped_when_disabled(tmp_path, make_config):
    config = make_config(include_guardrails=False)
    generate_guardrails_layer(config, tmp_path)
    assert not (tmp_path / "guardrails").exists()


@pytest.mark.unit
def test_generates_valid_python(tmp_path, make_config):
    config = make_config(include_guardrails=True)
    generate_guardrails_layer(config, tmp_path)

    for py_file in (tmp_path / "guardrails").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
def test_generates_input_and_output_filters(tmp_path, make_config):
    config = make_config(include_guardrails=True)
    generate_guardrails_layer(config, tmp_path)

    assert (tmp_path / "guardrails" / "input_filters.py").exists()
    assert (tmp_path / "guardrails" / "output_filters.py").exists()
