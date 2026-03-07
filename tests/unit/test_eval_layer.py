"""Unit tests for eval_layer generator."""
import ast
import yaml
import pytest
from launchpadai.generators.eval_layer import generate_eval_layer


@pytest.mark.unit
def test_skipped_when_disabled(tmp_path, make_config):
    config = make_config(include_eval=False)
    generate_eval_layer(config, tmp_path)
    assert not (tmp_path / "eval").exists()


@pytest.mark.unit
def test_generates_valid_python(tmp_path, make_config):
    config = make_config(include_eval=True)
    generate_eval_layer(config, tmp_path)

    for py_file in (tmp_path / "eval").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
def test_test_cases_yaml_valid(tmp_path, make_config):
    config = make_config(include_eval=True)
    generate_eval_layer(config, tmp_path)

    yaml_file = tmp_path / "eval" / "datasets" / "test_cases.yaml"
    assert yaml_file.exists()
    data = yaml.safe_load(yaml_file.read_text())
    assert "test_cases" in data
    assert len(data["test_cases"]) > 0
