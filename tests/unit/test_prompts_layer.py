"""Unit tests for prompts_layer generator."""
import ast
import yaml
import pytest
from launchpadai.generators.prompts_layer import generate_prompts_layer


@pytest.mark.unit
@pytest.mark.parametrize("include_rag", [True, False])
def test_generates_valid_python(tmp_path, make_config, include_rag):
    config = make_config(include_rag=include_rag)
    generate_prompts_layer(config, tmp_path)

    content = (tmp_path / "prompts" / "templates.py").read_text()
    ast.parse(content)


@pytest.mark.unit
def test_system_prompt_generated(tmp_path, make_config):
    config = make_config(project_name="test-agent")
    generate_prompts_layer(config, tmp_path)

    assert (tmp_path / "prompts" / "system" / "default.md").exists()
    content = (tmp_path / "prompts" / "system" / "default.md").read_text()
    assert "test-agent" in content


@pytest.mark.unit
def test_few_shot_examples_valid_yaml(tmp_path, make_config):
    config = make_config()
    generate_prompts_layer(config, tmp_path)

    yaml_file = tmp_path / "prompts" / "few_shot" / "examples.yaml"
    assert yaml_file.exists()
    yaml.safe_load(yaml_file.read_text())
