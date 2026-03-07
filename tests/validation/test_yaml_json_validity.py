"""Validation tests — verify all generated YAML and JSON files parse correctly."""
import json
import yaml
import pytest
from launchpadai.generators.project import ProjectGenerator


@pytest.mark.validation
@pytest.mark.parametrize("observability", ["langfuse", "langsmith", "opentelemetry", "none"])
def test_yaml_files_valid(tmp_path, make_config, observability):
    config = make_config(
        include_rag=True, include_eval=True, include_ml_pipeline=True,
        include_data_layer=True, include_notebooks=True,
        observability=observability,
    )
    project_path = tmp_path / "yaml-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    for yaml_file in project_path.rglob("*.yaml"):
        try:
            yaml.safe_load(yaml_file.read_text())
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML {yaml_file.relative_to(project_path)}: {e}")


@pytest.mark.validation
def test_nextjs_package_json_valid(tmp_path, make_config):
    config = make_config(ui="nextjs")
    project_path = tmp_path / "json-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    for json_file in project_path.rglob("*.json"):
        try:
            json.loads(json_file.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON {json_file.relative_to(project_path)}: {e}")
