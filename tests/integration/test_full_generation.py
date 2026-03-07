"""Integration tests — full project generation with pairwise configs.

Uses allpairspy to generate ~150-250 configs that cover every 2-way
interaction between config options, then validates each generated project.
"""
import ast
import json
import yaml
import pytest
from launchpadai.generators.project import ProjectGenerator
from tests.conftest import PAIRWISE_CONFIGS


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    PAIRWISE_CONFIGS,
    ids=[f"pw_{i}" for i in range(len(PAIRWISE_CONFIGS))],
)
def test_pairwise_generation(tmp_path, config):
    """Full project generation + validation for a pairwise config."""
    project_path = tmp_path / config["project_name"]
    project_path.mkdir(parents=True)

    # Generation should not raise
    generator = ProjectGenerator(config, project_path)
    generator.generate()

    # 1. All .py files must parse
    errors = []
    for py_file in project_path.rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{py_file.relative_to(project_path)}: {e}")
    assert not errors, f"Syntax errors in config {config['project_name']}:\n" + "\n".join(errors)

    # 2. All .yaml files must parse
    for yaml_file in project_path.rglob("*.yaml"):
        try:
            yaml.safe_load(yaml_file.read_text())
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML {yaml_file.relative_to(project_path)}: {e}")

    # 3. All .json and .ipynb files must parse
    for json_file in project_path.rglob("*.json"):
        try:
            json.loads(json_file.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON {json_file.relative_to(project_path)}: {e}")
    for ipynb_file in project_path.rglob("*.ipynb"):
        try:
            json.loads(ipynb_file.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid notebook {ipynb_file.relative_to(project_path)}: {e}")

    # 4. requirements.txt must exist with packages
    req_file = project_path / "requirements.txt"
    assert req_file.exists()
    packages = [l for l in req_file.read_text().split("\n") if l.strip() and not l.startswith("#")]
    assert len(packages) >= 2, "requirements.txt should have at least base deps"

    # 5. launchpad.yaml must exist
    assert (project_path / "launchpad.yaml").exists()
