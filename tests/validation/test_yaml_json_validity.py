"""Validation tests — verify all generated YAML and JSON files parse correctly."""
import json
import yaml
import pytest
from launchpadai.generators.project import ProjectGenerator


def _assert_all_yaml_json_valid(project_path):
    for pattern in ("*.yaml", "*.yml"):
        for yaml_file in project_path.rglob(pattern):
            try:
                yaml.safe_load(yaml_file.read_text())
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML {yaml_file.relative_to(project_path)}: {e}")
    for json_file in project_path.rglob("*.json"):
        try:
            json.loads(json_file.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON {json_file.relative_to(project_path)}: {e}")


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

    _assert_all_yaml_json_valid(project_path)


@pytest.mark.validation
def test_nextjs_package_json_valid(tmp_path, make_config):
    config = make_config(ui="nextjs")
    project_path = tmp_path / "json-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    _assert_all_yaml_json_valid(project_path)


@pytest.mark.validation
@pytest.mark.parametrize("vector_db", ["chroma", "pinecone"])
def test_docker_compose_valid_yaml(tmp_path, make_config, vector_db):
    """docker-compose.yml must be valid YAML for both vector DBs."""
    config = make_config(vector_db=vector_db, include_rag=True, include_docker=True)
    project_path = tmp_path / f"docker-{vector_db}"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    compose_file = project_path / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml missing with include_docker=True"
    compose = yaml.safe_load(compose_file.read_text())
    assert isinstance(compose, dict)
    assert "agent" in compose["services"]
    depends_on = compose["services"]["agent"].get("depends_on", [])
    assert isinstance(depends_on, list)
    if vector_db == "chroma":
        assert "chroma" in compose["services"]
    else:
        # Pinecone is a managed service — no extra local container
        assert depends_on == []


@pytest.mark.validation
def test_agentscript_sfdx_and_agent_bundle(tmp_path, make_config):
    """agentscript projects ship a valid sfdx-project.json and an .agent bundle."""
    config = make_config(framework="agentscript")
    project_path = tmp_path / "as-json"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    sfdx_file = project_path / "sfdx-project.json"
    assert sfdx_file.exists(), "sfdx-project.json missing for agentscript"
    sfdx = json.loads(sfdx_file.read_text())
    assert isinstance(sfdx, dict)

    agent_bundles = list(
        (project_path / "force-app" / "main" / "aiAuthoringBundles").glob("*/*.agent")
    )
    assert agent_bundles, ".agent authoring bundle missing for agentscript"

    # agentscript adds Salesforce connection vars to .env.example
    env_example = (project_path / ".env.example").read_text()
    assert "SF_" in env_example

    _assert_all_yaml_json_valid(project_path)


@pytest.mark.validation
def test_env_example_has_no_removed_providers(tmp_path, make_config):
    """.env.example must not mention removed providers/dbs."""
    config = make_config(include_rag=True)
    project_path = tmp_path / "env-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    env_example = (project_path / ".env.example").read_text().lower()
    for removed in ("google", "cohere", "weaviate", "qdrant", "pgvector"):
        assert removed not in env_example, f"Removed option '{removed}' still in .env.example"
