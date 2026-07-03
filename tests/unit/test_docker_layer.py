"""Unit tests for docker_layer generator."""
import yaml
import pytest
from launchpadai.generators.docker_layer import generate_docker_layer


VECTOR_DBS = ["chroma", "pinecone"]


@pytest.mark.unit
def test_skipped_when_disabled(tmp_path, make_config):
    config = make_config(include_docker=False)
    generate_docker_layer(config, tmp_path)
    assert not (tmp_path / "docker-compose.yml").exists()


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", VECTOR_DBS)
def test_generates_compose_and_dockerfile(tmp_path, make_config, vector_db):
    config = make_config(include_docker=True, vector_db=vector_db)
    generate_docker_layer(config, tmp_path)

    assert (tmp_path / "docker-compose.yml").exists()
    assert (tmp_path / "Dockerfile").exists()


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", VECTOR_DBS)
def test_compose_is_valid_yaml(tmp_path, make_config, vector_db):
    config = make_config(include_docker=True, vector_db=vector_db)
    generate_docker_layer(config, tmp_path)

    data = yaml.safe_load((tmp_path / "docker-compose.yml").read_text())
    assert "services" in data
    assert "agent" in data["services"]


@pytest.mark.unit
def test_chroma_service_in_compose(tmp_path, make_config):
    config = make_config(include_docker=True, vector_db="chroma")
    generate_docker_layer(config, tmp_path)

    content = (tmp_path / "docker-compose.yml").read_text()
    assert "chroma" in content
    assert "chromadb/chroma" in content
    assert "depends_on:\n      - chroma" in content

    data = yaml.safe_load(content)
    assert data["services"]["agent"]["depends_on"] == ["chroma"]


@pytest.mark.unit
def test_pinecone_has_empty_depends_on(tmp_path, make_config):
    """Pinecone is a managed service — no local container to depend on."""
    config = make_config(include_docker=True, vector_db="pinecone")
    generate_docker_layer(config, tmp_path)

    content = (tmp_path / "docker-compose.yml").read_text()
    assert "depends_on: []" in content

    data = yaml.safe_load(content)
    assert data["services"]["agent"]["depends_on"] == []
