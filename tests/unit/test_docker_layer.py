"""Unit tests for docker_layer generator."""
import pytest
from launchpadai.generators.docker_layer import generate_docker_layer


VECTOR_DBS = ["chroma", "pinecone", "weaviate", "qdrant", "pgvector"]


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
def test_chroma_service_in_compose(tmp_path, make_config):
    config = make_config(include_docker=True, vector_db="chroma")
    generate_docker_layer(config, tmp_path)

    content = (tmp_path / "docker-compose.yml").read_text()
    assert "chroma" in content
    assert "chromadb/chroma" in content


@pytest.mark.unit
def test_pgvector_service_in_compose(tmp_path, make_config):
    config = make_config(include_docker=True, vector_db="pgvector")
    generate_docker_layer(config, tmp_path)

    content = (tmp_path / "docker-compose.yml").read_text()
    assert "postgres" in content
    assert "pgvector/pgvector" in content
