"""Unit tests for knowledge_layer generator."""
import ast
import pytest
from launchpadai.generators.knowledge_layer import generate_knowledge_layer


VECTOR_DBS = ["chroma", "pinecone", "weaviate", "qdrant", "pgvector"]


@pytest.mark.unit
def test_skipped_when_rag_disabled(tmp_path, make_config):
    config = make_config(include_rag=False)
    generate_knowledge_layer(config, tmp_path)
    assert not (tmp_path / "knowledge").exists()


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", VECTOR_DBS)
def test_generates_valid_python(tmp_path, make_config, vector_db):
    config = make_config(include_rag=True, vector_db=vector_db)
    generate_knowledge_layer(config, tmp_path)

    for py_file in (tmp_path / "knowledge").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", VECTOR_DBS)
def test_generates_expected_files(tmp_path, make_config, vector_db):
    config = make_config(include_rag=True, vector_db=vector_db)
    generate_knowledge_layer(config, tmp_path)

    assert (tmp_path / "knowledge" / "ingestion" / "loaders.py").exists()
    assert (tmp_path / "knowledge" / "ingestion" / "chunkers.py").exists()
    assert (tmp_path / "knowledge" / "vectorstore" / "client.py").exists()
    assert (tmp_path / "knowledge" / "retrieval" / "retriever.py").exists()


@pytest.mark.unit
def test_chroma_vectorstore_client(tmp_path, make_config):
    config = make_config(include_rag=True, vector_db="chroma")
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "vectorstore" / "client.py").read_text()
    assert "import chromadb" in content
    assert "PersistentClient" in content


@pytest.mark.unit
def test_pinecone_vectorstore_client(tmp_path, make_config):
    config = make_config(include_rag=True, vector_db="pinecone")
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "vectorstore" / "client.py").read_text()
    assert "from pinecone import Pinecone" in content


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", ["weaviate", "qdrant", "pgvector"])
def test_stub_vectorstore_has_not_implemented(tmp_path, make_config, vector_db):
    config = make_config(include_rag=True, vector_db=vector_db)
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "vectorstore" / "client.py").read_text()
    assert "NotImplementedError" in content
