"""Unit tests for knowledge_layer generator."""
import ast
import pytest
from launchpadai.generators.knowledge_layer import generate_knowledge_layer


VECTOR_DBS = ["chroma", "pinecone"]


@pytest.mark.unit
def test_skipped_when_rag_disabled(tmp_path, make_config):
    config = make_config(include_rag=False)
    generate_knowledge_layer(config, tmp_path)
    assert not (tmp_path / "knowledge").exists()


@pytest.mark.unit
@pytest.mark.parametrize("retrieval", ["custom", "llamaindex"])
@pytest.mark.parametrize("vector_db", VECTOR_DBS)
def test_generates_valid_python(tmp_path, make_config, vector_db, retrieval):
    config = make_config(include_rag=True, vector_db=vector_db, retrieval=retrieval)
    generate_knowledge_layer(config, tmp_path)

    for py_file in (tmp_path / "knowledge").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", VECTOR_DBS)
def test_custom_retrieval_generates_expected_files(tmp_path, make_config, vector_db):
    config = make_config(include_rag=True, vector_db=vector_db, retrieval="custom")
    generate_knowledge_layer(config, tmp_path)

    assert (tmp_path / "knowledge" / "ingestion" / "loaders.py").exists()
    assert (tmp_path / "knowledge" / "ingestion" / "chunkers.py").exists()
    assert (tmp_path / "knowledge" / "vectorstore" / "client.py").exists()
    assert (tmp_path / "knowledge" / "retrieval" / "retriever.py").exists()


@pytest.mark.unit
def test_chroma_vectorstore_client(tmp_path, make_config):
    config = make_config(include_rag=True, vector_db="chroma", retrieval="custom")
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "vectorstore" / "client.py").read_text()
    assert "import chromadb" in content
    assert "PersistentClient" in content


@pytest.mark.unit
def test_pinecone_vectorstore_client(tmp_path, make_config):
    config = make_config(include_rag=True, vector_db="pinecone", retrieval="custom")
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "vectorstore" / "client.py").read_text()
    assert "from pinecone import Pinecone" in content


@pytest.mark.unit
def test_custom_retriever_interface(tmp_path, make_config):
    config = make_config(include_rag=True, retrieval="custom")
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "retrieval" / "retriever.py").read_text()
    assert "class Retriever" in content
    assert "def retrieve(" in content
    assert "def format_context(" in content
    assert "retriever = Retriever()" in content
    # retrieve_with_scores was removed from the Retriever interface
    assert "retrieve_with_scores" not in content


# ---------------------------------------------------------------------------
# LlamaIndex retrieval option
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_llamaindex_generates_only_retriever(tmp_path, make_config):
    """LlamaIndex owns loading/chunking/indexing — no ingestion/ or vectorstore/."""
    config = make_config(include_rag=True, retrieval="llamaindex")
    generate_knowledge_layer(config, tmp_path)

    assert (tmp_path / "knowledge" / "retrieval" / "retriever.py").exists()
    assert not (tmp_path / "knowledge" / "ingestion").exists()
    assert not (tmp_path / "knowledge" / "vectorstore").exists()


@pytest.mark.unit
def test_llamaindex_retriever_interface(tmp_path, make_config):
    config = make_config(include_rag=True, retrieval="llamaindex")
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "retrieval" / "retriever.py").read_text()
    assert "class Retriever" in content
    assert "def retrieve(" in content
    assert "def format_context(" in content
    assert "def rebuild(" in content
    assert "retriever = Retriever()" in content


@pytest.mark.unit
@pytest.mark.parametrize("embedding_model,expected", [
    ("openai-small", "OpenAIEmbedding"),
    ("openai-large", "OpenAIEmbedding"),
    ("bge-m3", "HuggingFaceEmbedding"),
    ("gte-qwen2", "HuggingFaceEmbedding"),
    ("nomic", "HuggingFaceEmbedding"),
])
def test_llamaindex_retriever_embedding_integration(tmp_path, make_config, embedding_model, expected):
    config = make_config(include_rag=True, retrieval="llamaindex", embedding_model=embedding_model)
    generate_knowledge_layer(config, tmp_path)

    content = (tmp_path / "knowledge" / "retrieval" / "retriever.py").read_text()
    assert expected in content
