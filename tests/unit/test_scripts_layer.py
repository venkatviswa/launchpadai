"""Unit tests for scripts_layer generator."""
import ast
import pytest
from launchpadai.generators.scripts_layer import generate_scripts_layer


@pytest.mark.unit
@pytest.mark.parametrize("retrieval", ["custom", "llamaindex"])
@pytest.mark.parametrize("include_rag", [True, False])
def test_generates_valid_python(tmp_path, make_config, include_rag, retrieval):
    config = make_config(include_rag=include_rag, retrieval=retrieval)
    generate_scripts_layer(config, tmp_path)

    content = (tmp_path / "scripts" / "ingest.py").read_text()
    ast.parse(content)


@pytest.mark.unit
def test_custom_rag_script_has_ingestion_logic(tmp_path, make_config):
    config = make_config(include_rag=True, retrieval="custom")
    generate_scripts_layer(config, tmp_path)

    content = (tmp_path / "scripts" / "ingest.py").read_text()
    assert "DocumentLoader" in content
    assert "vectorstore" in content


@pytest.mark.unit
def test_llamaindex_script_rebuilds_index(tmp_path, make_config):
    """LlamaIndex retrieval delegates ingestion to the retriever's rebuild()."""
    config = make_config(include_rag=True, retrieval="llamaindex")
    generate_scripts_layer(config, tmp_path)

    content = (tmp_path / "scripts" / "ingest.py").read_text()
    assert "retriever.rebuild()" in content
    assert "DocumentLoader" not in content
    assert "vectorstore" not in content


@pytest.mark.unit
def test_no_rag_script_has_placeholder(tmp_path, make_config):
    config = make_config(include_rag=False)
    generate_scripts_layer(config, tmp_path)

    content = (tmp_path / "scripts" / "ingest.py").read_text()
    assert "RAG not enabled" in content
