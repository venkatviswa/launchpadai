"""Unit tests for requirements.txt generator."""
import pytest
from launchpadai.generators.requirements import (
    generate_requirements,
    LLM_DEPS, FRAMEWORK_DEPS, EMBEDDING_DEPS, VECTORDB_DEPS,
    OBSERVABILITY_DEPS, UI_DEPS, ML_FRAMEWORK_DEPS, DATA_FORMAT_DEPS,
    BASE_DEPS, RAG_DEPS, GUARDRAIL_DEPS, MCP_DEPS,
)


def _parse_requirements(path):
    """Parse requirements.txt into a set of package names (without versions)."""
    content = path.read_text()
    packages = set()
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            name = line.split(">=")[0].split("==")[0].split("<")[0].strip()
            packages.add(name)
    return packages


@pytest.mark.unit
@pytest.mark.parametrize("llm_provider", ["openai", "anthropic", "google", "ollama", "multiple"])
def test_llm_packages_present(tmp_path, make_config, llm_provider):
    config = make_config(llm_provider=llm_provider)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in LLM_DEPS[llm_provider]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {llm_provider}"


@pytest.mark.unit
@pytest.mark.parametrize("framework", ["plain", "langchain", "llamaindex", "crewai", "haystack"])
def test_framework_packages_present(tmp_path, make_config, framework):
    config = make_config(framework=framework)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in FRAMEWORK_DEPS[framework]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {framework}"


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", ["chroma", "pinecone", "weaviate", "qdrant", "pgvector"])
def test_vectordb_packages_present(tmp_path, make_config, vector_db):
    config = make_config(vector_db=vector_db)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in VECTORDB_DEPS[vector_db]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {vector_db}"


@pytest.mark.unit
@pytest.mark.parametrize("embedding_model", ["openai-small", "openai-large", "cohere", "bge-m3", "gte-qwen2", "nomic", "ollama"])
def test_embedding_packages_present(tmp_path, make_config, embedding_model):
    config = make_config(embedding_model=embedding_model)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in EMBEDDING_DEPS[embedding_model]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {embedding_model}"


@pytest.mark.unit
@pytest.mark.parametrize("observability", ["langfuse", "langsmith", "opentelemetry", "none"])
def test_observability_packages(tmp_path, make_config, observability):
    config = make_config(observability=observability)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in OBSERVABILITY_DEPS[observability]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {observability}"


@pytest.mark.unit
@pytest.mark.parametrize("ui", ["streamlit", "gradio", "nextjs", "none"])
def test_ui_packages(tmp_path, make_config, ui):
    config = make_config(ui=ui)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in UI_DEPS[ui]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for UI {ui}"


@pytest.mark.unit
def test_rag_packages_when_enabled(tmp_path, make_config):
    config = make_config(include_rag=True)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in RAG_DEPS:
        assert dep.split(">=")[0] in packages


@pytest.mark.unit
def test_rag_packages_absent_when_disabled(tmp_path, make_config):
    config = make_config(include_rag=False)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in RAG_DEPS:
        assert dep.split(">=")[0] not in packages


@pytest.mark.unit
def test_guardrail_packages(tmp_path, make_config):
    config = make_config(include_guardrails=True)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")
    assert "presidio-analyzer" in packages

    config = make_config(include_guardrails=False)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")
    assert "presidio-analyzer" not in packages


@pytest.mark.unit
@pytest.mark.parametrize("ml_framework", ["sklearn", "pytorch", "xgboost", "transformers"])
def test_ml_framework_packages(tmp_path, make_config, ml_framework):
    config = make_config(include_ml_pipeline=True, ml_framework=ml_framework)
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in ML_FRAMEWORK_DEPS[ml_framework]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {ml_framework}"


@pytest.mark.unit
def test_langchain_llm_integration_packages(tmp_path, make_config):
    """LangChain + specific LLM should include langchain-<provider> package."""
    config = make_config(framework="langchain", llm_provider="anthropic")
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")
    assert "langchain-anthropic" in packages


@pytest.mark.unit
def test_no_duplicate_packages(tmp_path, make_config):
    """Full config should not produce duplicate packages."""
    config = make_config(
        include_rag=True, include_guardrails=True, include_eval=True,
        include_mcp=True, include_notebooks=True, include_data_layer=True,
        include_ml_pipeline=True, include_docker=True,
        ui="streamlit", observability="langfuse",
    )
    generate_requirements(config, tmp_path)

    content = (tmp_path / "requirements.txt").read_text()
    lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
    names = [l.split(">=")[0] for l in lines]
    assert len(names) == len(set(names)), f"Duplicates: {[n for n in names if names.count(n) > 1]}"


@pytest.mark.unit
def test_base_deps_always_present(tmp_path, make_config):
    config = make_config()
    generate_requirements(config, tmp_path)
    packages = _parse_requirements(tmp_path / "requirements.txt")

    for dep in BASE_DEPS:
        assert dep.split(">=")[0] in packages
