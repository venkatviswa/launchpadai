"""Validation tests — verify requirements.txt has correct packages for the config."""
import pytest
from launchpadai.generators.project import ProjectGenerator
from launchpadai.generators.requirements import (
    LLM_DEPS, FRAMEWORK_DEPS, EMBEDDING_DEPS, VECTORDB_DEPS,
)


FRAMEWORKS = ["plain", "langgraph", "crewai", "agentscript"]
LLM_PROVIDERS = ["openai", "anthropic", "ollama"]


def _dep_name(dep):
    return dep.split(">=")[0].split("==")[0].split("<")[0].strip()


def _parse_requirements(path):
    content = path.read_text()
    packages = set()
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            packages.add(_dep_name(line))
    return packages


def _generate(make_config, tmp_path, **overrides):
    config = make_config(**overrides)
    project_path = tmp_path / "req-check"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()
    return config, project_path


@pytest.mark.validation
@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("llm_provider", LLM_PROVIDERS)
def test_full_project_requirements_match_config(tmp_path, make_config, framework, llm_provider):
    """After full generation, requirements.txt must include correct stack packages."""
    config, project_path = _generate(
        make_config, tmp_path, framework=framework, llm_provider=llm_provider,
    )
    packages = _parse_requirements(project_path / "requirements.txt")

    # API stack is always present
    for dep in ("fastapi", "uvicorn", "pydantic"):
        assert dep in packages, f"Missing always-on dep {dep}"

    # Framework deps
    for dep in FRAMEWORK_DEPS[framework]:
        assert _dep_name(dep) in packages, f"Missing {_dep_name(dep)} for {framework}"

    # LLM deps
    for dep in LLM_DEPS[llm_provider]:
        assert _dep_name(dep) in packages, f"Missing {_dep_name(dep)} for {llm_provider}"

    # langgraph adapter needs the langchain provider package too
    if framework == "langgraph":
        assert f"langchain-{llm_provider}" in packages

    # Default config: RAG on with custom retrieval → embedding + vector db deps
    assert "tiktoken" in packages
    for dep in EMBEDDING_DEPS[config["embedding_model"]]:
        assert _dep_name(dep) in packages, f"Missing {_dep_name(dep)} embedding dep"
    for dep in VECTORDB_DEPS[config["vector_db"]]:
        assert _dep_name(dep) in packages, f"Missing {_dep_name(dep)} for {config['vector_db']}"


@pytest.mark.validation
def test_framework_marker_deps(tmp_path, make_config):
    """Each framework pulls in its signature package."""
    expectations = {
        "plain": set(),
        "langgraph": {"langgraph", "langchain", "langchain-core"},
        "crewai": {"crewai"},
        "agentscript": {"simple-salesforce"},
    }
    for framework, expected in expectations.items():
        config = make_config(framework=framework)
        project_path = tmp_path / f"fw-{framework}"
        project_path.mkdir()
        ProjectGenerator(config, project_path).generate()
        packages = _parse_requirements(project_path / "requirements.txt")
        missing = expected - packages
        assert not missing, f"Missing {missing} for {framework}"


@pytest.mark.validation
def test_no_rag_omits_rag_stack(tmp_path, make_config):
    """Without RAG there is no tiktoken, no vector db, no embedding stack."""
    _, project_path = _generate(make_config, tmp_path, include_rag=False)
    packages = _parse_requirements(project_path / "requirements.txt")

    assert "tiktoken" not in packages
    assert "chromadb" not in packages
    assert "pinecone" not in packages
    assert "sentence-transformers" not in packages
    assert "llama-index-core" not in packages
    # API stack still always present
    assert {"fastapi", "uvicorn", "pydantic"} <= packages


@pytest.mark.validation
@pytest.mark.parametrize("embedding_model,expected_llamaindex_pkg", [
    ("openai-small", "llama-index-embeddings-openai"),
    ("openai-large", "llama-index-embeddings-openai"),
    ("bge-m3", "llama-index-embeddings-huggingface"),
    ("nomic", "llama-index-embeddings-huggingface"),
])
def test_llamaindex_retrieval_deps(tmp_path, make_config, embedding_model, expected_llamaindex_pkg):
    """retrieval='llamaindex' swaps the custom RAG stack for llama-index packages."""
    _, project_path = _generate(
        make_config, tmp_path,
        include_rag=True, retrieval="llamaindex", embedding_model=embedding_model,
    )
    packages = _parse_requirements(project_path / "requirements.txt")

    assert "llama-index-core" in packages
    assert expected_llamaindex_pkg in packages
    # LlamaIndex owns embedding + storage — no custom stack deps
    assert "chromadb" not in packages
    assert "pinecone" not in packages
    assert "sentence-transformers" not in packages


@pytest.mark.validation
def test_pinecone_dep_is_modern_package_name(tmp_path, make_config):
    """The pinecone dependency is 'pinecone', not the legacy 'pinecone-client'."""
    _, project_path = _generate(make_config, tmp_path, vector_db="pinecone")
    packages = _parse_requirements(project_path / "requirements.txt")

    assert "pinecone" in packages
    assert "pinecone-client" not in packages


@pytest.mark.validation
def test_langfuse_pinned_below_v3(tmp_path, make_config):
    """Generated tracer targets the langfuse v2 SDK, so the dep must be pinned <3."""
    _, project_path = _generate(make_config, tmp_path, observability="langfuse")
    content = (project_path / "requirements.txt").read_text()

    langfuse_lines = [
        line for line in content.splitlines()
        if line.strip().startswith("langfuse")
    ]
    assert langfuse_lines, "langfuse dep missing"
    assert any("<3.0.0" in line for line in langfuse_lines), \
        f"langfuse must be pinned <3.0.0, got: {langfuse_lines}"
