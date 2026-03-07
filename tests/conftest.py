"""Shared test fixtures for LaunchpadAI test suite."""
import pytest
from pathlib import Path
from allpairspy import AllPairs

from launchpadai.generators.project import ProjectGenerator


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "project_name": "test-project",
    "framework": "plain",
    "llm_provider": "openai",
    "embedding_model": "openai-small",
    "vector_db": "chroma",
    "include_rag": True,
    "include_guardrails": False,
    "include_eval": False,
    "include_mcp": False,
    "observability": "none",
    "ui": "none",
    "auth": "none",
    "include_notebooks": False,
    "include_data_layer": False,
    "data_format": "csv",
    "include_ml_pipeline": False,
    "ml_framework": "sklearn",
    "include_docker": False,
    "agent_description": "A test AI assistant",
}


@pytest.fixture
def make_config():
    """Factory that produces valid config dicts with overrides.

    Applies the same conditional fallbacks as the real CLI:
    - auth → "none" when ui is "none"
    - auth "oauth" → "multi_user" when ui is not "nextjs"
    """
    def _make(**overrides):
        config = {**DEFAULT_CONFIG, **overrides}
        if config["ui"] == "none":
            config["auth"] = "none"
        if config["auth"] == "oauth" and config["ui"] != "nextjs":
            config["auth"] = "multi_user"
        return config
    return _make


# ---------------------------------------------------------------------------
# Project generation fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def generated_project(tmp_path, make_config):
    """Generate a full project into tmp_path and return (config, path)."""
    def _generate(**overrides):
        config = make_config(**overrides)
        project_path = tmp_path / config["project_name"]
        project_path.mkdir(parents=True, exist_ok=True)
        generator = ProjectGenerator(config, project_path)
        generator.generate()
        return config, project_path
    return _generate


# ---------------------------------------------------------------------------
# Pairwise combinatorial configs
# ---------------------------------------------------------------------------

_PAIRWISE_KEYS = [
    "framework",
    "llm_provider",
    "embedding_model",
    "vector_db",
    "include_rag",
    "include_guardrails",
    "include_eval",
    "include_mcp",
    "observability",
    "ui",
    "auth",
    "include_notebooks",
    "include_data_layer",
    "data_format",
    "include_ml_pipeline",
    "ml_framework",
    "include_docker",
]

_PAIRWISE_PARAMETERS = [
    ["plain", "langchain", "llamaindex", "crewai", "haystack"],
    ["openai", "anthropic", "google", "ollama", "multiple"],
    ["openai-small", "openai-large", "cohere", "bge-m3", "gte-qwen2", "nomic", "ollama"],
    ["chroma", "pinecone", "weaviate", "qdrant", "pgvector"],
    [True, False],   # include_rag
    [True, False],   # include_guardrails
    [True, False],   # include_eval
    [True, False],   # include_mcp
    ["langfuse", "langsmith", "opentelemetry", "none"],
    ["streamlit", "gradio", "nextjs", "none"],
    ["none", "simple", "multi_user", "oauth"],
    [True, False],   # include_notebooks
    [True, False],   # include_data_layer
    ["csv", "parquet", "json", "sql"],
    [True, False],   # include_ml_pipeline
    ["sklearn", "pytorch", "xgboost", "transformers"],
    [True, False],   # include_docker
]


def _is_valid_pairwise(row):
    """Constraint filter for conditional config dependencies."""
    if len(row) > 10:
        ui, auth = row[9], row[10]
        if ui == "none" and auth != "none":
            return False
        if auth == "oauth" and ui != "nextjs":
            return False
    if len(row) > 12:
        notebooks, data_layer = row[11], row[12]
        if not notebooks and data_layer:
            return False
    if len(row) > 13:
        data_layer, data_format = row[12], row[13]
        if not data_layer and data_format != "csv":
            return False
    if len(row) > 15:
        ml_pipeline, ml_framework = row[14], row[15]
        if not ml_pipeline and ml_framework != "sklearn":
            return False
    return True


def _generate_pairwise_configs():
    """Generate pairwise test configurations."""
    configs = []
    for i, row in enumerate(AllPairs(_PAIRWISE_PARAMETERS, filter_func=_is_valid_pairwise)):
        config = {**DEFAULT_CONFIG}
        for key, value in zip(_PAIRWISE_KEYS, row):
            config[key] = value
        config["project_name"] = f"pw-project-{i}"
        configs.append(config)
    return configs


PAIRWISE_CONFIGS = _generate_pairwise_configs()
