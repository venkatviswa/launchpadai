"""Shared test fixtures for LaunchpadAI test suite."""
import pytest
from pathlib import Path
from allpairspy import AllPairs

from launchpadai.config import AgentSpec, ProjectConfig
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
    "retrieval": "custom",
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
    "orchestration": "single",
}

MULTI_AGENT_SPECS = [
    {"name": "researcher", "role": "Research Analyst", "goal": "Find accurate information"},
    {"name": "writer", "role": "Response Writer", "goal": "Write clear answers"},
]


@pytest.fixture
def make_config():
    """Factory that produces validated ProjectConfig objects with overrides.

    Applies the same conditional fallbacks as the real CLI:
    - auth → "none" when ui is "none"
    - auth "oauth" → "multi_user" when ui is not "nextjs"
    - multi-agent orchestration without explicit agents → two default agents
    """
    def _make(**overrides):
        config = {**DEFAULT_CONFIG, **overrides}
        if config["ui"] == "none":
            config["auth"] = "none"
        if config["auth"] == "oauth" and config["ui"] != "nextjs":
            config["auth"] = "multi_user"
        if config.get("orchestration") in ("sequential", "supervisor") and not config.get("agents"):
            config["agents"] = MULTI_AGENT_SPECS
        return ProjectConfig.model_validate(config)
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
    "retrieval",
    "orchestration",
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
    ["plain", "langgraph", "crewai", "agentscript"],
    ["openai", "anthropic", "ollama"],
    ["openai-small", "openai-large", "bge-m3", "gte-qwen2", "nomic"],
    ["chroma", "pinecone"],
    ["custom", "llamaindex"],
    ["single", "sequential", "supervisor"],
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

_KEY_INDEX = {key: i for i, key in enumerate(_PAIRWISE_KEYS)}


def _is_valid_pairwise(row):
    """Constraint filter for conditional config dependencies."""
    def val(key):
        idx = _KEY_INDEX[key]
        return row[idx] if len(row) > idx else None

    framework, orchestration = val("framework"), val("orchestration")
    if framework == "agentscript" and orchestration not in (None, "single"):
        return False
    ui, auth = val("ui"), val("auth")
    if ui is not None and auth is not None:
        if ui == "none" and auth != "none":
            return False
        if auth == "oauth" and ui != "nextjs":
            return False
    notebooks, data_layer = val("include_notebooks"), val("include_data_layer")
    if notebooks is not None and data_layer is not None and not notebooks and data_layer:
        return False
    data_format = val("data_format")
    if data_layer is not None and data_format is not None and not data_layer and data_format != "csv":
        return False
    ml_pipeline, ml_framework = val("include_ml_pipeline"), val("ml_framework")
    if ml_pipeline is not None and ml_framework is not None and not ml_pipeline and ml_framework != "sklearn":
        return False
    return True


def _generate_pairwise_configs():
    """Generate pairwise test configurations as validated ProjectConfig objects."""
    configs = []
    for i, row in enumerate(AllPairs(_PAIRWISE_PARAMETERS, filter_func=_is_valid_pairwise)):
        config = {**DEFAULT_CONFIG}
        for key, value in zip(_PAIRWISE_KEYS, row):
            config[key] = value
        config["project_name"] = f"pw-project-{i}"
        if config["ui"] == "none":
            config["auth"] = "none"
        if config["orchestration"] in ("sequential", "supervisor"):
            config["agents"] = MULTI_AGENT_SPECS
        configs.append(ProjectConfig.model_validate(config))
    return configs


PAIRWISE_CONFIGS = _generate_pairwise_configs()
