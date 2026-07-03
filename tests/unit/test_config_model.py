"""Unit tests for the ProjectConfig model and the framework adapter registry."""
import pytest
from pydantic import ValidationError

from launchpadai.config import AgentSpec, ProjectConfig
from launchpadai.frameworks.registry import framework_names, get_adapter, validate_config
from launchpadai.generators.project import ProjectGenerator


TWO_AGENTS = [
    {"name": "researcher", "role": "Research Analyst", "goal": "Find accurate information"},
    {"name": "writer", "role": "Response Writer", "goal": "Write clear answers"},
]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_defaults_single_assistant_agent():
    config = ProjectConfig(project_name="p")
    assert len(config.agents) == 1
    assert config.agents[0].name == "assistant"
    assert config.agents[0].goal == config.agent_description


@pytest.mark.unit
def test_orchestration_defaults_to_single_for_one_agent():
    config = ProjectConfig(project_name="p")
    assert config.orchestration == "single"


@pytest.mark.unit
def test_orchestration_defaults_to_sequential_for_two_agents():
    config = ProjectConfig(project_name="p", agents=TWO_AGENTS)
    assert config.orchestration == "sequential"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_duplicate_agent_names_rejected():
    dupes = [
        {"name": "worker", "role": "A", "goal": "x"},
        {"name": "worker", "role": "B", "goal": "y"},
    ]
    with pytest.raises(ValidationError, match="unique"):
        ProjectConfig(project_name="p", agents=dupes)


@pytest.mark.unit
def test_single_orchestration_with_two_agents_rejected():
    with pytest.raises(ValidationError, match="exactly one agent"):
        ProjectConfig(project_name="p", agents=TWO_AGENTS, orchestration="single")


@pytest.mark.unit
@pytest.mark.parametrize("orchestration", ["sequential", "supervisor"])
def test_multi_agent_orchestration_with_one_agent_rejected(orchestration):
    with pytest.raises(ValidationError, match="at least two agents"):
        ProjectConfig(project_name="p", orchestration=orchestration)


@pytest.mark.unit
@pytest.mark.parametrize("ui", ["streamlit", "gradio", "none"])
def test_oauth_requires_nextjs(ui):
    with pytest.raises(ValidationError):
        ProjectConfig(project_name="p", ui=ui, auth="oauth")


@pytest.mark.unit
def test_oauth_with_nextjs_accepted():
    config = ProjectConfig(project_name="p", ui="nextjs", auth="oauth")
    assert config.auth == "oauth"


@pytest.mark.unit
@pytest.mark.parametrize("auth", ["simple", "multi_user"])
def test_auth_without_ui_rejected(auth):
    with pytest.raises(ValidationError, match="auth requires a UI"):
        ProjectConfig(project_name="p", ui="none", auth=auth)


@pytest.mark.unit
@pytest.mark.parametrize("llm_provider", ["google", "multiple"])
def test_removed_llm_providers_rejected(llm_provider):
    with pytest.raises(ValidationError):
        ProjectConfig(project_name="p", llm_provider=llm_provider)


@pytest.mark.unit
@pytest.mark.parametrize("embedding_model", ["cohere", "ollama"])
def test_removed_embedding_models_rejected(embedding_model):
    with pytest.raises(ValidationError):
        ProjectConfig(project_name="p", embedding_model=embedding_model)


@pytest.mark.unit
@pytest.mark.parametrize("vector_db", ["weaviate", "qdrant", "pgvector"])
def test_removed_vector_dbs_rejected(vector_db):
    with pytest.raises(ValidationError):
        ProjectConfig(project_name="p", vector_db=vector_db)


@pytest.mark.unit
def test_agent_spec_name_must_be_slug():
    with pytest.raises(ValidationError):
        AgentSpec(name="Not A Slug", role="R", goal="G")


# ---------------------------------------------------------------------------
# Dict-style compatibility
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dict_style_access(make_config):
    config = make_config(framework="langgraph")
    assert config["framework"] == "langgraph"
    assert "framework" in config
    assert "nonexistent_key" not in config
    assert config.get("framework") == "langgraph"
    assert config.get("nonexistent_key") is None
    with pytest.raises(KeyError):
        config["nonexistent_key"]


@pytest.mark.unit
def test_to_dict_round_trip(make_config):
    config = make_config(framework="crewai", orchestration="supervisor")
    restored = ProjectConfig.model_validate(config.to_dict())
    assert restored == config
    assert [a.name for a in restored.agents] == ["researcher", "writer"]


# ---------------------------------------------------------------------------
# Framework adapter registry
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_registry_contains_all_adapters():
    names = framework_names()
    assert set(names) == {"plain", "langgraph", "crewai", "agentscript"}


@pytest.mark.unit
def test_removed_frameworks_not_registered():
    for removed in ("langchain", "llamaindex", "haystack"):
        assert removed not in framework_names()


@pytest.mark.unit
def test_get_adapter_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_adapter("nope")


@pytest.mark.unit
@pytest.mark.parametrize("orchestration", ["sequential", "supervisor"])
def test_validate_config_rejects_agentscript_multi_agent(make_config, orchestration):
    config = make_config(framework="agentscript", orchestration=orchestration)
    with pytest.raises(ValueError, match="does not support"):
        validate_config(config)


@pytest.mark.unit
@pytest.mark.parametrize("framework", ["plain", "langgraph", "crewai"])
@pytest.mark.parametrize("orchestration", ["single", "sequential", "supervisor"])
def test_validate_config_accepts_tier1_orchestrations(make_config, framework, orchestration):
    config = make_config(framework=framework, orchestration=orchestration)
    validate_config(config)  # should not raise


@pytest.mark.unit
def test_project_generator_rejects_unknown_framework(tmp_path, make_config):
    config = make_config(framework="haystack")
    with pytest.raises(KeyError):
        ProjectGenerator(config, tmp_path)


@pytest.mark.unit
def test_project_generator_rejects_agentscript_multi_agent(tmp_path, make_config):
    config = make_config(framework="agentscript", orchestration="sequential")
    with pytest.raises(ValueError):
        ProjectGenerator(config, tmp_path)
