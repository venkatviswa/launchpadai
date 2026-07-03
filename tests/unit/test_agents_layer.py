"""Unit tests for agents_layer generator."""
import ast
import json
import pytest
from launchpadai.generators.agents_layer import generate_agents_layer


FRAMEWORKS = ["plain", "langgraph", "crewai", "agentscript"]

# Framework-specific orchestration module inside agents/
EXPECTED_FILES = {
    "plain": "pipeline.py",
    "langgraph": "graph.py",
    "crewai": "crew.py",
    "agentscript": "client.py",
}

MULTI_AGENT_FRAMEWORKS = ["plain", "langgraph", "crewai"]


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("include_rag", [True, False])
@pytest.mark.parametrize("include_guardrails", [True, False])
def test_generates_valid_python(tmp_path, make_config, framework, include_rag, include_guardrails):
    config = make_config(framework=framework, include_rag=include_rag, include_guardrails=include_guardrails)
    generate_agents_layer(config, tmp_path)

    for py_file in (tmp_path / "agents").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generates_correct_file(tmp_path, make_config, framework):
    config = make_config(framework=framework)
    generate_agents_layer(config, tmp_path)

    expected = EXPECTED_FILES[framework]
    assert (tmp_path / "agents" / expected).exists(), f"Expected agents/{expected} for {framework}"
    assert (tmp_path / "agents" / "__init__.py").exists()


# ---------------------------------------------------------------------------
# Per-agent slices (framework-agnostic)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_default_agent_slice_generated(tmp_path, make_config, framework):
    """Default config has a single 'assistant' agent slice."""
    config = make_config(framework=framework)
    generate_agents_layer(config, tmp_path)

    slice_dir = tmp_path / "agents" / "assistant"
    assert (slice_dir / "__init__.py").exists()
    assert (slice_dir / "prompts" / "system.md").exists()
    assert (slice_dir / "tools.py").exists()


@pytest.mark.unit
@pytest.mark.parametrize("framework", MULTI_AGENT_FRAMEWORKS)
@pytest.mark.parametrize("orchestration", ["sequential", "supervisor"])
def test_multi_agent_slices_generated(tmp_path, make_config, framework, orchestration):
    config = make_config(framework=framework, orchestration=orchestration)
    generate_agents_layer(config, tmp_path)

    for name in ("researcher", "writer"):
        slice_dir = tmp_path / "agents" / name
        assert (slice_dir / "__init__.py").exists(), f"Missing slice for {name}"
        assert (slice_dir / "prompts" / "system.md").exists()
        assert (slice_dir / "tools.py").exists()


@pytest.mark.unit
def test_slice_prompt_contains_role_and_goal(tmp_path, make_config):
    config = make_config(orchestration="sequential")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "researcher" / "prompts" / "system.md").read_text()
    assert "Research Analyst" in content
    assert "Find accurate information" in content


@pytest.mark.unit
def test_slice_tools_defines_empty_tools_list(tmp_path, make_config):
    config = make_config()
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "assistant" / "tools.py").read_text()
    assert "from tools.registry import Tool" in content
    assert "TOOLS: list[Tool] = []" in content


# ---------------------------------------------------------------------------
# Uniform entrypoint (agents/__init__.py)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_entrypoint_uniform_interface(tmp_path, make_config, framework):
    config = make_config(framework=framework)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "__init__.py").read_text()
    assert "class AgentEntrypoint" in content
    assert "agent = AgentEntrypoint()" in content
    assert "def run(self, user_message: str, session_id: str = \"default\")" in content
    assert "def reset(self, session_id: str = \"default\")" in content


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_entrypoint_includes_guardrails_when_enabled(tmp_path, make_config, framework):
    config = make_config(framework=framework, include_guardrails=True)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "__init__.py").read_text()
    assert "from guardrails.input_filters import check_input" in content
    assert "from guardrails.output_filters import check_output" in content


@pytest.mark.unit
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_entrypoint_excludes_guardrails_when_disabled(tmp_path, make_config, framework):
    config = make_config(framework=framework, include_guardrails=False)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "__init__.py").read_text()
    assert "check_input" not in content
    assert "check_output" not in content


@pytest.mark.unit
@pytest.mark.parametrize("observability", ["langfuse", "langsmith", "opentelemetry"])
def test_entrypoint_imports_tracer_when_observability(tmp_path, make_config, observability):
    config = make_config(observability=observability)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "__init__.py").read_text()
    assert "from observability.tracer import tracer" in content


@pytest.mark.unit
def test_entrypoint_no_tracer_when_observability_none(tmp_path, make_config):
    config = make_config(observability="none")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "__init__.py").read_text()
    assert "from observability.tracer import tracer" not in content


# ---------------------------------------------------------------------------
# Plain framework
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_plain_includes_rag_imports_when_enabled(tmp_path, make_config):
    config = make_config(framework="plain", include_rag=True)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "base.py").read_text()
    assert "from knowledge.retrieval.retriever import retriever" in content
    assert "retriever.retrieve" in content


@pytest.mark.unit
def test_plain_excludes_rag_when_disabled(tmp_path, make_config):
    config = make_config(framework="plain", include_rag=False)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "base.py").read_text()
    assert "retriever" not in content


@pytest.mark.unit
def test_plain_has_agent_class(tmp_path, make_config):
    config = make_config(framework="plain")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "base.py").read_text()
    assert "class Agent" in content
    assert "tool_calls" in content  # OpenAI wire format tool-use loop


@pytest.mark.unit
@pytest.mark.parametrize("orchestration", ["single", "sequential", "supervisor"])
def test_plain_pipeline_structure(tmp_path, make_config, orchestration):
    config = make_config(framework="plain", orchestration=orchestration)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "pipeline.py").read_text()
    assert "AGENT_NAMES" in content
    assert "def run(" in content
    assert "def reset(" in content


@pytest.mark.unit
def test_plain_supervisor_has_routing_prompt(tmp_path, make_config):
    config = make_config(framework="plain", orchestration="supervisor")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "pipeline.py").read_text()
    assert "ROUTING_PROMPT" in content


@pytest.mark.unit
def test_plain_pipeline_lists_configured_agents(tmp_path, make_config):
    config = make_config(framework="plain", orchestration="sequential")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "pipeline.py").read_text()
    assert '"researcher"' in content
    assert '"writer"' in content


# ---------------------------------------------------------------------------
# LangGraph framework
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_langgraph_has_stategraph(tmp_path, make_config):
    config = make_config(framework="langgraph")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "graph.py").read_text()
    assert "StateGraph" in content
    assert "class AgentState" in content
    assert "init_chat_model" in content
    assert "MemorySaver" in content
    assert "START" in content and "END" in content
    assert "def run(" in content


@pytest.mark.unit
def test_langgraph_sequential_chains_agents(tmp_path, make_config):
    config = make_config(framework="langgraph", orchestration="sequential")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "graph.py").read_text()
    assert 'graph.add_edge("researcher", "writer")' in content


@pytest.mark.unit
def test_langgraph_supervisor_has_conditional_edges(tmp_path, make_config):
    config = make_config(framework="langgraph", orchestration="supervisor")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "graph.py").read_text()
    assert "_supervisor" in content
    assert "add_conditional_edges" in content


# ---------------------------------------------------------------------------
# CrewAI framework
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_crewai_has_crew_pattern(tmp_path, make_config):
    config = make_config(framework="crewai")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "crew.py").read_text()
    assert "from crewai import Agent, Crew, LLM, Process, Task" in content
    assert "def run(" in content


@pytest.mark.unit
@pytest.mark.parametrize("llm_provider,prefix", [
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("ollama", "ollama"),
])
def test_crewai_uses_litellm_prefix(tmp_path, make_config, llm_provider, prefix):
    config = make_config(framework="crewai", llm_provider=llm_provider)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "crew.py").read_text()
    assert f'model="{prefix}/" + settings.LLM_MODEL' in content


@pytest.mark.unit
def test_crewai_one_agent_and_task_per_spec(tmp_path, make_config):
    config = make_config(framework="crewai", orchestration="sequential")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "crew.py").read_text()
    assert "researcher = Agent(" in content
    assert "writer = Agent(" in content
    assert "researcher_task = Task(" in content
    assert "writer_task = Task(" in content


@pytest.mark.unit
def test_crewai_supervisor_uses_hierarchical_process(tmp_path, make_config):
    config = make_config(framework="crewai", orchestration="supervisor")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "crew.py").read_text()
    assert "Process.hierarchical" in content
    assert "manager_llm" in content


@pytest.mark.unit
@pytest.mark.parametrize("orchestration", ["single", "sequential"])
def test_crewai_non_supervisor_uses_sequential_process(tmp_path, make_config, orchestration):
    config = make_config(framework="crewai", orchestration=orchestration)
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "crew.py").read_text()
    assert "Process.sequential" in content
    assert "manager_llm" not in content


# ---------------------------------------------------------------------------
# AgentScript framework
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_agentscript_has_client(tmp_path, make_config):
    config = make_config(framework="agentscript")
    generate_agents_layer(config, tmp_path)

    content = (tmp_path / "agents" / "client.py").read_text()
    assert "class AgentforceClient" in content
    assert "client = AgentforceClient()" in content
    assert "def run" in content


@pytest.mark.unit
def test_agentscript_has_agent_file(tmp_path, make_config):
    config = make_config(framework="agentscript")
    generate_agents_layer(config, tmp_path)

    agent_dirs = list((tmp_path / "force-app" / "main" / "aiAuthoringBundles").iterdir())
    assert len(agent_dirs) == 1
    agent_files = list(agent_dirs[0].glob("*.agent"))
    assert len(agent_files) == 1
    content = agent_files[0].read_text()
    assert "system:" in content
    assert "topic main_assistant:" in content
    assert "start_agent:" in content


@pytest.mark.unit
def test_agentscript_has_sfdx_project(tmp_path, make_config):
    config = make_config(framework="agentscript")
    generate_agents_layer(config, tmp_path)

    assert (tmp_path / "sfdx-project.json").exists()
    sfdx = json.loads((tmp_path / "sfdx-project.json").read_text())
    assert "packageDirectories" in sfdx


@pytest.mark.unit
def test_agentscript_includes_rag_action_when_enabled(tmp_path, make_config):
    config = make_config(framework="agentscript", include_rag=True)
    generate_agents_layer(config, tmp_path)

    agent_dirs = list((tmp_path / "force-app" / "main" / "aiAuthoringBundles").iterdir())
    content = list(agent_dirs[0].glob("*.agent"))[0].read_text()
    assert "retrieve_knowledge" in content


@pytest.mark.unit
def test_agentscript_includes_guardrail_instructions_when_enabled(tmp_path, make_config):
    config = make_config(framework="agentscript", include_guardrails=True)
    generate_agents_layer(config, tmp_path)

    agent_dirs = list((tmp_path / "force-app" / "main" / "aiAuthoringBundles").iterdir())
    content = list(agent_dirs[0].glob("*.agent"))[0].read_text()
    assert "SAFETY" in content
