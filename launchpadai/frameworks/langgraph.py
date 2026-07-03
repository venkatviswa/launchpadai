"""LangGraph framework adapter — stateful graph orchestration."""
from pathlib import Path

from launchpadai.frameworks._entrypoint import build_entrypoint
from launchpadai.frameworks.base import FrameworkAdapter, write

# init_chat_model provider identifiers per launchpadai llm_provider key
_LC_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "ollama": "ollama",
}


def generate(config, project_path: Path):
    base = project_path / "agents"
    _write_graph(config, base)
    write(
        base / "__init__.py",
        build_entrypoint(
            config,
            inner_import="from agents.graph import run as _run",
            reset_body=(
                "# LangGraph checkpointer state is keyed by thread (session_id);\n"
                "# start a fresh conversation by using a new session_id.\n"
                "pass"
            ),
            framework_note=(
                "The inner orchestration is a LangGraph state graph "
                "(see agents/graph.py)."
            ),
        ),
    )


def _write_graph(config, base: Path):
    agent_names = [a.name for a in config["agents"]]
    orchestration = config["orchestration"]
    provider = _LC_PROVIDERS[config["llm_provider"]]
    names_literal = "[" + ", ".join(f'"{n}"' for n in agent_names) + "]"

    rag_block = ""
    if config["include_rag"]:
        rag_block = '''

def _with_context(prompt: str, query: str) -> str:
    """Append retrieved context to an agent's system prompt."""
    from knowledge.retrieval.retriever import retriever

    results = retriever.retrieve(query)
    context = retriever.format_context(results)
    return prompt + "\\n\\n## Retrieved context\\n" + context
'''
        prompt_expr = '_with_context(prompt, _last_user_message(state))'
    else:
        prompt_expr = "prompt"

    header = f'''"""LangGraph agent graph — {orchestration} orchestration.

Each agent lives in agents/<name>/ with its own prompts/system.md and
tools.py; this module wires them into a graph. State is checkpointed per
session (thread_id = session_id).
"""
import importlib
from pathlib import Path
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config.settings import settings

_AGENTS_DIR = Path(__file__).parent

AGENT_NAMES = {names_literal}

_model = init_chat_model(
    settings.LLM_MODEL,
    model_provider="{provider}",
    temperature=settings.LLM_TEMPERATURE,
)


class AgentState(TypedDict):
    """State flowing through the graph."""

    messages: Annotated[list, add_messages]
    next: str


def _load_prompt(name: str) -> str:
    return (_AGENTS_DIR / name / "prompts" / "system.md").read_text()


def _load_tools(name: str) -> list:
    """Convert the slice's registry-style tools into LangChain tools."""
    module = importlib.import_module(f"agents.{{name}}.tools")
    return [
        StructuredTool.from_function(func=t.func, name=t.name, description=t.description)
        for t in getattr(module, "TOOLS", [])
    ]


def _last_user_message(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""
{rag_block}

def _agent_node(name: str):
    """Build a graph node for one agent: LLM call with its prompt + tool loop."""
    prompt = _load_prompt(name)
    tools = _load_tools(name)
    tools_by_name = {{t.name: t for t in tools}}
    model = _model.bind_tools(tools) if tools else _model

    def node(state: AgentState) -> dict:
        system = SystemMessage(content={prompt_expr})
        messages = [system] + list(state["messages"])
        response = model.invoke(messages)
        new_messages = [response]

        iterations = 0
        while getattr(response, "tool_calls", None) and iterations < 10:
            iterations += 1
            for tc in response.tool_calls:
                tool = tools_by_name.get(tc["name"])
                if tool is None:
                    content = f"Tool '{{tc['name']}}' is not registered."
                else:
                    try:
                        content = str(tool.invoke(tc["args"]))
                    except Exception as e:
                        content = f"Tool error: {{type(e).__name__}}"
                new_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
            response = model.invoke(messages + new_messages)
            new_messages.append(response)

        return {{"messages": new_messages}}

    return node
'''

    if orchestration == "single":
        build_code = f'''

def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("{agent_names[0]}", _agent_node("{agent_names[0]}"))
    graph.add_edge(START, "{agent_names[0]}")
    graph.add_edge("{agent_names[0]}", END)
    return graph.compile(checkpointer=MemorySaver())
'''
    elif orchestration == "sequential":
        edges = ['    graph.add_edge(START, "%s")' % agent_names[0]]
        for a, b in zip(agent_names, agent_names[1:]):
            edges.append(f'    graph.add_edge("{a}", "{b}")')
        edges.append(f'    graph.add_edge("{agent_names[-1]}", END)')
        edges_code = "\n".join(edges)
        build_code = f'''

def _build_graph():
    graph = StateGraph(AgentState)
    for name in AGENT_NAMES:
        graph.add_node(name, _agent_node(name))
{edges_code}
    return graph.compile(checkpointer=MemorySaver())
'''
    else:  # supervisor
        directory = "\\n".join(
            f"- {a.name}: {a.role} — {a.goal}".replace('"', "'")
            for a in config["agents"]
        )
        build_code = f'''

ROUTING_PROMPT = (
    "You are a supervisor routing user requests to the best specialist agent.\\n"
    "Available agents:\\n"
    "{directory}\\n\\n"
    "Respond with ONLY the name of the agent that should handle the request."
)


def _supervisor(state: AgentState) -> dict:
    """Pick the agent that should handle the latest user message."""
    decision = _model.invoke(
        [SystemMessage(content=ROUTING_PROMPT), HumanMessage(content=_last_user_message(state))]
    )
    content = decision.content if isinstance(decision.content, str) else str(decision.content)
    choice = content.strip().lower()
    for name in AGENT_NAMES:
        if name in choice:
            return {{"next": name}}
    return {{"next": AGENT_NAMES[0]}}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _supervisor)
    for name in AGENT_NAMES:
        graph.add_node(name, _agent_node(name))
        graph.add_edge(name, END)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", lambda state: state["next"], {{name: name for name in AGENT_NAMES}}
    )
    return graph.compile(checkpointer=MemorySaver())
'''

    run_code = '''

graph = _build_graph()


def run(user_message: str, session_id: str = "default") -> dict:
    """Invoke the graph for one user turn."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        {"configurable": {"thread_id": session_id}},
    )
    final = result["messages"][-1]
    content = final.content
    if isinstance(content, list):  # e.g. Anthropic content blocks
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return {"response": content}
'''

    write(base / "graph.py", header + build_code + run_code)


ADAPTER = FrameworkAdapter(
    name="langgraph",
    display_name="LangGraph",
    tier=1,
    description="Graph-based orchestration with checkpointed state (LangChain ecosystem)",
    orchestrations=("single", "sequential", "supervisor"),
    generate=generate,
)
