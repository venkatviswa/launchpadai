"""Plain Python framework adapter — explicit agent loop, no framework dependencies."""
from pathlib import Path

from launchpadai.frameworks._entrypoint import build_entrypoint
from launchpadai.frameworks.base import FrameworkAdapter, write


def generate(config, project_path: Path):
    base = project_path / "agents"
    _write_agent_class(config, base)
    _write_pipeline(config, base)
    write(
        base / "__init__.py",
        build_entrypoint(
            config,
            inner_import="from agents.pipeline import run as _run",
            reset_body="from agents.pipeline import reset as _reset\n_reset(session_id)",
            framework_note=(
                "The inner orchestration is a plain Python agent loop "
                "(see agents/base.py and agents/pipeline.py)."
            ),
        ),
    )


def _write_agent_class(config, base: Path):
    rag_imports = ""
    rag_retrieve = ""
    rag_context = ""
    if config["include_rag"]:
        rag_imports = "from knowledge.retrieval.retriever import retriever\n"
        rag_retrieve = """
        # Retrieve relevant context for this agent's input
        results = retriever.retrieve(user_message)
        context = retriever.format_context(results)
        system_prompt = self.system_prompt + "\\n\\n## Retrieved context\\n" + context
"""
        rag_context = "system_prompt"
    else:
        rag_retrieve = """
        system_prompt = self.system_prompt
"""
        rag_context = "system_prompt"

    write(base / "base.py", f'''"""Base Agent — the core reasoning loop.

A plain Python agent with no framework dependencies. Each agent has its own
system prompt and tool list; the LLM decides whether to respond directly or
use tools.
"""
import json

from models.llm.provider import llm
from prompts.templates import build_messages
from tools.registry import Tool, registry
from tools.example_tool import *  # noqa: F401,F403 — registers shared example tools
from memory.conversation import ConversationMemory
{rag_imports}

class Agent:
    """One agent with its own prompt, tools, and tool-use loop."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.max_iterations = max_iterations
        self.memory = ConversationMemory()

    def _all_tools(self) -> dict[str, Tool]:
        """Shared registry tools plus this agent's own tools."""
        tools = {{name: registry.get(name) for name in registry.list_names()}}
        tools.update({{t.name: t for t in self.tools}})
        return tools

    def run(self, user_message: str, session_id: str = "default") -> dict:
        """Process a user message and return {{"response": ...}}.

        The loop: retrieve context (RAG), build messages, call the LLM,
        execute any requested tools, repeat until the LLM answers in text.
        """{rag_retrieve}
        history = self.memory.get_history(session_id)
        messages = build_messages(
            user_message=user_message,
            system_prompt={rag_context},
            conversation_history=history,
        )

        available = self._all_tools()
        tool_schemas = [t.to_schema() for t in available.values()]
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            response = llm.chat(messages, tools=tool_schemas if tool_schemas else None)
            tool_calls = self._extract_tool_calls(response)

            if tool_calls:
                # Echo the assistant tool-call turn in OpenAI wire format;
                # providers adapt this internally (see models/llm/provider.py).
                messages.append({{
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {{
                            "id": tc["id"],
                            "type": "function",
                            "function": {{
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            }},
                        }}
                        for tc in tool_calls
                    ],
                }})
                for tc in tool_calls:
                    tool = available.get(tc["name"])
                    if tool is None:
                        result_str = f"Tool '{{tc['name']}}' is not registered."
                    else:
                        try:
                            result = tool.execute(**tc["arguments"])
                            result_str = result if isinstance(result, str) else json.dumps(result)
                        except Exception as e:
                            result_str = f"Tool error: {{type(e).__name__}}"
                    messages.append({{
                        "role": "tool",
                        "content": result_str,
                        "tool_call_id": tc["id"],
                    }})
                continue  # loop back so the LLM can process tool results

            response_text = self._extract_text(response)

            self.memory.add(session_id, "user", user_message)
            self.memory.add(session_id, "assistant", response_text)

            return {{
                "response": response_text,
                "iterations": iterations,
                "agent": self.name,
            }}

        return {{
            "response": "I've reached my processing limit. Please try again.",
            "iterations": iterations,
            "agent": self.name,
        }}

    def _extract_tool_calls(self, response) -> list[dict]:
        """Extract tool calls from an LLM response into {{id, name, arguments}} dicts."""
        # Ollama format (plain dict)
        if isinstance(response, dict):
            calls = response.get("message", {{}}).get("tool_calls") or []
            return [
                {{
                    "id": tc.get("id", f"call_{{i}}"),
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }}
                for i, tc in enumerate(calls)
            ]
        # OpenAI format
        if hasattr(response, "choices"):
            msg = response.choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return [
                    {{
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }}
                    for tc in msg.tool_calls
                ]
        # Anthropic format
        elif hasattr(response, "content"):
            tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if tool_blocks:
                return [{{"id": b.id, "name": b.name, "arguments": b.input}} for b in tool_blocks]
        return []

    def _extract_text(self, response) -> str:
        """Extract text content from an LLM response."""
        # Ollama format (plain dict)
        if isinstance(response, dict):
            return response.get("message", {{}}).get("content", "")
        # OpenAI format
        if hasattr(response, "choices"):
            return response.choices[0].message.content or ""
        # Anthropic format
        elif hasattr(response, "content"):
            text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
            return text_blocks[0].text if text_blocks else ""
        return str(response)
''')


def _write_pipeline(config, base: Path):
    agent_names = [a.name for a in config["agents"]]
    orchestration = config["orchestration"]
    names_literal = "[" + ", ".join(f'"{n}"' for n in agent_names) + "]"

    header = f'''"""Agent pipeline — builds agents from their slices and orchestrates them.

Orchestration mode: {orchestration}
Each agent lives in agents/<name>/ with its own prompts/system.md and tools.py.
"""
import importlib
from pathlib import Path

from agents.base import Agent

_AGENTS_DIR = Path(__file__).parent

AGENT_NAMES = {names_literal}


def _load_prompt(name: str) -> str:
    return (_AGENTS_DIR / name / "prompts" / "system.md").read_text()


def _load_tools(name: str) -> list:
    module = importlib.import_module(f"agents.{{name}}.tools")
    return getattr(module, "TOOLS", [])


AGENTS = {{
    name: Agent(name=name, system_prompt=_load_prompt(name), tools=_load_tools(name))
    for name in AGENT_NAMES
}}


def reset(session_id: str = "default"):
    """Clear conversation memory for every agent in this session."""
    for name, agent_obj in AGENTS.items():
        agent_obj.memory.clear(session_id)
        agent_obj.memory.clear(f"{{session_id}}:{{name}}")
'''

    if orchestration == "single":
        run_code = f'''

def run(user_message: str, session_id: str = "default") -> dict:
    """Run the single configured agent."""
    return AGENTS["{agent_names[0]}"].run(user_message, session_id=session_id)
'''
    elif orchestration == "sequential":
        run_code = '''

def run(user_message: str, session_id: str = "default") -> dict:
    """Run agents in order; each receives the original request plus the previous output."""
    message = user_message
    steps = []
    for name in AGENT_NAMES:
        result = AGENTS[name].run(message, session_id=f"{session_id}:{name}")
        steps.append({"agent": name, "response": result["response"]})
        message = (
            f"Original request:\\n{user_message}\\n\\n"
            f"Output from the previous agent ({name}):\\n{result['response']}"
        )
    return {"response": steps[-1]["response"], "steps": steps}
'''
    else:  # supervisor
        directory = "\\n".join(
            f"- {a.name}: {a.role} — {a.goal}".replace('"', "'")
            for a in config["agents"]
        )
        run_code = f'''

ROUTING_PROMPT = (
    "You are a supervisor routing user requests to the best specialist agent.\\n"
    "Available agents:\\n"
    "{directory}\\n\\n"
    "Respond with ONLY the name of the agent that should handle the request."
)


def _route(user_message: str) -> str:
    from models.llm.provider import llm

    choice = llm.simple(ROUTING_PROMPT + "\\n\\nUser request: " + user_message)
    choice = choice.strip().lower()
    for name in AGENT_NAMES:
        if name in choice:
            return name
    return AGENT_NAMES[0]


def run(user_message: str, session_id: str = "default") -> dict:
    """Route the request to the most suitable agent, then run it."""
    name = _route(user_message)
    result = AGENTS[name].run(user_message, session_id=session_id)
    result["agent"] = name
    return result
'''

    write(base / "pipeline.py", header + run_code)


ADAPTER = FrameworkAdapter(
    name="plain",
    display_name="Plain Python",
    tier=1,
    description="No framework — explicit, dependency-free agent loop with full control",
    orchestrations=("single", "sequential", "supervisor"),
    generate=generate,
)
