"""Generate the agents layer — core agent loop and orchestration."""
from pathlib import Path


def generate_agents_layer(config: dict, project_path: Path):
    """Generate agent files based on framework choice."""
    base = project_path / "agents"
    _write(base / "__init__.py", "")

    if config["framework"] == "plain":
        _generate_plain_agent(config, base)
    elif config["framework"] == "langchain":
        _generate_langchain_agent(config, base)
    elif config["framework"] == "llamaindex":
        _generate_llamaindex_agent(config, base)
    elif config["framework"] == "crewai":
        _generate_crewai_agent(config, base)
    else:
        _generate_plain_agent(config, base)


def _generate_plain_agent(config: dict, base: Path):
    """Generate a plain Python agent with explicit loop."""

    rag_imports = ""
    rag_retrieve = ""
    if config["include_rag"]:
        rag_imports = """from knowledge.retrieval.retriever import retriever"""
        rag_retrieve = """
        # Step 1: Retrieve relevant context
        results = retriever.retrieve(user_message)
        context = retriever.format_context(results)"""

    guardrail_imports = ""
    guardrail_input = ""
    guardrail_output = ""
    if config["include_guardrails"]:
        guardrail_imports = """from guardrails.input_filters import check_input
from guardrails.output_filters import check_output"""
        guardrail_input = """
        # Check input safety
        input_check = check_input(user_message)
        if not input_check["safe"]:
            return {"response": input_check["message"], "blocked": True}"""
        guardrail_output = """
        # Check output safety
        output_check = check_output(response_text)
        if not output_check["safe"]:
            response_text = output_check["filtered_text"]"""

    _write(base / "base.py", f'''"""Base Agent — the core reasoning loop.

This is a plain Python agent with no framework dependencies.
The LLM decides whether to respond directly or use tools.
"""
import json
from models.llm.provider import llm
from prompts.templates import build_messages
from tools.registry import registry
from tools.example_tool import *  # noqa: Register example tools
from memory.conversation import ConversationMemory
{rag_imports}
{guardrail_imports}


class Agent:
    """Core agent with tool-use loop."""

    def __init__(self, system_prompt: str = "default", max_iterations: int = 10):
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.memory = ConversationMemory()

    def run(self, user_message: str, session_id: str = "default") -> dict:
        """Process a user message and return a response.

        This is the core agent loop:
        1. Check input guardrails
        2. Retrieve relevant context (RAG)
        3. Build prompt with context + history
        4. Call LLM
        5. If LLM wants to use a tool → execute and loop
        6. If LLM responds with text → check output guardrails and return
        """
        {guardrail_input}
        {rag_retrieve}

        # Build messages with context and conversation history
        history = self.memory.get_history(session_id)
        messages = build_messages(
            user_message=user_message,
            context={'"context"' if config["include_rag"] else '""'},
            system_prompt_name=self.system_prompt,
            conversation_history=history,
        )

        # Agent loop — LLM decides whether to use tools or respond
        tools = registry.list_schemas()
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            response = llm.chat(messages, tools=tools if tools else None)

            # Check if the LLM wants to call a tool
            tool_calls = self._extract_tool_calls(response)

            if tool_calls:
                for tool_call in tool_calls:
                    # Validate tool name is registered before execution (OWASP Agentic AI)
                    tool_name = tool_call.get("name", "")
                    if not registry.has_tool(tool_name):
                        result_str = f"Tool '{{tool_name}}' is not registered."
                    else:
                        try:
                            result = registry.execute(
                                tool_name,
                                **tool_call.get("arguments", {{}}),
                            )
                            result_str = json.dumps(result) if not isinstance(result, str) else result
                        except Exception as e:
                            result_str = f"Tool error: {{type(e).__name__}}"

                    # Add tool call and result to messages
                    messages.append({{"role": "assistant", "content": None, "tool_calls": [tool_call]}})
                    messages.append({{"role": "tool", "content": result_str, "tool_call_id": tool_call.get("id", "")}})

                continue  # Loop back for LLM to process tool result

            # No tool call — extract the text response
            response_text = self._extract_text(response)
            {guardrail_output}

            # Save to conversation memory
            self.memory.add(session_id, "user", user_message)
            self.memory.add(session_id, "assistant", response_text)

            return {{
                "response": response_text,
                "iterations": iterations,
                "session_id": session_id,
            }}

        return {{"response": "I've reached my processing limit. Please try again.", "iterations": iterations}}

    def _extract_tool_calls(self, response) -> list[dict]:
        """Extract tool calls from LLM response (provider-specific)."""
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
                return [
                    {{
                        "id": b.id,
                        "name": b.name,
                        "arguments": b.input,
                    }}
                    for b in tool_blocks
                ]
        return []

    def _extract_text(self, response) -> str:
        """Extract text content from LLM response (provider-specific)."""
        # OpenAI format
        if hasattr(response, "choices"):
            return response.choices[0].message.content or ""
        # Anthropic format
        elif hasattr(response, "content"):
            text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
            return text_blocks[0].text if text_blocks else ""
        return str(response)


# Default agent instance
agent = Agent()
''')


def _generate_langchain_agent(config: dict, base: Path):
    """Generate a LangChain/LangGraph agent."""
    _write(base / "graph.py", '''"""LangGraph Agent — stateful graph-based orchestration.

This defines the agent as a state machine with nodes and edges.
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    """State that flows through the agent graph."""
    messages: Annotated[list, add_messages]
    context: str
    iteration: int


def retrieve_context(state: AgentState) -> dict:
    """Retrieve relevant documents for RAG."""
    # TODO: Implement retrieval from your vector store
    from knowledge.retrieval.retriever import retriever
    last_message = state["messages"][-1].content
    results = retriever.retrieve(last_message)
    context = retriever.format_context(results)
    return {"context": context}


def call_agent(state: AgentState) -> dict:
    """Call the LLM with context and tools."""
    from models.llm.provider import llm
    from prompts.templates import build_messages

    messages = state["messages"]
    context = state.get("context", "")

    # Rebuild with system prompt and context
    full_messages = build_messages(
        user_message=messages[-1].content,
        context=context,
    )

    response = llm.chat(full_messages)
    return {"messages": [response], "iteration": state.get("iteration", 0) + 1}


def should_continue(state: AgentState) -> str:
    """Decide whether to continue the loop or finish."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def execute_tools(state: AgentState) -> dict:
    """Execute tool calls from the LLM."""
    from tools.registry import registry
    import json

    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        try:
            result = registry.execute(tc["name"], **tc["args"])
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tc["id"]})
        except Exception as e:
            results.append({"role": "tool", "content": f"Error: {e}", "tool_call_id": tc["id"]})
    return {"messages": results}


# Build the graph
def create_agent():
    """Create and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", execute_tools)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Default agent
agent = create_agent()
''')


def _generate_llamaindex_agent(config: dict, base: Path):
    """Generate a LlamaIndex agent."""
    _write(base / "agent.py", '''"""LlamaIndex Agent — document-centric agent with query engine."""
# TODO: Configure your LlamaIndex agent
# See https://docs.llamaindex.ai/en/stable/

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def create_agent():
    """Create a LlamaIndex ReAct agent with RAG."""
    # Load documents
    documents = SimpleDirectoryReader("data/documents").load_data()

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine tool
    query_tool = QueryEngineTool(
        query_engine=index.as_query_engine(),
        metadata=ToolMetadata(
            name="knowledge_base",
            description="Search the knowledge base for relevant information.",
        ),
    )

    # Create agent
    agent = ReActAgent.from_tools([query_tool], verbose=True)
    return agent


agent = create_agent()
''')


def _generate_crewai_agent(config: dict, base: Path):
    """Generate a CrewAI agent."""
    _write(base / "crew.py", '''"""CrewAI Agent — multi-agent collaboration."""
# TODO: Configure your CrewAI agents and tasks
# See https://docs.crewai.com/

from crewai import Agent, Task, Crew, Process


# Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate and relevant information to answer questions",
    backstory="You are a thorough researcher who finds precise information.",
    verbose=True,
)

writer = Agent(
    role="Response Writer",
    goal="Write clear, helpful responses based on research findings",
    backstory="You excel at turning research into clear, actionable answers.",
    verbose=True,
)


def run_crew(question: str) -> str:
    """Run the crew to answer a question."""
    research_task = Task(
        description=f"Research the following question: {question}",
        expected_output="Detailed findings with sources",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a clear, concise response based on the research.",
        expected_output="A well-structured answer",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
