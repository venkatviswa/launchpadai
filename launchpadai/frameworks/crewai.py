"""CrewAI framework adapter — role/task multi-agent crews."""
from pathlib import Path

from launchpadai.frameworks._entrypoint import build_entrypoint
from launchpadai.frameworks.base import FrameworkAdapter, write

# LiteLLM model prefixes per launchpadai llm_provider key (CrewAI uses LiteLLM)
_LITELLM_PREFIXES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "ollama": "ollama",
}


def generate(config, project_path: Path):
    base = project_path / "agents"
    _write_crew(config, base)
    write(
        base / "__init__.py",
        build_entrypoint(
            config,
            inner_import="from agents.crew import run as _run",
            reset_body=(
                "# CrewAI crews are built per run; there is no per-session state to clear.\n"
                "pass"
            ),
            framework_note=(
                "The inner orchestration is a CrewAI crew built from the "
                "project's agent definitions (see agents/crew.py)."
            ),
        ),
    )


def _write_crew(config, base: Path):
    orchestration = config["orchestration"]
    prefix = _LITELLM_PREFIXES[config["llm_provider"]]

    process = "Process.hierarchical" if orchestration == "supervisor" else "Process.sequential"
    manager_arg = "\n        manager_llm=_llm," if orchestration == "supervisor" else ""

    agent_defs = []
    task_defs = []
    agent_vars = []
    for i, spec in enumerate(config["agents"]):
        role = spec.role.replace('"', "'")
        goal = spec.goal.replace('"', "'")
        agent_vars.append(spec.name)
        agent_defs.append(f'''
{spec.name} = Agent(
    role="{role}",
    goal="{goal}",
    backstory=_load_prompt("{spec.name}"),
    llm=_llm,
    verbose=True,
)''')
        if i == 0:
            task_desc = 'Handle the following user request: {user_message}'
        else:
            task_desc = (
                "Continue the work on the user request '{user_message}' "
                "using the previous task's output."
            )
        context_arg = f"\n    context=[{agent_vars[i - 1]}_task]," if i > 0 and orchestration == "sequential" else ""
        task_defs.append(f'''
{spec.name}_task = Task(
    description="{task_desc}",
    expected_output="A clear, complete result for this step",
    agent={spec.name},{context_arg}
)''')

    agents_list = ", ".join(agent_vars)
    tasks_list = ", ".join(f"{v}_task" for v in agent_vars)

    write(base / "crew.py", f'''"""CrewAI crew — built from the project's agent definitions.

Orchestration mode: {orchestration}
Each agent lives in agents/<name>/ with its own prompts/system.md; the
system prompt is used as the agent's backstory. To give an agent tools,
wrap the functions in agents/<name>/tools.py with crewai.tools' @tool
decorator and pass them via Agent(tools=[...]).
"""
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task

from config.settings import settings

_AGENTS_DIR = Path(__file__).parent

_llm = LLM(
    model="{prefix}/" + settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
)


def _load_prompt(name: str) -> str:
    return (_AGENTS_DIR / name / "prompts" / "system.md").read_text()

{"".join(agent_defs)}

{"".join(task_defs)}


def _build_crew() -> Crew:
    return Crew(
        agents=[{agents_list}],
        tasks=[{tasks_list}],
        process={process},{manager_arg}
        verbose=True,
    )


def run(user_message: str, session_id: str = "default") -> dict:
    """Run the crew against one user request."""
    crew = _build_crew()
    result = crew.kickoff(inputs={{"user_message": user_message}})
    return {{"response": str(result)}}
''')


ADAPTER = FrameworkAdapter(
    name="crewai",
    display_name="CrewAI",
    tier=1,
    description="Role-based multi-agent crews with sequential or hierarchical process",
    orchestrations=("single", "sequential", "supervisor"),
    generate=generate,
)
