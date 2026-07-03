"""Generate the agents layer — per-agent vertical slices plus framework orchestration.

Each agent in config.agents becomes a slice at agents/<name>/ containing its
system prompt and tool list. The slice layout is framework-agnostic; the
orchestration files (and the agents/__init__.py entrypoint) are written by
the selected framework adapter.
"""
from pathlib import Path

from launchpadai.frameworks.registry import get_adapter


def generate_agents_layer(config, project_path: Path):
    """Generate agent slices, then delegate orchestration to the framework adapter."""
    base = project_path / "agents"
    base.mkdir(parents=True, exist_ok=True)

    for spec in config["agents"]:
        _generate_slice(config, base, spec)

    adapter = get_adapter(config["framework"])
    adapter.generate(config, project_path)


def _generate_slice(config, base: Path, spec):
    slice_dir = base / spec.name
    _write(slice_dir / "__init__.py", "")

    rag_note = ""
    if config["include_rag"]:
        rag_note = (
            "\n## Context\n"
            "Retrieved documents relevant to the request are appended below your\n"
            "instructions at runtime. Ground your answers in them; if they don't\n"
            "contain the answer, say so.\n"
        )

    _write(slice_dir / "prompts" / "system.md", f"""You are {spec.role} for {config['project_name']}.

## Your goal
{spec.goal}

## Instructions
- Answer accurately; if you don't know something, say so honestly — do not make up information
- When using tools, explain what you're doing and why
- Be concise but thorough
- Cite sources when referencing specific documents
{rag_note}""")

    _write(slice_dir / "tools.py", f'''"""Tools available to the '{spec.name}' agent.

Add Tool instances to TOOLS to give this agent capabilities beyond the
shared project-wide tools registered in tools/. Example:

    from tools.registry import Tool

    def lookup_order(order_id: str) -> str:
        ...

    TOOLS = [
        Tool(
            name="lookup_order",
            description="Look up an order by its ID.",
            parameters={{
                "type": "object",
                "properties": {{"order_id": {{"type": "string"}}}},
                "required": ["order_id"],
            }},
            func=lookup_order,
        ),
    ]
"""
from tools.registry import Tool

TOOLS: list[Tool] = []
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
