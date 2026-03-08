"""Generate the tools layer — external integrations and MCP."""
from pathlib import Path


def generate_tools_layer(config: dict, project_path: Path):
    """Generate tool integration files."""
    base = project_path / "tools"
    _write(base / "__init__.py", "")

    # Tool registry
    _write(base / "registry.py", '''"""Tool Registry — register and discover tools for the agent.

Tools are functions the agent can call to interact with external systems.
Each tool needs a name, description, parameters schema, and execute function.
"""


class Tool:
    """A tool the agent can use."""

    def __init__(self, name: str, description: str, parameters: dict, func: callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def execute(self, **kwargs):
        """Execute the tool with given parameters."""
        return self.func(**kwargs)

    def to_schema(self) -> dict:
        """Convert to the schema format expected by the LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered (safe pre-execution check)."""
        return name in self._tools

    def execute(self, name: str, **kwargs):
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return tool.execute(**kwargs)

    def list_schemas(self) -> list[dict]:
        """Get all tool schemas for the LLM."""
        return [tool.to_schema() for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())


# Global registry
registry = ToolRegistry()
''')

    # Example tool
    _write(base / "example_tool.py", '''"""Example tool — shows how to create and register tools.

Replace this with your actual tools (API calls, database queries, etc.)
"""
from tools.registry import Tool, registry


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in the specified timezone."""
    from datetime import datetime, timezone as tz
    import zoneinfo

    try:
        zone = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        now = datetime.now(tz.utc)
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")


# Register the tool
registry.register(Tool(
    name="get_current_time",
    description="Get the current date and time. Use when the user asks about the current time.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name (e.g., 'US/Eastern', 'Europe/London'). Defaults to UTC.",
            }
        },
    },
    func=get_current_time,
))
''')

    # MCP integration
    if config["include_mcp"]:
        _write(base / "mcp" / "__init__.py", "")
        _write(base / "mcp" / "servers.py", '''"""MCP Server Configuration.

Model Context Protocol (MCP) allows the agent to connect to external
services through a standardized interface.

Add your MCP server configurations here.
"""

# MCP server registry
MCP_SERVERS = {
    # Example: Uncomment and configure as needed
    #
    # "filesystem": {
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    # },
    #
    # "database": {
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://..."],
    # },
}


def get_server_config(name: str) -> dict | None:
    """Get MCP server configuration by name."""
    return MCP_SERVERS.get(name)


def list_servers() -> list[str]:
    """List all configured MCP servers."""
    return list(MCP_SERVERS.keys())
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
