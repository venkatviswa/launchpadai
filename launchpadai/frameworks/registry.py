"""Framework adapter registry — the single place frameworks are wired in.

CLI choices, generation dispatch, and config validation all derive from
ADAPTERS. Adding a framework means adding an adapter module and one entry
here; core generators never branch on framework names.
"""
from launchpadai.frameworks.agentscript import ADAPTER as _agentscript
from launchpadai.frameworks.base import FrameworkAdapter
from launchpadai.frameworks.crewai import ADAPTER as _crewai
from launchpadai.frameworks.langgraph import ADAPTER as _langgraph
from launchpadai.frameworks.plain import ADAPTER as _plain

ADAPTERS: dict[str, FrameworkAdapter] = {
    adapter.name: adapter
    for adapter in (_plain, _langgraph, _crewai, _agentscript)
}


def framework_names() -> list[str]:
    return list(ADAPTERS.keys())


def get_adapter(name: str) -> FrameworkAdapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown framework '{name}'. Available: {', '.join(framework_names())}"
        ) from None


def validate_config(config) -> None:
    """Check framework-dependent constraints that ProjectConfig can't know about."""
    adapter = get_adapter(config["framework"])
    if config["orchestration"] not in adapter.orchestrations:
        raise ValueError(
            f"Framework '{adapter.name}' does not support "
            f"orchestration='{config['orchestration']}' "
            f"(supported: {', '.join(adapter.orchestrations)})"
        )
