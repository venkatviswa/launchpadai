"""Framework adapters — each orchestration framework plugs in here.

Adding a framework = adding a module with a `FrameworkAdapter` instance and a
`generate(config, project_path)` function, then registering it in
`registry.py`. Core generators never branch on framework names.
"""
from launchpadai.frameworks.base import FrameworkAdapter
from launchpadai.frameworks.registry import (
    ADAPTERS,
    framework_names,
    get_adapter,
    validate_config,
)

__all__ = [
    "ADAPTERS",
    "FrameworkAdapter",
    "framework_names",
    "get_adapter",
    "validate_config",
]
