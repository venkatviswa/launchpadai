"""Unit tests for tools_layer generator."""
import ast
import pytest
from launchpadai.generators.tools_layer import generate_tools_layer


@pytest.mark.unit
@pytest.mark.parametrize("include_mcp", [True, False])
def test_generates_valid_python(tmp_path, make_config, include_mcp):
    config = make_config(include_mcp=include_mcp)
    generate_tools_layer(config, tmp_path)

    for py_file in (tmp_path / "tools").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
def test_mcp_files_when_enabled(tmp_path, make_config):
    config = make_config(include_mcp=True)
    generate_tools_layer(config, tmp_path)

    assert (tmp_path / "tools" / "mcp" / "servers.py").exists()


@pytest.mark.unit
def test_no_mcp_files_when_disabled(tmp_path, make_config):
    config = make_config(include_mcp=False)
    generate_tools_layer(config, tmp_path)

    assert not (tmp_path / "tools" / "mcp").exists()


@pytest.mark.unit
def test_registry_always_generated(tmp_path, make_config):
    config = make_config()
    generate_tools_layer(config, tmp_path)

    assert (tmp_path / "tools" / "registry.py").exists()
    assert (tmp_path / "tools" / "example_tool.py").exists()
