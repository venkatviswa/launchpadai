"""Unit tests for memory_layer generator."""
import ast
import pytest
from launchpadai.generators.memory_layer import generate_memory_layer


@pytest.mark.unit
def test_generates_valid_python(tmp_path, make_config):
    config = make_config()
    generate_memory_layer(config, tmp_path)

    content = (tmp_path / "memory" / "conversation.py").read_text()
    ast.parse(content)


@pytest.mark.unit
def test_generates_expected_files(tmp_path, make_config):
    config = make_config()
    generate_memory_layer(config, tmp_path)

    assert (tmp_path / "memory" / "__init__.py").exists()
    assert (tmp_path / "memory" / "conversation.py").exists()
