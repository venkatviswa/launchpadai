"""Unit tests for observability_layer generator."""
import ast
import pytest
from launchpadai.generators.observability_layer import generate_observability_layer


@pytest.mark.unit
def test_skipped_when_none(tmp_path, make_config):
    config = make_config(observability="none")
    generate_observability_layer(config, tmp_path)
    assert not (tmp_path / "observability").exists()


@pytest.mark.unit
@pytest.mark.parametrize("obs", ["langfuse", "langsmith", "opentelemetry"])
def test_generates_valid_python(tmp_path, make_config, obs):
    config = make_config(observability=obs)
    generate_observability_layer(config, tmp_path)

    for py_file in (tmp_path / "observability").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("obs", ["langfuse", "langsmith", "opentelemetry"])
def test_generates_expected_files(tmp_path, make_config, obs):
    config = make_config(observability=obs)
    generate_observability_layer(config, tmp_path)

    assert (tmp_path / "observability" / "tracer.py").exists()
    assert (tmp_path / "observability" / "cost_tracker.py").exists()


@pytest.mark.unit
def test_opentelemetry_has_fastapi_instrument(tmp_path, make_config):
    config = make_config(observability="opentelemetry")
    generate_observability_layer(config, tmp_path)

    assert (tmp_path / "observability" / "fastapi_instrument.py").exists()
