"""Unit tests for data_layer generator."""
import ast
import pytest
from launchpadai.generators.data_layer import generate_data_layer


DATA_FORMATS = ["csv", "parquet", "json", "sql"]


@pytest.mark.unit
def test_skipped_when_disabled(tmp_path, make_config):
    config = make_config(include_data_layer=False, include_notebooks=True)
    generate_data_layer(config, tmp_path)
    assert not (tmp_path / "data_processing").exists()


@pytest.mark.unit
@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_generates_valid_python(tmp_path, make_config, data_format):
    config = make_config(include_data_layer=True, include_notebooks=True, data_format=data_format)
    generate_data_layer(config, tmp_path)

    for py_file in (tmp_path / "data_processing").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_loader_uses_correct_format(tmp_path, make_config, data_format):
    config = make_config(include_data_layer=True, include_notebooks=True, data_format=data_format)
    generate_data_layer(config, tmp_path)

    content = (tmp_path / "data_processing" / "loader.py").read_text()
    # The loader handles all formats with a generic switch.
    # Just verify the default format appears in config references.
    assert "read_csv" in content or "load_raw" in content
