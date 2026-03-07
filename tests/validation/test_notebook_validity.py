"""Validation tests — verify generated Jupyter notebooks are valid .ipynb files."""
import json
import pytest
from launchpadai.generators.project import ProjectGenerator


REQUIRED_KEYS = {"nbformat", "nbformat_minor", "metadata", "cells"}
REQUIRED_CELL_KEYS = {"cell_type", "metadata", "source"}


@pytest.mark.validation
@pytest.mark.parametrize("data_format", ["csv", "parquet", "json", "sql"])
@pytest.mark.parametrize("include_rag", [True, False])
@pytest.mark.parametrize("include_ml_pipeline", [True, False])
def test_notebooks_are_valid_ipynb(tmp_path, make_config, data_format, include_rag, include_ml_pipeline):
    config = make_config(
        include_notebooks=True, include_data_layer=True,
        data_format=data_format, include_rag=include_rag,
        include_ml_pipeline=include_ml_pipeline,
    )
    project_path = tmp_path / "nb-test"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    notebooks = list(project_path.rglob("*.ipynb"))
    assert len(notebooks) >= 1, "Expected at least one notebook"

    for nb_file in notebooks:
        content = nb_file.read_text()
        nb = json.loads(content)

        # Required top-level keys
        missing = REQUIRED_KEYS - set(nb.keys())
        assert not missing, f"Missing keys in {nb_file.name}: {missing}"

        assert nb["nbformat"] == 4, f"Expected nbformat 4 in {nb_file.name}"

        # Each cell must have required keys
        for i, cell in enumerate(nb["cells"]):
            cell_missing = REQUIRED_CELL_KEYS - set(cell.keys())
            assert not cell_missing, f"Cell {i} missing keys in {nb_file.name}: {cell_missing}"
            assert cell["cell_type"] in ("code", "markdown", "raw"), \
                f"Invalid cell_type '{cell['cell_type']}' in {nb_file.name} cell {i}"
