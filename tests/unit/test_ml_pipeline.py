"""Unit tests for ml_pipeline generator."""
import ast
import pytest
from launchpadai.generators.ml_pipeline import generate_ml_pipeline


ML_FRAMEWORKS = ["sklearn", "pytorch", "xgboost", "transformers"]


@pytest.mark.unit
def test_skipped_when_disabled(tmp_path, make_config):
    config = make_config(include_ml_pipeline=False)
    generate_ml_pipeline(config, tmp_path)
    assert not (tmp_path / "ml_models").exists()


@pytest.mark.unit
@pytest.mark.parametrize("ml_framework", ML_FRAMEWORKS)
def test_generates_valid_python(tmp_path, make_config, ml_framework):
    config = make_config(include_ml_pipeline=True, ml_framework=ml_framework)
    generate_ml_pipeline(config, tmp_path)

    for py_file in (tmp_path / "ml_models").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)

    # Also check the tool file
    tool_file = tmp_path / "tools" / "ml_predict.py"
    if tool_file.exists():
        ast.parse(tool_file.read_text())


@pytest.mark.unit
@pytest.mark.parametrize("ml_framework", ML_FRAMEWORKS)
def test_generates_expected_files(tmp_path, make_config, ml_framework):
    config = make_config(include_ml_pipeline=True, ml_framework=ml_framework)
    generate_ml_pipeline(config, tmp_path)

    assert (tmp_path / "ml_models" / "training" / "train.py").exists()
    assert (tmp_path / "ml_models" / "inference" / "predictor.py").exists()
    assert (tmp_path / "ml_models" / "registry.py").exists()
    assert (tmp_path / "ml_models" / "configs" / "training_config.yaml").exists()
