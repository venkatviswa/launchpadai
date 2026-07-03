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


@pytest.mark.unit
def test_xgboost_does_not_pass_use_label_encoder(tmp_path, make_config):
    """use_label_encoder was removed from the XGBoost sklearn API."""
    config = make_config(include_ml_pipeline=True, ml_framework="xgboost")
    generate_ml_pipeline(config, tmp_path)

    content = (tmp_path / "ml_models" / "training" / "train.py").read_text()
    assert "use_label_encoder" not in content


@pytest.mark.unit
def test_transformers_uses_eval_strategy(tmp_path, make_config):
    """TrainingArguments renamed evaluation_strategy to eval_strategy."""
    config = make_config(include_ml_pipeline=True, ml_framework="transformers")
    generate_ml_pipeline(config, tmp_path)

    content = (tmp_path / "ml_models" / "training" / "train.py").read_text()
    assert "eval_strategy=" in content
    assert "evaluation_strategy=" not in content
