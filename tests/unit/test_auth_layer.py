"""Unit tests for auth_layer generator."""
import ast
import pytest
from launchpadai.generators.auth_layer import generate_auth_layer


@pytest.mark.unit
def test_skipped_when_none(tmp_path, make_config):
    config = make_config(auth="none")
    generate_auth_layer(config, tmp_path)
    assert not (tmp_path / "auth").exists()


@pytest.mark.unit
@pytest.mark.parametrize("auth", ["simple", "multi_user"])
def test_generates_valid_python(tmp_path, make_config, auth):
    config = make_config(ui="streamlit", auth=auth)
    generate_auth_layer(config, tmp_path)

    for py_file in (tmp_path / "auth").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("auth", ["simple", "multi_user"])
def test_generates_expected_files(tmp_path, make_config, auth):
    config = make_config(ui="streamlit", auth=auth)
    generate_auth_layer(config, tmp_path)

    assert (tmp_path / "auth" / "provider.py").exists()
    assert (tmp_path / "auth" / "middleware.py").exists()
    assert (tmp_path / "auth" / "routes.py").exists()
