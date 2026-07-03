"""Unit tests for ui_layer generator."""
import ast
import json
import pytest
from launchpadai.generators.ui_layer import generate_ui_layer


@pytest.mark.unit
def test_no_ui_generates_nothing(tmp_path, make_config):
    config = make_config(ui="none")
    generate_ui_layer(config, tmp_path)
    assert not (tmp_path / "ui").exists()


@pytest.mark.unit
@pytest.mark.parametrize("ui", ["streamlit", "gradio"])
@pytest.mark.parametrize("auth", ["none", "simple", "multi_user"])
def test_python_ui_generates_valid_code(tmp_path, make_config, ui, auth):
    config = make_config(ui=ui, auth=auth)
    generate_ui_layer(config, tmp_path)

    app_file = tmp_path / "ui" / "app.py"
    assert app_file.exists()
    ast.parse(app_file.read_text())


@pytest.mark.unit
def test_streamlit_auth_integration(tmp_path, make_config):
    config = make_config(ui="streamlit", auth="multi_user")
    generate_ui_layer(config, tmp_path)

    content = (tmp_path / "ui" / "app.py").read_text()
    assert "auth_provider" in content
    assert "login_form" in content


@pytest.mark.unit
def test_streamlit_multi_user_login_passes_username_variable(tmp_path, make_config):
    """multi_user login must pass the username the user typed, not a literal."""
    config = make_config(ui="streamlit", auth="multi_user")
    generate_ui_layer(config, tmp_path)

    content = (tmp_path / "ui" / "app.py").read_text()
    assert 'username = st.text_input("Username"' in content
    assert "username=username," in content
    assert 'username="username"' not in content


@pytest.mark.unit
def test_streamlit_simple_auth_passes_no_username(tmp_path, make_config):
    config = make_config(ui="streamlit", auth="simple")
    generate_ui_layer(config, tmp_path)

    content = (tmp_path / "ui" / "app.py").read_text()
    assert "username=None," in content


@pytest.mark.unit
def test_ui_imports_agent_entrypoint(tmp_path, make_config):
    config = make_config(ui="streamlit")
    generate_ui_layer(config, tmp_path)

    content = (tmp_path / "ui" / "app.py").read_text()
    assert "from agents import agent" in content


@pytest.mark.unit
def test_streamlit_no_auth(tmp_path, make_config):
    config = make_config(ui="streamlit", auth="none")
    generate_ui_layer(config, tmp_path)

    content = (tmp_path / "ui" / "app.py").read_text()
    assert "auth_provider" not in content


@pytest.mark.unit
def test_gradio_auth_integration(tmp_path, make_config):
    config = make_config(ui="gradio", auth="simple")
    generate_ui_layer(config, tmp_path)

    content = (tmp_path / "ui" / "app.py").read_text()
    assert "auth" in content.lower()


@pytest.mark.unit
def test_nextjs_generates_valid_files(tmp_path, make_config):
    config = make_config(ui="nextjs")
    generate_ui_layer(config, tmp_path)

    pkg = tmp_path / "ui" / "package.json"
    assert pkg.exists()
    json.loads(pkg.read_text())  # valid JSON

    page = tmp_path / "ui" / "app" / "page.js"
    assert page.exists()

    layout = tmp_path / "ui" / "app" / "layout.js"
    assert layout.exists()


@pytest.mark.unit
def test_project_name_in_ui(tmp_path, make_config):
    config = make_config(ui="streamlit", project_name="cool-agent")
    generate_ui_layer(config, tmp_path)

    content = (tmp_path / "ui" / "app.py").read_text()
    assert "cool-agent" in content
