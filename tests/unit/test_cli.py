"""CLI integration tests — the Typer app driven through CliRunner."""

from typer.testing import CliRunner

from launchpadai.cli.main import app

runner = CliRunner()

MINIMAL_FLAGS = [
    "--no-rag", "--no-guardrails", "--no-eval", "--no-mcp",
    "--observability", "none", "--ui", "none",
    "--no-notebooks", "--no-data-layer", "--no-ml", "--no-docker",
]


def _init(name, tmp_path, *extra):
    return runner.invoke(app, ["init", name, "-o", str(tmp_path), *MINIMAL_FLAGS, *extra])


class TestInit:
    def test_defaults_generates_project(self, tmp_path):
        result = _init("cli-test", tmp_path)
        assert result.exit_code == 0, result.output
        project = tmp_path / "cli-test"
        assert (project / "launchpad.yaml").exists()
        assert (project / "agents" / "__init__.py").exists()

    def test_invalid_project_name_rejected(self, tmp_path):
        for bad in ["../escape", "quote\"name", "under_score", "9digit", "-leading-hyphen"]:
            result = _init(bad, tmp_path)
            assert result.exit_code != 0, f"{bad!r} should be rejected: {result.output}"
        assert not (tmp_path / "escape").exists()

    def test_uppercase_name_is_normalized(self, tmp_path):
        """Uppercase and spaces are normalized to a slug, not rejected."""
        result = _init("My Agent", tmp_path)
        assert result.exit_code == 0, result.output
        assert (tmp_path / "my-agent" / "launchpad.yaml").exists()

    def test_existing_dir_requires_force(self, tmp_path):
        assert _init("cli-test", tmp_path).exit_code == 0
        result = _init("cli-test", tmp_path)
        assert result.exit_code == 1
        assert "--force" in result.output

    def test_force_refuses_non_launchpad_directory(self, tmp_path):
        target = tmp_path / "cli-test"
        target.mkdir()
        (target / "precious.txt").write_text("do not delete")

        result = _init("cli-test", tmp_path, "--force")

        assert result.exit_code == 1
        assert (target / "precious.txt").exists(), "non-project files must survive"

    def test_force_regenerates_clean(self, tmp_path):
        """Disabled features must not leave stale files behind."""
        assert runner.invoke(
            app,
            ["init", "cli-test", "-o", str(tmp_path), "--rag", "--guardrails",
             "--ui", "streamlit", "--no-eval", "--no-mcp", "--observability", "none",
             "--no-notebooks", "--no-data-layer", "--no-ml", "--no-docker"],
        ).exit_code == 0
        project = tmp_path / "cli-test"
        assert (project / "guardrails").exists()
        assert (project / "ui").exists()

        result = _init("cli-test", tmp_path, "--force")

        assert result.exit_code == 0, result.output
        assert not (project / "guardrails").exists(), "stale guardrails/ must be removed"
        assert not (project / "ui").exists(), "stale ui/ must be removed"
        assert not (project / "knowledge").exists(), "stale knowledge/ must be removed"
        assert (project / "launchpad.yaml").exists()


class TestProjectCommands:
    def test_ingest_requires_launchpad_project(self, tmp_path):
        result = runner.invoke(app, ["ingest", str(tmp_path)])
        assert result.exit_code == 1
        assert "launchpad.yaml" in result.output

    def test_evaluate_requires_launchpad_project(self, tmp_path):
        result = runner.invoke(app, ["evaluate", str(tmp_path)])
        assert result.exit_code == 1
        assert "launchpad.yaml" in result.output


class TestAuthWiring:
    """The critical review finding: auth must actually protect /chat."""

    def test_authenticated_api_is_wired(self, tmp_path):
        result = runner.invoke(
            app,
            ["init", "auth-test", "-o", str(tmp_path), "--ui", "streamlit",
             "--auth", "simple", "--no-rag", "--no-guardrails", "--no-eval",
             "--no-mcp", "--observability", "none", "--no-notebooks",
             "--no-data-layer", "--no-ml", "--no-docker"],
        )
        assert result.exit_code == 0, result.output
        routes = (tmp_path / "auth-test" / "api" / "routes.py").read_text()
        assert "from auth.middleware import require_auth" in routes
        assert "app.include_router(auth_router)" in routes
        assert "Depends(require_auth)" in routes

        tests = (tmp_path / "auth-test" / "tests" / "test_api.py").read_text()
        assert "test_chat_requires_auth" in tests

    def test_unauthenticated_api_has_no_auth_imports(self, tmp_path):
        result = _init("noauth-test", tmp_path)
        assert result.exit_code == 0, result.output
        routes = (tmp_path / "noauth-test" / "api" / "routes.py").read_text()
        assert "require_auth" not in routes
        assert "from api.schemas import ChatRequest, ChatResponse" in routes
