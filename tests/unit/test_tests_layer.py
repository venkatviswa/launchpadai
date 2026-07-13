"""Tests for the tests layer generator and the mock LLM provider generation."""
import ast

import pytest

from launchpadai.generators.config_files import generate_config_files
from launchpadai.generators.models_layer import generate_models_layer
from launchpadai.generators.requirements import generate_requirements
from launchpadai.generators.tests_layer import generate_tests_layer

FRAMEWORKS = ["plain", "langgraph", "crewai", "agentscript"]


def _parses(path):
    ast.parse(path.read_text())


class TestMockProviderGeneration:
    def test_mock_module_generated(self, make_config, tmp_path):
        config = make_config()
        generate_models_layer(config, tmp_path)
        mock = tmp_path / "models" / "llm" / "mock.py"
        assert mock.exists()
        _parses(mock)
        content = mock.read_text()
        assert "class MockLLMProvider" in content
        assert "def queue_response" in content
        assert "def queue_tool_call" in content
        assert "def reset" in content

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama"])
    def test_provider_singleton_switches_on_llm_mock(self, make_config, tmp_path, provider):
        config = make_config(llm_provider=provider)
        generate_models_layer(config, tmp_path)
        content = (tmp_path / "models" / "llm" / "provider.py").read_text()
        assert "if settings.LLM_MOCK:" in content
        assert "MockLLMProvider()" in content
        assert "llm = LLMProvider()" in content

    def test_settings_define_llm_mock(self, make_config, tmp_path):
        config = make_config()
        generate_config_files(config, tmp_path)
        settings = (tmp_path / "config" / "settings.py").read_text()
        assert 'LLM_MOCK = os.getenv("LLM_MOCK"' in settings

    def test_env_example_mentions_llm_mock(self, make_config, tmp_path):
        config = make_config()
        generate_config_files(config, tmp_path)
        assert "LLM_MOCK=1" in (tmp_path / ".env.example").read_text()

    def test_dev_requirements_generated(self, make_config, tmp_path):
        config = make_config()
        generate_requirements(config, tmp_path)
        assert "pytest" not in (tmp_path / "requirements.txt").read_text()
        dev = (tmp_path / "requirements-dev.txt").read_text()
        assert "pytest" in dev
        assert "httpx" in dev


class TestGeneratedDocs:
    """The generated README must document the offline test workflow."""

    def test_readme_documents_offline_tests(self, make_config, tmp_path):
        config = make_config()
        generate_config_files(config, tmp_path)
        readme = (tmp_path / "README.md").read_text()
        assert "requirements-dev.txt" in readme
        assert "pytest tests" in readme


class TestTestsLayerGeneration:
    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_contract_tests_generated_for_every_framework(self, make_config, tmp_path, framework):
        config = make_config(framework=framework)
        generate_tests_layer(config, tmp_path)

        conftest = tmp_path / "tests" / "conftest.py"
        test_agent = tmp_path / "tests" / "test_agent.py"
        assert conftest.exists() and test_agent.exists()
        _parses(conftest)
        _parses(test_agent)

        assert 'os.environ.setdefault("LLM_MOCK", "1")' in conftest.read_text()
        content = test_agent.read_text()
        assert "def test_entrypoint_contract" in content
        assert '"_traced_run"' in content

    def test_full_loop_tests_only_for_plain(self, make_config, tmp_path):
        for framework in FRAMEWORKS:
            target = tmp_path / framework
            config = make_config(framework=framework)
            generate_tests_layer(config, target)
            content = (target / "tests" / "test_agent.py").read_text()
            if framework == "plain":
                assert "test_agent_full_loop_offline" in content
                assert "test_conversation_memory_offline" in content
            else:
                assert "test_agent_full_loop_offline" not in content

    def test_tool_loop_test_only_for_single_orchestration(self, make_config, tmp_path):
        single = make_config(framework="plain", orchestration="single")
        generate_tests_layer(single, tmp_path / "single")
        assert "test_agent_tool_loop_offline" in (
            tmp_path / "single" / "tests" / "test_agent.py"
        ).read_text()

        multi = make_config(framework="plain", orchestration="sequential")
        generate_tests_layer(multi, tmp_path / "multi")
        content = (tmp_path / "multi" / "tests" / "test_agent.py").read_text()
        assert "test_agent_tool_loop_offline" not in content
        assert "test_agent_full_loop_offline" in content

    def test_rag_projects_get_retriever_stub(self, make_config, tmp_path):
        config = make_config(framework="plain", include_rag=True)
        generate_tests_layer(config, tmp_path / "rag")
        assert "stub_retriever" in (tmp_path / "rag" / "tests" / "test_agent.py").read_text()

        config = make_config(framework="plain", include_rag=False)
        generate_tests_layer(config, tmp_path / "norag")
        assert "stub_retriever" not in (tmp_path / "norag" / "tests" / "test_agent.py").read_text()

    def test_guardrail_tests_gated_on_flag(self, make_config, tmp_path):
        with_guards = make_config(include_guardrails=True)
        generate_tests_layer(with_guards, tmp_path / "guards")
        guard_file = tmp_path / "guards" / "tests" / "test_guardrails.py"
        assert guard_file.exists()
        _parses(guard_file)
        agent_tests = (tmp_path / "guards" / "tests" / "test_agent.py").read_text()
        assert "test_input_guardrails_block_prompt_injection" in agent_tests
        assert "test_output_guardrails_redact_pii" in agent_tests

        without = make_config(include_guardrails=False)
        generate_tests_layer(without, tmp_path / "noguards")
        assert not (tmp_path / "noguards" / "tests" / "test_guardrails.py").exists()
        assert "guardrails" not in (
            tmp_path / "noguards" / "tests" / "test_agent.py"
        ).read_text().lower()
