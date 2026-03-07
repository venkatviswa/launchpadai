"""Validation tests — verify requirements.txt has correct packages for the config."""
import pytest
from launchpadai.generators.project import ProjectGenerator
from launchpadai.generators.requirements import (
    LLM_DEPS, FRAMEWORK_DEPS, EMBEDDING_DEPS, VECTORDB_DEPS,
)


def _parse_requirements(path):
    content = path.read_text()
    packages = set()
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            name = line.split(">=")[0].split("==")[0].split("<")[0].strip()
            packages.add(name)
    return packages


@pytest.mark.validation
@pytest.mark.parametrize("framework", ["plain", "langchain", "llamaindex", "crewai", "haystack"])
@pytest.mark.parametrize("llm_provider", ["openai", "anthropic", "google", "ollama"])
def test_full_project_requirements_match_config(tmp_path, make_config, framework, llm_provider):
    """After full generation, requirements.txt must include correct stack packages."""
    config = make_config(framework=framework, llm_provider=llm_provider)
    project_path = tmp_path / "req-check"
    project_path.mkdir()
    ProjectGenerator(config, project_path).generate()

    packages = _parse_requirements(project_path / "requirements.txt")

    # Framework deps
    for dep in FRAMEWORK_DEPS[framework]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {framework}"

    # LLM deps
    for dep in LLM_DEPS[llm_provider]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {llm_provider}"

    # Vector DB deps (chroma by default)
    for dep in VECTORDB_DEPS[config["vector_db"]]:
        dep_name = dep.split(">=")[0]
        assert dep_name in packages, f"Missing {dep_name} for {config['vector_db']}"
