"""Unit tests for models_layer generator."""
import ast
import pytest
from launchpadai.generators.models_layer import generate_models_layer


LLM_PROVIDERS = ["openai", "anthropic", "google", "ollama", "multiple"]
EMBEDDING_MODELS = ["openai-small", "openai-large", "cohere", "bge-m3", "gte-qwen2", "nomic", "ollama"]


@pytest.mark.unit
@pytest.mark.parametrize("llm_provider", LLM_PROVIDERS)
@pytest.mark.parametrize("embedding_model", EMBEDDING_MODELS)
def test_generates_valid_python(tmp_path, make_config, llm_provider, embedding_model):
    config = make_config(llm_provider=llm_provider, embedding_model=embedding_model)
    generate_models_layer(config, tmp_path)

    for py_file in (tmp_path / "models").rglob("*.py"):
        content = py_file.read_text()
        if content.strip():
            ast.parse(content)


@pytest.mark.unit
@pytest.mark.parametrize("llm_provider", LLM_PROVIDERS)
def test_llm_has_singleton(tmp_path, make_config, llm_provider):
    config = make_config(llm_provider=llm_provider)
    generate_models_layer(config, tmp_path)

    content = (tmp_path / "models" / "llm" / "provider.py").read_text()
    assert "llm = LLMProvider()" in content
    assert "class LLMProvider" in content


@pytest.mark.unit
def test_openai_llm_uses_correct_sdk(tmp_path, make_config):
    config = make_config(llm_provider="openai")
    generate_models_layer(config, tmp_path)

    content = (tmp_path / "models" / "llm" / "provider.py").read_text()
    assert "from openai import OpenAI" in content
    assert "chat.completions.create" in content


@pytest.mark.unit
def test_anthropic_llm_uses_correct_sdk(tmp_path, make_config):
    config = make_config(llm_provider="anthropic")
    generate_models_layer(config, tmp_path)

    content = (tmp_path / "models" / "llm" / "provider.py").read_text()
    assert "import anthropic" in content
    assert "messages.create" in content


@pytest.mark.unit
def test_stub_providers_raise_not_implemented(tmp_path, make_config):
    for provider in ["google", "multiple"]:
        config = make_config(llm_provider=provider)
        generate_models_layer(config, tmp_path)

        content = (tmp_path / "models" / "llm" / "provider.py").read_text()
        assert "NotImplementedError" in content


@pytest.mark.unit
@pytest.mark.parametrize("embedding_model", EMBEDDING_MODELS)
def test_embedding_has_singleton(tmp_path, make_config, embedding_model):
    config = make_config(embedding_model=embedding_model)
    generate_models_layer(config, tmp_path)

    content = (tmp_path / "models" / "embeddings" / "provider.py").read_text()
    assert "embeddings = EmbeddingProvider()" in content


@pytest.mark.unit
def test_openai_embedding_model_names(tmp_path, make_config):
    config = make_config(embedding_model="openai-small")
    generate_models_layer(config, tmp_path)
    content = (tmp_path / "models" / "embeddings" / "provider.py").read_text()
    assert "text-embedding-3-small" in content

    config = make_config(embedding_model="openai-large")
    generate_models_layer(config, tmp_path)
    content = (tmp_path / "models" / "embeddings" / "provider.py").read_text()
    assert "text-embedding-3-large" in content


@pytest.mark.unit
@pytest.mark.parametrize("model", ["bge-m3", "gte-qwen2", "nomic"])
def test_local_embeddings_use_sentence_transformers(tmp_path, make_config, model):
    config = make_config(embedding_model=model)
    generate_models_layer(config, tmp_path)

    content = (tmp_path / "models" / "embeddings" / "provider.py").read_text()
    assert "SentenceTransformer" in content
