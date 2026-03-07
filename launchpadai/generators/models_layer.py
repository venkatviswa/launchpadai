"""Generate the models layer — LLM and embedding abstractions."""
from pathlib import Path


def generate_models_layer(config: dict, project_path: Path):
    """Generate LLM and embedding model wrapper files."""

    _write(project_path / "models" / "__init__.py", "")
    _write(project_path / "models" / "llm" / "__init__.py", "")
    _write(project_path / "models" / "embeddings" / "__init__.py", "")

    # LLM Provider
    llm_code = _get_llm_provider(config)
    _write(project_path / "models" / "llm" / "provider.py", llm_code)

    # Embedding Provider
    emb_code = _get_embedding_provider(config)
    _write(project_path / "models" / "embeddings" / "provider.py", emb_code)


def _get_llm_provider(config: dict) -> str:
    """Generate LLM provider code based on selection."""
    provider = config["llm_provider"]

    header = '''"""LLM Provider — abstraction layer for language model calls.

Swap providers by changing config/settings.py without touching agent code.
"""
from config.settings import settings

'''

    if provider == "openai":
        return header + '''from openai import OpenAI


class LLMProvider:
    """OpenAI LLM provider."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL

    def chat(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> dict:
        """Send a chat completion request."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
            **kwargs,
        }
        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)
        return response

    def simple(self, prompt: str) -> str:
        """Simple single-turn completion."""
        response = self.chat([{"role": "user", "content": prompt}])
        return response.choices[0].message.content


# Singleton instance
llm = LLMProvider()
'''

    elif provider == "anthropic":
        return header + '''import anthropic


class LLMProvider:
    """Anthropic Claude LLM provider."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL

    def chat(self, messages: list[dict], system: str = None, tools: list[dict] = None, **kwargs) -> dict:
        """Send a message to Claude."""
        params = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": messages,
            **kwargs,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools

        response = self.client.messages.create(**params)
        return response

    def simple(self, prompt: str) -> str:
        """Simple single-turn completion."""
        response = self.chat([{"role": "user", "content": prompt}])
        return response.content[0].text


# Singleton instance
llm = LLMProvider()
'''

    elif provider == "ollama":
        return header + '''import ollama


class LLMProvider:
    """Ollama local LLM provider."""

    def __init__(self):
        self.model = settings.LLM_MODEL

    def chat(self, messages: list[dict], **kwargs) -> dict:
        """Send a chat completion request."""
        response = ollama.chat(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response

    def simple(self, prompt: str) -> str:
        """Simple single-turn completion."""
        response = self.chat([{"role": "user", "content": prompt}])
        return response["message"]["content"]


# Singleton instance
llm = LLMProvider()
'''

    else:  # google or multiple
        return header + '''# TODO: Configure your LLM provider
# See config/settings.py to set provider and API keys


class LLMProvider:
    """LLM provider — configure for your chosen provider."""

    def __init__(self):
        pass

    def chat(self, messages: list[dict], **kwargs) -> dict:
        raise NotImplementedError("Configure your LLM provider in models/llm/provider.py")

    def simple(self, prompt: str) -> str:
        raise NotImplementedError("Configure your LLM provider in models/llm/provider.py")


# Singleton instance
llm = LLMProvider()
'''


def _get_embedding_provider(config: dict) -> str:
    """Generate embedding provider code based on selection."""
    model = config["embedding_model"]

    header = '''"""Embedding Provider — abstraction layer for embedding generation.

IMPORTANT: Use the same model for both indexing and query-time embedding.
"""
from config.settings import settings

'''

    if model.startswith("openai"):
        model_name = "text-embedding-3-small" if model == "openai-small" else "text-embedding-3-large"
        return header + f'''from openai import OpenAI


class EmbeddingProvider:
    """OpenAI embedding provider."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "{model_name}"

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


# Singleton instance
embeddings = EmbeddingProvider()
'''

    elif model in ("bge-m3", "gte-qwen2", "nomic"):
        model_names = {
            "bge-m3": "BAAI/bge-m3",
            "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "nomic": "nomic-ai/nomic-embed-text-v1.5",
        }
        return header + f'''from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """Local HuggingFace embedding provider ({model})."""

    def __init__(self):
        self.model = SentenceTransformer("{model_names[model]}")

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        return self.model.encode(texts).tolist()


# Singleton instance
embeddings = EmbeddingProvider()
'''

    else:
        return header + '''# TODO: Configure your embedding provider


class EmbeddingProvider:
    """Embedding provider — configure for your chosen model."""

    def __init__(self):
        pass

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Configure in models/embeddings/provider.py")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Configure in models/embeddings/provider.py")


# Singleton instance
embeddings = EmbeddingProvider()
'''


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
