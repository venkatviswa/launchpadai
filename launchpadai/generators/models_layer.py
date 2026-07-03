"""Generate the models layer — LLM and embedding abstractions."""
from pathlib import Path


# Replaces the plain singleton at the bottom of every provider template so
# LLM_MOCK=1 swaps in the offline mock without constructing the real client.
_SINGLETON_SWITCH = '''# Singleton instance — set LLM_MOCK=1 for offline development and tests
if settings.LLM_MOCK:
    from models.llm.mock import MockLLMProvider

    llm = MockLLMProvider()
else:
    llm = LLMProvider()'''


def generate_models_layer(config: dict, project_path: Path):
    """Generate LLM and embedding model wrapper files."""

    _write(project_path / "models" / "__init__.py", "")
    _write(project_path / "models" / "llm" / "__init__.py", "")
    _write(project_path / "models" / "embeddings" / "__init__.py", "")

    # LLM Provider (with LLM_MOCK switch on the singleton)
    llm_code = _get_llm_provider(config)
    llm_code = llm_code.replace("# Singleton instance\nllm = LLMProvider()", _SINGLETON_SWITCH)
    _write(project_path / "models" / "llm" / "provider.py", llm_code)

    # Mock LLM provider — offline development and generated tests
    _write(project_path / "models" / "llm" / "mock.py", _MOCK_PROVIDER)

    # Embedding Provider
    emb_code = _get_embedding_provider(config)
    _write(project_path / "models" / "embeddings" / "provider.py", emb_code)


_MOCK_PROVIDER = '''"""Mock LLM Provider — deterministic, offline stand-in for the real provider.

Enable it by setting LLM_MOCK=1 (see config/settings.py). No API keys, no
network. Useful for developing offline and for the generated tests.

Script behavior from tests:

    from models.llm.provider import llm  # MockLLMProvider when LLM_MOCK=1

    llm.queue_tool_call("get_current_time", {"timezone": "UTC"})
    llm.queue_response("The current time was retrieved.")
    result = agent.run("What time is it?")

Unqueued calls echo the last user message, so any conversation works
out of the box.
"""


class MockLLMProvider:
    """Records every call and replays queued responses (or echoes)."""

    def __init__(self):
        self.calls: list[dict] = []
        self._queue: list[dict] = []

    def reset(self):
        """Clear recorded calls and any queued responses."""
        self.calls = []
        self._queue = []

    def queue_response(self, text: str):
        """Queue a plain text response for the next chat() call."""
        self._queue.append({"message": {"content": text}})

    def queue_tool_call(self, name: str, arguments: dict = None, call_id: str = None):
        """Queue a tool call for the next chat() call."""
        self._queue.append({
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": call_id or f"mock_call_{len(self._queue)}",
                    "function": {"name": name, "arguments": arguments or {}},
                }],
            }
        })

    def _last_user_message(self, messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"]
        return ""

    def chat(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> dict:
        """Return the next queued response, or echo the last user message.

        Responses use the same dict shape the agent loop already understands
        (see Agent._extract_text / _extract_tool_calls).
        """
        self.calls.append({"messages": list(messages), "tools": tools})
        if self._queue:
            return self._queue.pop(0)
        return {"message": {"content": f"[mock response] You said: {self._last_user_message(messages)}"}}

    def simple(self, prompt: str) -> str:
        """Simple single-turn completion."""
        response = self.chat([{"role": "user", "content": prompt}])
        return response["message"]["content"]
'''


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
        return header + '''import json

import anthropic


class LLMProvider:
    """Anthropic Claude LLM provider.

    Accepts OpenAI-style message lists and tool schemas and adapts them to
    the Anthropic Messages API, so agent code stays provider-agnostic.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Split out system text and convert OpenAI-style turns to Anthropic format."""
        system_parts = []
        converted = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_parts.append(m["content"])
            elif role == "assistant" and m.get("tool_calls"):
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    args = tc["function"]["arguments"]
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(args) if isinstance(args, str) else args,
                    })
                converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": m.get("content", ""),
                    }],
                })
            else:
                converted.append({"role": role, "content": m["content"]})
        return "\\n\\n".join(system_parts), converted

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI function-tool schemas to Anthropic tool schemas."""
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for t in tools
        ]

    def chat(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> dict:
        """Send a message to Claude (accepts OpenAI-style messages/tools)."""
        system, converted = self._convert_messages(messages)
        params = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "temperature": kwargs.pop("temperature", settings.LLM_TEMPERATURE),
            "messages": converted,
            **kwargs,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self._convert_tools(tools)

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
