"""Typed project configuration — the single source of truth for what gets generated.

All configuration crossing the CLI → generator boundary is validated here.
Generators may access values dict-style (``config["framework"]``) for
backwards compatibility, or as attributes.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Orchestration = Literal["single", "sequential", "supervisor"]


class AgentSpec(BaseModel):
    """One agent in the generated project — becomes a vertical slice under agents/<name>/."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_]*$",
        max_length=40,
        description="Python-identifier-safe slug; used as the slice directory name",
    )
    role: str = Field(..., min_length=1, description="Short role title, e.g. 'Research Analyst'")
    goal: str = Field(..., min_length=1, description="What this agent is responsible for")


class ProjectConfig(BaseModel):
    """Validated project configuration gathered from the wizard or CLI flags."""

    project_name: str = Field(..., min_length=1)
    framework: str = "plain"  # validated against the adapter registry at generation time
    llm_provider: Literal["openai", "anthropic", "ollama"] = "anthropic"
    embedding_model: Literal[
        "openai-small", "openai-large", "bge-m3", "gte-qwen2", "nomic"
    ] = "openai-small"
    vector_db: Literal["chroma", "pinecone"] = "chroma"
    retrieval: Literal["custom", "llamaindex"] = "custom"
    include_rag: bool = True
    include_guardrails: bool = True
    include_eval: bool = True
    include_mcp: bool = True
    observability: Literal["langfuse", "langsmith", "opentelemetry", "none"] = "langfuse"
    ui: Literal["streamlit", "gradio", "nextjs", "none"] = "streamlit"
    auth: Literal["none", "simple", "multi_user", "oauth"] = "none"
    include_notebooks: bool = True
    include_data_layer: bool = True
    data_format: Literal["csv", "parquet", "json", "sql"] = "csv"
    include_ml_pipeline: bool = False
    ml_framework: Literal["sklearn", "pytorch", "xgboost", "transformers"] = "sklearn"
    include_docker: bool = True
    agent_description: str = "An AI-powered assistant"
    agents: list[AgentSpec] = Field(default_factory=list)
    orchestration: Optional[Orchestration] = None

    @model_validator(mode="after")
    def _apply_defaults_and_check_consistency(self):
        if not self.agents:
            self.agents = [
                AgentSpec(name="assistant", role="Assistant", goal=self.agent_description)
            ]
        if self.orchestration is None:
            self.orchestration = "single" if len(self.agents) == 1 else "sequential"

        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError(f"agent names must be unique, got: {names}")
        if self.orchestration == "single" and len(self.agents) > 1:
            raise ValueError("orchestration='single' requires exactly one agent")
        if self.orchestration != "single" and len(self.agents) < 2:
            raise ValueError(
                f"orchestration='{self.orchestration}' requires at least two agents"
            )
        if self.auth == "oauth" and self.ui != "nextjs":
            raise ValueError("auth='oauth' is only supported with ui='nextjs'")
        if self.ui == "none" and self.auth != "none":
            raise ValueError("auth requires a UI; set ui or use auth='none'")
        return self

    # --- dict-style compatibility for generators and tests ---

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> dict:
        return self.model_dump()
