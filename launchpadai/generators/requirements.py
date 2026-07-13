"""Generate requirements.txt based on project configuration."""
from pathlib import Path


# Base dependencies every project needs
BASE_DEPS = [
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
]

# Development-only dependencies (requirements-dev.txt) — the generated tests/
# suite runs offline against the mock LLM provider
DEV_DEPS = [
    "pytest>=8.0.0",
    "httpx>=0.27.0",  # fastapi.testclient
]

LLM_DEPS = {
    "openai": ["openai>=1.30.0"],
    "anthropic": ["anthropic>=0.30.0"],
    "ollama": ["ollama>=0.2.0"],
}

FRAMEWORK_DEPS = {
    "plain": [],
    "langgraph": [
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
    ],
    "crewai": [
        "crewai>=0.80.0",
    ],
    "agentscript": [
        "simple-salesforce>=1.12.0",
    ],
}

# LangChain provider packages used by the LangGraph adapter (init_chat_model)
LANGGRAPH_LLM_DEPS = {
    "openai": ["langchain-openai>=0.2.0"],
    "anthropic": ["langchain-anthropic>=0.2.0"],
    "ollama": ["langchain-ollama>=0.2.0"],
}

EMBEDDING_DEPS = {
    "openai-small": ["openai>=1.30.0"],
    "openai-large": ["openai>=1.30.0"],
    "bge-m3": ["sentence-transformers>=2.7.0", "torch>=2.0.0"],
    "gte-qwen2": ["sentence-transformers>=2.7.0", "torch>=2.0.0"],
    "nomic": ["sentence-transformers>=2.7.0", "nomic>=3.0.0"],
}

VECTORDB_DEPS = {
    "chroma": ["chromadb>=0.5.0"],
    "pinecone": ["pinecone>=3.0.0"],
}

# Retrieval-layer option (pairs with any framework)
LLAMAINDEX_RETRIEVAL_DEPS = [
    "llama-index-core>=0.11.0",
]

LLAMAINDEX_EMBEDDING_DEPS = {
    "openai-small": ["llama-index-embeddings-openai>=0.2.0"],
    "openai-large": ["llama-index-embeddings-openai>=0.2.0"],
    "bge-m3": ["llama-index-embeddings-huggingface>=0.3.0"],
    "gte-qwen2": ["llama-index-embeddings-huggingface>=0.3.0"],
    "nomic": ["llama-index-embeddings-huggingface>=0.3.0"],
}

OBSERVABILITY_DEPS = {
    # Generated tracer code targets the langfuse v2 SDK API
    "langfuse": ["langfuse>=2.30.0,<3.0.0"],
    "langsmith": ["langsmith>=0.1.0"],
    "opentelemetry": [
        "opentelemetry-api>=1.25.0",
        "opentelemetry-sdk>=1.25.0",
        "opentelemetry-exporter-otlp>=1.25.0",
        "opentelemetry-instrumentation-fastapi>=0.46b0",
    ],
    "none": [],
}

UI_DEPS = {
    "streamlit": ["streamlit>=1.35.0"],
    "gradio": ["gradio>=4.30.0"],
    "nextjs": [],
    "none": [],
}

RAG_DEPS = [
    "tiktoken>=0.7.0",
]

GUARDRAIL_DEPS = [
    "presidio-analyzer>=2.2.0",
    "presidio-anonymizer>=2.2.0",
]

# The API layer is always generated, so these are always required
API_DEPS = [
    "fastapi>=0.111.0",
    "uvicorn>=0.30.0",
    "pydantic>=2.7.0",
]

MCP_DEPS = [
    "mcp>=1.0.0",
]

NOTEBOOK_DEPS = [
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]

DATA_LAYER_DEPS = [
    "pandas>=2.2.0",
    "numpy>=1.26.0",
]

DATA_FORMAT_DEPS = {
    "csv": [],
    "parquet": ["pyarrow>=16.0.0"],
    "json": [],
    "sql": ["sqlalchemy>=2.0.0", "psycopg2-binary>=2.9.0"],
}

ML_FRAMEWORK_DEPS = {
    "sklearn": ["scikit-learn>=1.5.0", "joblib>=1.4.0"],
    "pytorch": ["torch>=2.0.0", "scikit-learn>=1.5.0"],
    "xgboost": ["xgboost>=2.0.0", "scikit-learn>=1.5.0", "joblib>=1.4.0"],
    "transformers": [
        "transformers>=4.46.0",
        "datasets>=2.19.0",
        "torch>=2.0.0",
        "accelerate>=0.30.0",
        "scikit-learn>=1.5.0",
    ],
}


def generate_requirements(config: dict, project_path: Path):
    """Generate requirements.txt based on project configuration."""
    deps = set()

    # Base
    deps.update(BASE_DEPS)

    # Framework
    deps.update(FRAMEWORK_DEPS.get(config["framework"], []))

    # LLM provider
    deps.update(LLM_DEPS.get(config["llm_provider"], []))

    # LangChain provider packages for the LangGraph adapter
    if config["framework"] == "langgraph":
        deps.update(LANGGRAPH_LLM_DEPS.get(config["llm_provider"], []))

    # Observability
    deps.update(OBSERVABILITY_DEPS.get(config["observability"], []))

    # UI
    deps.update(UI_DEPS.get(config["ui"], []))

    # RAG stack — embeddings and vector store are only needed when RAG is on
    if config["include_rag"]:
        deps.update(RAG_DEPS)
        if config.get("retrieval") == "llamaindex":
            # LlamaIndex owns embedding + storage for its own index
            deps.update(LLAMAINDEX_RETRIEVAL_DEPS)
            deps.update(LLAMAINDEX_EMBEDDING_DEPS.get(config["embedding_model"], []))
        else:
            deps.update(EMBEDDING_DEPS.get(config["embedding_model"], []))
            deps.update(VECTORDB_DEPS.get(config["vector_db"], []))

    if config["include_guardrails"]:
        deps.update(GUARDRAIL_DEPS)

    if config["include_mcp"]:
        deps.update(MCP_DEPS)

    # Notebooks
    if config.get("include_notebooks"):
        deps.update(NOTEBOOK_DEPS)

    # Data layer
    if config.get("include_data_layer"):
        deps.update(DATA_LAYER_DEPS)
        deps.update(DATA_FORMAT_DEPS.get(config.get("data_format", "csv"), []))

    # ML Pipeline
    if config.get("include_ml_pipeline"):
        deps.update(ML_FRAMEWORK_DEPS.get(config.get("ml_framework", "sklearn"), []))

    # API layer is always generated
    deps.update(API_DEPS)

    # Sort and write
    sorted_deps = sorted(deps, key=lambda x: x.lower())
    req_content = "# Generated by LaunchpadAI\n"
    req_content += f"# Framework: {config['framework']}\n"
    req_content += f"# LLM: {config['llm_provider']}\n"
    req_content += f"# Vector DB: {config['vector_db']}\n\n"
    req_content += "\n".join(sorted_deps) + "\n"

    with open(project_path / "requirements.txt", "w") as f:
        f.write(req_content)

    with open(project_path / "requirements-dev.txt", "w") as f:
        f.write(
            "# Development dependencies — run the generated tests with:\n"
            "#   pip install -r requirements.txt -r requirements-dev.txt\n"
            "#   pytest tests\n\n"
            + "\n".join(DEV_DEPS)
            + "\n"
        )
