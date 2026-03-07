"""Architecture Diagram Generator — creates Mermaid diagrams from project config.

Generates:
1. End-to-end data flow diagram (how a user request flows through the system)
2. Component architecture diagram (what's in the stack)
3. Standalone .mermaid file for rendering
4. Embedded Mermaid block for README.md
"""


# Display names for config values
DISPLAY_NAMES = {
    # Frameworks
    "plain": "Plain Python",
    "langchain": "LangChain / LangGraph",
    "llamaindex": "LlamaIndex",
    "crewai": "CrewAI",
    "haystack": "Haystack",
    # LLM providers
    "openai": "OpenAI GPT-4o",
    "anthropic": "Anthropic Claude",
    "google": "Google Gemini",
    "ollama": "Ollama (Local)",
    "multiple": "Multiple LLMs",
    # Embeddings
    "openai-small": "OpenAI text-embedding-3-small",
    "openai-large": "OpenAI text-embedding-3-large",
    "cohere": "Cohere embed-v4",
    "bge-m3": "BGE-M3 (Local)",
    "gte-qwen2": "GTE-Qwen2 (Local)",
    "nomic": "Nomic v1.5 (Local)",
    # Vector DBs
    "chroma": "ChromaDB",
    "pinecone": "Pinecone",
    "weaviate": "Weaviate",
    "qdrant": "Qdrant",
    "pgvector": "pgvector (PostgreSQL)",
    # Observability
    "langfuse": "LangFuse",
    "langsmith": "LangSmith",
    "opentelemetry": "OpenTelemetry",
    "none": "None",
    # UI
    "streamlit": "Streamlit",
    "gradio": "Gradio",
    "nextjs": "Next.js",
    # Auth
    "simple": "Password Auth",
    "multi_user": "Multi-User Auth",
    "oauth": "OAuth / SSO",
    # ML
    "sklearn": "scikit-learn",
    "pytorch": "PyTorch",
    "xgboost": "XGBoost",
    "transformers": "HuggingFace Transformers",
}


def _dn(key: str) -> str:
    """Get display name for a config value."""
    return DISPLAY_NAMES.get(key, key.replace("-", " ").replace("_", " ").title())


def generate_architecture_diagram(config: dict) -> str:
    """Generate a Mermaid flowchart showing the end-to-end request flow."""

    framework = _dn(config["framework"])
    llm = _dn(config["llm_provider"])
    embedding = _dn(config["embedding_model"])
    vectordb = _dn(config["vector_db"])
    ui = _dn(config.get("ui", "none"))
    obs = config.get("observability", "none")
    auth = config.get("auth", "none")
    has_rag = config.get("include_rag", False)
    has_guardrails = config.get("include_guardrails", False)
    has_mcp = config.get("include_mcp", False)
    has_ml = config.get("include_ml_pipeline", False)
    ml_framework = _dn(config.get("ml_framework", "sklearn"))

    lines = []
    lines.append("flowchart TD")

    # ── Subgraph: User Interface Layer ─────────────────────────
    lines.append("    subgraph UI_LAYER[\"🖥️ User Interface\"]")
    if config.get("ui", "none") != "none":
        lines.append(f'        USER(["👤 User"]) --> UI["{ui}"]')
    else:
        lines.append(f'        USER(["👤 User"]) --> CLI["CLI Interface"]')
    lines.append("    end")
    lines.append("")

    # ── Auth Gate ──────────────────────────────────────────────
    if auth != "none":
        auth_name = _dn(auth)
        if config.get("ui", "none") != "none":
            lines.append(f'    UI --> AUTH{{"🔐 {auth_name}"}}')
        else:
            lines.append(f'    CLI --> AUTH{{"🔐 {auth_name}"}}')
        lines.append(f'    AUTH --> API["⚡ FastAPI"]')
    else:
        if config.get("ui", "none") != "none":
            lines.append('    UI --> API["⚡ FastAPI"]')
        else:
            lines.append('    CLI --> API["⚡ FastAPI"]')
    lines.append("")

    # ── Subgraph: Agent Core ──────────────────────────────────
    lines.append("    subgraph AGENT_CORE[\"🧠 Agent Core\"]")
    lines.append(f'        API --> ORCHESTRATOR["🔄 Orchestrator<br/>{framework}"]')

    if has_guardrails:
        lines.append('        ORCHESTRATOR --> INPUT_GUARD["🛡️ Input<br/>Guardrails"]')
        lines.append('        INPUT_GUARD --> PROMPT["📝 Prompt<br/>Templates"]')
    else:
        lines.append('        ORCHESTRATOR --> PROMPT["📝 Prompt<br/>Templates"]')

    lines.append(f'        PROMPT --> LLM_CALL["🤖 LLM Call<br/>{llm}"]')

    if has_guardrails:
        lines.append('        LLM_CALL --> OUTPUT_GUARD["🛡️ Output<br/>Guardrails"]')
        lines.append('        OUTPUT_GUARD --> RESPONSE["📤 Response"]')
    else:
        lines.append('        LLM_CALL --> RESPONSE["📤 Response"]')

    lines.append('        MEMORY["💾 Conversation<br/>Memory"] <--> ORCHESTRATOR')
    lines.append("    end")
    lines.append("")

    # ── Subgraph: Knowledge / RAG ─────────────────────────────
    if has_rag:
        lines.append("    subgraph RAG_LAYER[\"📚 Knowledge / RAG\"]")
        lines.append(f'        RETRIEVER["🔍 Retriever"] --> VECTORDB[("🗄️ {vectordb}")]')
        lines.append(f'        EMBEDDER["📊 Embedding<br/>{embedding}"] --> VECTORDB')
        lines.append("    end")
        lines.append("")
        lines.append("    ORCHESTRATOR --> RETRIEVER")
        lines.append("    RETRIEVER --> PROMPT")
        lines.append("")

    # ── Subgraph: Tools ───────────────────────────────────────
    tool_items = []
    tool_items.append('TOOLS_REG["🔧 Tool<br/>Registry"]')
    if has_mcp:
        tool_items.append('MCP["🔌 MCP<br/>Servers"]')
    if has_ml:
        tool_items.append(f'ML_PREDICT["🧪 ML Predict<br/>{ml_framework}"]')
    tool_items.append('EXT_API["🌐 External<br/>APIs"]')

    lines.append("    subgraph TOOLS_LAYER[\"🛠️ Tools & Integrations\"]")
    for item in tool_items:
        lines.append(f"        {item}")
    lines.append("    end")
    lines.append("")
    lines.append("    LLM_CALL <-->|\"tool calls\"| TOOLS_REG")
    if has_mcp:
        lines.append("    TOOLS_REG --> MCP")
    if has_ml:
        lines.append("    TOOLS_REG --> ML_PREDICT")
    lines.append("    TOOLS_REG --> EXT_API")
    lines.append("")

    # ── Subgraph: ML Pipeline ─────────────────────────────────
    if has_ml:
        lines.append("    subgraph ML_LAYER[\"🧬 ML Pipeline\"]")
        lines.append(f'        TRAINING["🏋️ Training<br/>{ml_framework}"] --> MODEL_STORE[("📦 Model<br/>Artifacts")]')
        lines.append('        MODEL_STORE --> INFERENCE["⚡ Inference<br/>Engine"]')
        lines.append("    end")
        lines.append("")
        lines.append("    ML_PREDICT --> INFERENCE")
        lines.append("")

    # ── Subgraph: Data Layer ──────────────────────────────────
    if config.get("include_data_layer"):
        fmt = config.get("data_format", "csv").upper()
        lines.append("    subgraph DATA_LAYER[\"💿 Data Layer\"]")
        lines.append(f'        RAW_DATA["📂 Raw Data<br/>{fmt}"] --> PROCESSING["⚙️ Data<br/>Processing"]')
        lines.append('        PROCESSING --> PROCESSED["📂 Processed<br/>Data"]')
        lines.append("    end")
        lines.append("")
        if has_rag:
            lines.append("    PROCESSED -->|\"ingestion\"| EMBEDDER")
        if has_ml:
            lines.append("    PROCESSED -->|\"training data\"| TRAINING")
        lines.append("")

    # ── Observability (cross-cutting) ─────────────────────────
    if obs != "none":
        obs_name = _dn(obs)
        lines.append("    subgraph OBS_LAYER[\"📊 Observability\"]")
        lines.append(f'        TRACER["🔭 {obs_name}<br/>Tracing"]')
        lines.append('        COST["💰 Cost<br/>Tracker"]')
        lines.append("    end")
        lines.append("")
        lines.append("    LLM_CALL -.->|\"traces\"| TRACER")
        if has_rag:
            lines.append("    RETRIEVER -.->|\"traces\"| TRACER")
        lines.append("    TRACER -.-> COST")
        lines.append("")

    # ── Response back to user ─────────────────────────────────
    if config.get("ui", "none") != "none":
        lines.append("    RESPONSE --> UI")
    else:
        lines.append("    RESPONSE --> CLI")
    lines.append("")

    # ── Styling ───────────────────────────────────────────────
    lines.append("    %% Styling")
    lines.append("    classDef uiStyle fill:#4A90D9,stroke:#2C5F8A,color:#fff")
    lines.append("    classDef agentStyle fill:#7B68EE,stroke:#5B4BC7,color:#fff")
    lines.append("    classDef ragStyle fill:#2ECC71,stroke:#1A9B52,color:#fff")
    lines.append("    classDef toolStyle fill:#E67E22,stroke:#C0641A,color:#fff")
    lines.append("    classDef mlStyle fill:#E74C3C,stroke:#C0392B,color:#fff")
    lines.append("    classDef obsStyle fill:#9B59B6,stroke:#7D3C98,color:#fff")
    lines.append("    classDef dataStyle fill:#1ABC9C,stroke:#148F77,color:#fff")
    lines.append("")
    lines.append("    class USER,UI,CLI uiStyle")
    lines.append("    class ORCHESTRATOR,PROMPT,LLM_CALL,RESPONSE,MEMORY,INPUT_GUARD,OUTPUT_GUARD agentStyle")
    if has_rag:
        lines.append("    class RETRIEVER,VECTORDB,EMBEDDER ragStyle")
    lines.append("    class TOOLS_REG,MCP,ML_PREDICT,EXT_API toolStyle")
    if has_ml:
        lines.append("    class TRAINING,MODEL_STORE,INFERENCE mlStyle")
    if obs != "none":
        lines.append("    class TRACER,COST obsStyle")
    if config.get("include_data_layer"):
        lines.append("    class RAW_DATA,PROCESSING,PROCESSED dataStyle")

    return "\n".join(lines)


def generate_stack_summary(config: dict) -> str:
    """Generate a text summary of the technology stack."""

    sections = []

    sections.append(f"| Layer | Technology | Purpose |")
    sections.append(f"|-------|-----------|---------|")
    sections.append(f"| **Framework** | {_dn(config['framework'])} | Agent orchestration and reasoning loop |")
    sections.append(f"| **LLM** | {_dn(config['llm_provider'])} | Language model for reasoning and generation |")
    sections.append(f"| **Embeddings** | {_dn(config['embedding_model'])} | Text-to-vector conversion for similarity search |")
    sections.append(f"| **Vector DB** | {_dn(config['vector_db'])} | Stores and searches document embeddings |")

    if config.get("include_rag"):
        sections.append(f"| **RAG Pipeline** | Custom | Document ingestion, chunking, and retrieval |")

    if config.get("include_guardrails"):
        sections.append(f"| **Guardrails** | Presidio + Custom | Input/output safety and PII detection |")

    if config.get("include_mcp"):
        sections.append(f"| **Tools** | MCP + Custom | External API and service integrations |")

    if config.get("include_ml_pipeline"):
        sections.append(f"| **ML Pipeline** | {_dn(config.get('ml_framework', 'sklearn'))} | Model training, inference, and registry |")

    if config.get("include_data_layer"):
        sections.append(f"| **Data Layer** | {config.get('data_format', 'csv').upper()} + Pandas | Data loading, cleaning, and transformation |")

    if config.get("observability", "none") != "none":
        sections.append(f"| **Observability** | {_dn(config['observability'])} | Tracing, monitoring, and cost tracking |")

    ui = config.get("ui", "none")
    if ui != "none":
        sections.append(f"| **Test UI** | {_dn(ui)} | Chat interface for testing the agent |")

    auth = config.get("auth", "none")
    if auth != "none":
        sections.append(f"| **Auth** | {_dn(auth)} | Authentication for UI and API |")

    if config.get("include_docker"):
        sections.append(f"| **Deployment** | Docker + Compose | Containerized deployment |")

    if config.get("include_eval"):
        sections.append(f"| **Evaluation** | Custom | Test cases and quality metrics |")

    if config.get("include_notebooks"):
        sections.append(f"| **Notebooks** | Jupyter | EDA, experiments, and analysis |")

    return "\n".join(sections)


def generate_request_flow_description(config: dict) -> str:
    """Generate a numbered description of the request flow."""

    steps = []
    step = 1

    ui = config.get("ui", "none")
    if ui != "none":
        steps.append(f"{step}. **User** sends a message through the **{_dn(ui)}** chat interface")
    else:
        steps.append(f"{step}. **User** sends a message through the **CLI**")
    step += 1

    auth = config.get("auth", "none")
    if auth != "none":
        steps.append(f"{step}. Request passes through **{_dn(auth)}** authentication")
        step += 1

    steps.append(f"{step}. **FastAPI** receives the request and routes to the agent")
    step += 1

    steps.append(f"{step}. **Orchestrator** ({_dn(config['framework'])}) manages the agent reasoning loop")
    step += 1

    if config.get("include_guardrails"):
        steps.append(f"{step}. **Input guardrails** check for prompt injection and PII")
        step += 1

    if config.get("include_rag"):
        steps.append(f"{step}. **Retriever** embeds the query using **{_dn(config['embedding_model'])}** and searches **{_dn(config['vector_db'])}** for relevant context")
        step += 1

    steps.append(f"{step}. **Prompt template** assembles system prompt + context + conversation history + user message")
    step += 1

    steps.append(f"{step}. **{_dn(config['llm_provider'])}** processes the assembled prompt and decides to respond or use tools")
    step += 1

    if config.get("include_mcp") or config.get("include_ml_pipeline"):
        tools = []
        if config.get("include_mcp"):
            tools.append("MCP servers")
        if config.get("include_ml_pipeline"):
            tools.append(f"ML predictions ({_dn(config.get('ml_framework', 'sklearn'))})")
        tools.append("custom functions")
        steps.append(f"{step}. If tool calls are needed, the agent executes them ({', '.join(tools)}) and loops back to the LLM")
        step += 1

    if config.get("include_guardrails"):
        steps.append(f"{step}. **Output guardrails** validate the response for safety and quality")
        step += 1

    steps.append(f"{step}. Response streams back to the user")

    return "\n".join(steps)
