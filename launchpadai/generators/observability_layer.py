"""Generate the observability layer — tracing, monitoring, and cost tracking.

Observability for LLM apps tracks:
- Every LLM call (prompt, response, tokens, latency, cost)
- Retrieval quality (what was retrieved, relevance scores)
- Tool executions (which tools, inputs, outputs, errors)
- End-to-end traces (full user request lifecycle)
- Agent reasoning steps (how many loops, which path taken)
- Cost tracking (tokens used, estimated spend)
"""
from pathlib import Path


def generate_observability_layer(config: dict, project_path: Path):
    """Generate observability setup based on chosen provider."""
    obs = config.get("observability", "none")
    if obs == "none":
        return

    base = project_path / "observability"
    _write(base / "__init__.py", "")

    # Provider-specific tracer
    if obs == "langfuse":
        _generate_langfuse(config, base)
    elif obs == "langsmith":
        _generate_langsmith(config, base)
    elif obs == "opentelemetry":
        _generate_opentelemetry(config, base)

    # Common: cost tracker (works with any provider)
    _generate_cost_tracker(config, base)

    # Common: dashboard info
    _generate_dashboard_info(config, base)


def _generate_langfuse(config: dict, base: Path):
    """Generate LangFuse observability setup."""

    _write(base / "tracer.py", '''"""LangFuse Observability — open-source LLM tracing and monitoring.

LangFuse provides:
- Trace visualization for every LLM call chain
- Prompt versioning and management
- Cost tracking per trace, user, and model
- Evaluation scoring (manual + automated)
- Session grouping (conversations)
- Dataset management for testing

Setup:
    1. Create account at https://cloud.langfuse.com (or self-host)
    2. Get API keys from Settings → API Keys
    3. Add to .env:
       LANGFUSE_PUBLIC_KEY=pk-...
       LANGFUSE_SECRET_KEY=sk-...
       LANGFUSE_HOST=https://cloud.langfuse.com

Docs: https://langfuse.com/docs
"""
import os
import time
import uuid
from functools import wraps
from config.settings import settings

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class LangFuseTracer:
    """LangFuse tracing wrapper for the agent."""

    def __init__(self):
        if not LANGFUSE_AVAILABLE:
            print("Warning: langfuse not installed. Run: pip install langfuse")
            self.client = None
            return

        self.client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

    def trace_agent_run(self, user_message: str, session_id: str = None):
        """Create a trace for an agent run. Returns a trace context."""
        if not self.client:
            return DummyTrace()

        trace = self.client.trace(
            name="agent_run",
            session_id=session_id,
            input={"message": user_message},
            metadata={"project": settings.PROJECT_NAME},
        )
        return TraceContext(trace)

    def flush(self):
        """Flush pending events to LangFuse."""
        if self.client:
            self.client.flush()


class TraceContext:
    """Context manager for a LangFuse trace with helper methods."""

    def __init__(self, trace):
        self.trace = trace
        self._start_time = time.time()

    def span(self, name: str, **kwargs):
        """Create a child span (for retrieval, tool calls, etc.)."""
        return self.trace.span(name=name, **kwargs)

    def generation(self, name: str, model: str, input_messages: list,
                   output: str = None, usage: dict = None, **kwargs):
        """Log an LLM generation call.

        Args:
            name: Name of the generation step
            model: Model identifier (e.g. 'claude-sonnet-4-20250514')
            input_messages: The messages sent to the LLM
            output: The LLM response text
            usage: Token counts {'input': N, 'output': N, 'total': N}
        """
        return self.trace.generation(
            name=name,
            model=model,
            input=input_messages,
            output=output,
            usage=usage,
            **kwargs,
        )

    def score(self, name: str, value: float, comment: str = None):
        """Add a score to the trace (for evaluation).

        Common scores:
        - 'relevance': How relevant was the response (0-1)
        - 'accuracy': Was the response factually correct (0-1)
        - 'user_feedback': Thumbs up/down from user (0 or 1)
        - 'latency': Response time classification
        """
        self.trace.score(name=name, value=value, comment=comment)

    def set_output(self, output: dict):
        """Set the final output of the trace."""
        self.trace.update(output=output)

    def end(self):
        """Mark trace as complete."""
        duration = time.time() - self._start_time
        self.trace.update(
            output={"duration_seconds": round(duration, 3)},
        )


class DummyTrace:
    """No-op trace for when LangFuse is not configured."""
    def span(self, *args, **kwargs): return self
    def generation(self, *args, **kwargs): return self
    def score(self, *args, **kwargs): pass
    def set_output(self, *args, **kwargs): pass
    def end(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass


# Singleton
tracer = LangFuseTracer()


# === Usage example in your agent ===
#
# from observability.tracer import tracer
#
# def run(self, user_message, session_id):
#     trace = tracer.trace_agent_run(user_message, session_id)
#
#     # Log retrieval
#     with trace.span(name="retrieval", input={"query": user_message}) as span:
#         results = retriever.retrieve(user_message)
#         span.update(output={"num_results": len(results)})
#
#     # Log LLM call
#     trace.generation(
#         name="main_llm_call",
#         model="claude-sonnet-4-20250514",
#         input_messages=messages,
#         output=response_text,
#         usage={"input": prompt_tokens, "output": completion_tokens},
#     )
#
#     # Log user feedback
#     trace.score("user_feedback", 1.0, comment="Thumbs up")
#
#     trace.set_output({"response": response_text})
#     trace.end()
#     tracer.flush()
''')


def _generate_langsmith(config: dict, base: Path):
    """Generate LangSmith observability setup."""

    _write(base / "tracer.py", '''"""LangSmith Observability — LangChain's tracing and evaluation platform.

LangSmith provides:
- Automatic tracing for LangChain/LangGraph (zero-code for LC users)
- Manual tracing via @traceable decorator for non-LC code
- Prompt playground for testing prompt variations
- Dataset & evaluation management
- Annotation queues for human review
- Online evaluation (auto-scoring of production traces)

Setup:
    1. Create account at https://smith.langchain.com
    2. Get API key from Settings
    3. Add to .env:
       LANGSMITH_API_KEY=lsv2_...
       LANGSMITH_PROJECT=your-project-name
       LANGCHAIN_TRACING_V2=true

Docs: https://docs.smith.langchain.com
"""
import os
from functools import wraps
from config.settings import settings

# LangSmith auto-traces when these env vars are set:
#   LANGCHAIN_TRACING_V2=true
#   LANGSMITH_API_KEY=...
#   LANGSMITH_PROJECT=...
# If using LangChain, tracing is automatic — no code changes needed.

try:
    from langsmith import traceable, Client
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


def ensure_tracing_enabled():
    """Set environment variables to enable LangSmith tracing."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("Warning: LANGSMITH_API_KEY not set. Tracing disabled.")
        return False
    project = os.getenv("LANGSMITH_PROJECT", settings.PROJECT_NAME)
    os.environ["LANGSMITH_PROJECT"] = project
    return True


class LangSmithTracer:
    """LangSmith tracing wrapper.

    For LangChain/LangGraph: tracing is automatic when env vars are set.
    For custom code: use the @traced decorator or manual client.
    """

    def __init__(self):
        self.enabled = ensure_tracing_enabled()
        self.client = Client() if LANGSMITH_AVAILABLE and self.enabled else None

    def traced(self, name: str = None, run_type: str = "chain"):
        """Decorator to trace a function.

        Usage:
            @tracer.traced("my_function")
            def my_function(input):
                ...
        """
        if not LANGSMITH_AVAILABLE:
            def passthrough(func):
                return func
            return passthrough

        return traceable(name=name, run_type=run_type)

    def log_feedback(self, run_id: str, key: str, score: float, comment: str = None):
        """Log feedback for a run.

        Args:
            run_id: The LangSmith run ID
            key: Feedback key (e.g. 'correctness', 'helpfulness')
            score: Score value (typically 0 or 1, or 0-5)
            comment: Optional text comment
        """
        if self.client:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment,
            )

    def create_dataset(self, name: str, description: str = ""):
        """Create an evaluation dataset in LangSmith."""
        if self.client:
            return self.client.create_dataset(name, description=description)

    def list_runs(self, project_name: str = None, limit: int = 20):
        """List recent runs for analysis."""
        if self.client:
            project = project_name or os.getenv("LANGSMITH_PROJECT")
            return list(self.client.list_runs(project_name=project, limit=limit))
        return []


# Singleton
tracer = LangSmithTracer()


# === Usage example ===
#
# # Option 1: Decorator (for non-LangChain code)
# @tracer.traced("process_query")
# def process_query(query: str):
#     context = retrieve(query)
#     response = llm.chat(messages)
#     return response
#
# # Option 2: LangChain automatic tracing
# # Just set env vars — every chain/agent call is traced automatically
#
# # Option 3: Log user feedback
# tracer.log_feedback(run_id="...", key="thumbs_up", score=1.0)
''')


def _generate_opentelemetry(config: dict, base: Path):
    """Generate OpenTelemetry observability setup."""

    _write(base / "tracer.py", '''"""OpenTelemetry Observability — vendor-neutral distributed tracing.

OpenTelemetry (OTel) provides:
- Vendor-neutral tracing standard (works with any backend)
- Distributed tracing across microservices
- Metrics collection (latency, throughput, error rates)
- Log correlation with traces
- Export to: Jaeger, Zipkin, Datadog, New Relic, Grafana Tempo, etc.

Advantages over LangFuse/LangSmith:
- No vendor lock-in — switch backends without code changes
- Works across your entire stack (not just LLM calls)
- Enterprise-grade, CNCF graduated project
- Rich ecosystem of exporters and instrumentors

Disadvantages:
- More setup work than LLM-specific tools
- No built-in prompt management or LLM-specific features
- Need to manually instrument LLM calls

Setup:
    1. Choose a backend (Jaeger for local dev, or cloud provider)
    2. Add to .env:
       OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
       OTEL_SERVICE_NAME=your-agent-name
    3. For local dev, run Jaeger:
       docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one

Docs: https://opentelemetry.io/docs/languages/python/
"""
import os
import time
from contextlib import contextmanager
from config.settings import settings

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def setup_tracing() -> bool:
    """Initialize OpenTelemetry tracing."""
    if not OTEL_AVAILABLE:
        print("Warning: opentelemetry not installed. Run: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        return False

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    service_name = os.getenv("OTEL_SERVICE_NAME", settings.PROJECT_NAME)

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        print(f"OpenTelemetry tracing initialized — exporting to {endpoint}")
        return True
    except Exception as e:
        print(f"Warning: Could not connect to OTel collector at {endpoint}: {e}")
        print("Tracing disabled. Start a collector or update OTEL_EXPORTER_OTLP_ENDPOINT")
        return False


class OTelTracer:
    """OpenTelemetry tracing wrapper for AI agent operations."""

    def __init__(self):
        self.enabled = setup_tracing()
        if self.enabled:
            self._tracer = trace.get_tracer(settings.PROJECT_NAME)
        else:
            self._tracer = None

    @contextmanager
    def trace_agent_run(self, user_message: str, session_id: str = None):
        """Trace a complete agent run.

        Usage:
            with tracer.trace_agent_run("hello", session_id="abc") as span:
                # ... agent logic ...
                span.set_attribute("response", "world")
        """
        if not self._tracer:
            yield DummySpan()
            return

        with self._tracer.start_as_current_span("agent_run") as span:
            span.set_attribute("user.message", user_message)
            span.set_attribute("session.id", session_id or "default")
            span.set_attribute("project", settings.PROJECT_NAME)
            yield span

    @contextmanager
    def trace_llm_call(self, model: str, messages: list = None):
        """Trace an LLM API call.

        Usage:
            with tracer.trace_llm_call("claude-sonnet-4-20250514") as span:
                response = llm.chat(messages)
                span.set_attribute("llm.tokens.input", input_tokens)
                span.set_attribute("llm.tokens.output", output_tokens)
        """
        if not self._tracer:
            yield DummySpan()
            return

        with self._tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.message_count", len(messages) if messages else 0)
            start = time.time()
            yield span
            span.set_attribute("llm.duration_ms", round((time.time() - start) * 1000))

    @contextmanager
    def trace_retrieval(self, query: str):
        """Trace a vector store retrieval."""
        if not self._tracer:
            yield DummySpan()
            return

        with self._tracer.start_as_current_span("retrieval") as span:
            span.set_attribute("retrieval.query", query)
            yield span

    @contextmanager
    def trace_tool_call(self, tool_name: str, tool_input: dict = None):
        """Trace a tool execution."""
        if not self._tracer:
            yield DummySpan()
            return

        with self._tracer.start_as_current_span("tool_call") as span:
            span.set_attribute("tool.name", tool_name)
            if tool_input:
                span.set_attribute("tool.input", str(tool_input)[:500])
            yield span


class DummySpan:
    """No-op span for when OTel is not configured."""
    def set_attribute(self, *args, **kwargs): pass
    def add_event(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass


# Singleton
tracer = OTelTracer()


# === Usage example in your agent ===
#
# from observability.tracer import tracer
#
# def run(self, user_message, session_id):
#     with tracer.trace_agent_run(user_message, session_id) as root_span:
#
#         # Trace retrieval
#         with tracer.trace_retrieval(user_message) as ret_span:
#             results = retriever.retrieve(user_message)
#             ret_span.set_attribute("retrieval.num_results", len(results))
#
#         # Trace LLM call
#         with tracer.trace_llm_call("claude-sonnet-4-20250514", messages) as llm_span:
#             response = llm.chat(messages)
#             llm_span.set_attribute("llm.tokens.input", response.usage.input_tokens)
#             llm_span.set_attribute("llm.tokens.output", response.usage.output_tokens)
#
#         # Trace tool call
#         with tracer.trace_tool_call("search_orders", {"order_id": "123"}) as tool_span:
#             result = order_api.search("123")
#             tool_span.set_attribute("tool.success", True)
#
#         root_span.set_attribute("response", response_text[:500])
''')

    # Add FastAPI auto-instrumentation helper
    _write(base / "fastapi_instrument.py", '''"""FastAPI auto-instrumentation for OpenTelemetry.

Add this to your API startup to automatically trace all HTTP requests.

Usage in api/routes.py:
    from observability.fastapi_instrument import instrument_app
    instrument_app(app)
"""

def instrument_app(app):
    """Add OpenTelemetry instrumentation to a FastAPI app."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        print("FastAPI OTel instrumentation enabled")
    except ImportError:
        print("Install: pip install opentelemetry-instrumentation-fastapi")
''')


def _generate_cost_tracker(config: dict, base: Path):
    """Generate cost tracking utility (works with any observability provider)."""

    _write(base / "cost_tracker.py", '''"""Cost Tracker — estimate and track LLM API costs.

Tracks token usage and estimates costs per model.
Works independently of your observability provider.

Pricing is approximate and should be updated periodically.
Check provider pricing pages for current rates.
"""
import json
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime


# Approximate pricing per 1M tokens (input / output) — UPDATE AS NEEDED
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # Embedding models
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


class CostTracker:
    """Track and estimate LLM API costs."""

    def __init__(self, log_file: str = "observability/cost_log.jsonl"):
        self.log_file = Path(log_file)
        self.session_costs = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "calls": 0,
        })

    def log_usage(self, model: str, input_tokens: int, output_tokens: int,
                  session_id: str = "default", metadata: dict = None):
        """Log a single API call's token usage and estimate cost.

        Args:
            model: Model identifier
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            session_id: Session to group costs by
            metadata: Optional extra info (user_id, agent_step, etc.)

        Returns:
            Estimated cost in USD for this call
        """
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        # Update session totals
        session = self.session_costs[session_id]
        session["total_cost"] += cost
        session["total_input_tokens"] += input_tokens
        session["total_output_tokens"] += output_tokens
        session["calls"] += 1

        # Append to log file
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "session_id": session_id,
        }
        if metadata:
            entry["metadata"] = metadata

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\\n")

        return cost

    def get_session_summary(self, session_id: str = "default") -> dict:
        """Get cost summary for a session."""
        session = self.session_costs[session_id]
        return {
            "session_id": session_id,
            "total_cost_usd": round(session["total_cost"], 4),
            "total_input_tokens": session["total_input_tokens"],
            "total_output_tokens": session["total_output_tokens"],
            "total_calls": session["calls"],
        }

    def get_total_summary(self) -> dict:
        """Get cost summary across all sessions."""
        total_cost = sum(s["total_cost"] for s in self.session_costs.values())
        total_calls = sum(s["calls"] for s in self.session_costs.values())
        total_input = sum(s["total_input_tokens"] for s in self.session_costs.values())
        total_output = sum(s["total_output_tokens"] for s in self.session_costs.values())

        return {
            "total_cost_usd": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_calls": total_calls,
            "sessions": len(self.session_costs),
        }

    def get_daily_report(self) -> list[dict]:
        """Parse log file and generate daily cost report."""
        if not self.log_file.exists():
            return []

        daily = defaultdict(lambda: {"cost": 0.0, "calls": 0, "tokens": 0})
        with open(self.log_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                day = entry["timestamp"][:10]
                daily[day]["cost"] += entry["estimated_cost_usd"]
                daily[day]["calls"] += 1
                daily[day]["tokens"] += entry["input_tokens"] + entry["output_tokens"]

        return [
            {"date": day, **data}
            for day, data in sorted(daily.items())
        ]


# Singleton
cost_tracker = CostTracker()


# === Usage example ===
#
# from observability.cost_tracker import cost_tracker
#
# # After each LLM call:
# response = llm.chat(messages)
# cost_tracker.log_usage(
#     model="claude-sonnet-4-20250514",
#     input_tokens=response.usage.input_tokens,
#     output_tokens=response.usage.output_tokens,
#     session_id="user_123",
# )
#
# # Check spending:
# print(cost_tracker.get_session_summary("user_123"))
# print(cost_tracker.get_total_summary())
''')


def _generate_dashboard_info(config: dict, base: Path):
    """Generate observability documentation and comparison."""

    obs = config.get("observability", "none")

    _write(base / "README.md", f'''# Observability — {config["project_name"]}

**Current provider: {obs}**

## What's Being Tracked

Every agent interaction generates traces covering:

| Signal | What's captured | Why it matters |
|--------|----------------|----------------|
| **LLM Calls** | Prompt, response, tokens, latency, model | Debug responses, optimize costs |
| **Retrieval** | Query, results, relevance scores | Find RAG quality issues |
| **Tool Calls** | Tool name, input, output, errors | Debug integrations |
| **Agent Steps** | Reasoning loop iterations, decisions | Understand agent behavior |
| **Costs** | Token counts, estimated USD spend | Budget management |
| **Errors** | Exceptions, timeouts, rate limits | Reliability monitoring |

## Provider Comparison

| Feature | LangFuse | LangSmith | OpenTelemetry |
|---------|----------|-----------|---------------|
| **Open source** | Yes (MIT) | No | Yes (Apache 2.0) |
| **Self-hostable** | Yes (Docker) | No | Yes (many backends) |
| **LLM-specific features** | Excellent | Excellent | Manual setup |
| **Prompt management** | Yes | Yes (Playground) | No |
| **Auto-tracing LangChain** | Yes | Yes (native) | Via instrumentor |
| **Evaluation tools** | Yes (scores, datasets) | Yes (datasets, annotators) | No |
| **Cost tracking** | Built-in | Built-in | Manual |
| **Non-LLM tracing** | Limited | Limited | Excellent |
| **Enterprise ready** | Growing | Yes | Yes |
| **Pricing** | Free tier + paid | Free tier + paid | Free (self-host) |
| **Vendor lock-in risk** | Low (open source) | Medium | None |

## Setup Guide

### LangFuse (Recommended for most teams)
```bash
# Cloud (fastest)
# 1. Sign up at https://cloud.langfuse.com
# 2. Create project, get API keys
# 3. Add to .env:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Self-hosted (full control)
docker compose -f docker-compose-langfuse.yml up -d
# Then set LANGFUSE_HOST=http://localhost:3000
```

### LangSmith (Best for LangChain users)
```bash
# 1. Sign up at https://smith.langchain.com
# 2. Get API key
# 3. Add to .env:
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT={config["project_name"]}
# If using LangChain, tracing is now automatic — no code changes needed
```

### OpenTelemetry (Best for enterprise / multi-service)
```bash
# Local dev with Jaeger:
docker run -d --name jaeger \\
  -p 16686:16686 \\
  -p 4317:4317 \\
  jaegertracing/all-in-one:latest

# Add to .env:
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME={config["project_name"]}

# View traces at http://localhost:16686
```

## Cost Tracking

The built-in cost tracker works regardless of your observability provider.
It logs every LLM call to `observability/cost_log.jsonl` and estimates costs
based on current model pricing.

```python
from observability.cost_tracker import cost_tracker

# Get spending summary
print(cost_tracker.get_total_summary())
print(cost_tracker.get_daily_report())
```

## Key Metrics to Monitor

**Reliability**: Error rate, timeout rate, retry rate
**Performance**: P50/P95/P99 latency, tokens per second
**Quality**: Retrieval relevance, user feedback scores, hallucination rate
**Cost**: Daily/weekly spend, cost per conversation, cost per user
**Usage**: Conversations per day, tool usage distribution, peak hours
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
