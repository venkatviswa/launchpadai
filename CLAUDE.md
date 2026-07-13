# CLAUDE.md

Project: **launchpadai — multi-framework agent scaffolding CLI** (Typer + programmatic generators).
Generates runnable, multi-agent-capable AI projects across orchestration frameworks (`--framework plain | langgraph | crewai | agentscript`) with a pluggable retrieval layer (`--retrieval custom | llamaindex`). Invoked as `launchpad init` (interactive wizard or fully non-interactive via flags).

These behavioral guidelines reduce common LLM coding mistakes (adapted from Andrej Karpathy's observations on LLM coding pitfalls, via multica-ai/andrej-karpathy-skills). They bias toward caution over speed — for trivial tasks, use judgment.

---

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly before implementing. If uncertain, ask.
- If multiple interpretations of the request exist, present them — never pick one silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name the confusion, and ask before writing code.

## 2. Simplicity First

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No unrequested "flexibility" or "configurability."
- No error handling for scenarios that cannot occur.
- If 200 lines could be 50, rewrite. Test: would a senior engineer call this overcomplicated?

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match the existing style even if you'd do it differently.
- If you spot unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR change orphaned; leave pre-existing dead code alone unless asked.
- Every changed line must trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

- Convert vague tasks into verifiable goals:
  - "Add validation" → write tests for invalid inputs, make them pass.
  - "Fix the bug" → write a failing test that reproduces it, make it pass.
  - "Refactor X" → tests green before AND after.
- For multi-step work, state a brief plan where each step has an explicit verification check, then execute the loop independently:

  ```
  1. [Step] → verify: [check]
  2. [Step] → verify: [check]
  3. [Step] → verify: [check]
  ```

- Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## 5. Type Safety: Pydantic v2

Configuration crossing a boundary MUST be a Pydantic v2 model — no bare dicts for anything the CLI hands to generators or that adapters consume.

- The boundary models live in `launchpadai/config.py` (`ProjectConfig`, `AgentSpec`) and `launchpadai/frameworks/base.py` (`FrameworkAdapter`). Extend these; do not reintroduce dict configs.
- Use the v2 API only: `model_validate()`, `model_dump()`, `ConfigDict`, `@field_validator` / `@model_validator`, `Field()` for constraints. Never v1 patterns (`.dict()`, `.parse_obj()`, `class Config`, `@validator`).
- Frozen models (`ConfigDict(frozen=True)`) for immutable value objects (`AgentSpec`, `FrameworkAdapter`).
- Validate at the edge, fail fast: the CLI builds a `ProjectConfig` before any generator runs; framework-dependent rules live in `frameworks/registry.py::validate_config`.
- `ProjectConfig` keeps dict-style access (`config["key"]`, `.get()`) as a compatibility shim for generators — new code should prefer attribute access.

## 6. Project-Specific Standards

- **CLI:** Typer with typed arguments; every command has `--help` text and a non-interactive mode (`launchpad init --defaults` plus per-option flags) so AI agents and CI can drive it. New options must be wired into BOTH the wizard (`cli/prompts.py`) and the flags (`cli/main.py`).
- **Generation is programmatic:** generators under `launchpadai/generators/` write files as Python string templates (no Copier/Jinja on disk). Keep generated code readable — it is the user's starting codebase, not throwaway output.
- **Generated projects must be runnable out of the box** — agent with one example tool, uniform entrypoint, passing imports, and a generated pytest suite that passes offline against the mock LLM provider (`LLM_MOCK=1`, `models/llm/mock.py`). Never empty skeletons, and never `NotImplementedError` stubs behind a wizard option: if an option is offered, its generated code must work.
- **Generated structure:** vertical slice per agent (`agents/<name>/` with `prompts/system.md` and `tools.py`), plus a uniform `agents/__init__.py` entrypoint (`agent.run(...)`) shared by UI/API/CLI/eval. Shared layers (`tools/`, `models/`, `guardrails/`, `memory/`, `observability/`, `knowledge/`) are framework-agnostic; only orchestration files inside `agents/` vary by adapter.
- **Guardrails and tracing wire at the entrypoint** (`frameworks/_entrypoint.py`), so every framework gets identical ingress/egress behavior. API auth (`simple`/`multi_user`) wires into `api/routes.py` (auth router + `require_auth` on `/chat`), verified by generated tests. Compliance is enforced by generated code, not TODO comments.
- **Testing:** pytest. Every adapter has structural generation tests (unit + validation) and a render → install → import → run-generated-tests smoke test (marked `slow`, run per-adapter in CI). The generated tests execute offline via the mock provider. Run tests before declaring done.
- **Tooling:** uv for dependency management (`uv sync`, `uv run pytest`). Python 3.10+.
- **Prompts are artifacts:** generated prompts live in versioned `prompts/` directories (project-level and per-agent-slice), never inline strings scattered in code.
- **No secrets in templates or examples.** Use `.env.example` with placeholder values only.

## 7. Framework Adapter Architecture (mandatory)

launchpadai is multi-framework by design. Frameworks are **adapters**, never hardcoded branches in core generators.

- **Adapter contract:** each framework is a module in `launchpadai/frameworks/` exposing a frozen `FrameworkAdapter` instance (`name`, `display_name`, `tier`, `description`, `orchestrations`, `generate`) and registered in `frameworks/registry.py`. The CLI derives `--framework` choices from the registry at runtime — never from a hardcoded list.
- **Adding a framework** = one adapter module + one registry entry + dependency mappings in `generators/requirements.py` + tests. Zero changes to core generators (they call `get_adapter(config.framework).generate(...)`).
- **The adapter owns** the orchestration files inside `agents/` and the `agents/__init__.py` entrypoint (built via `frameworks/_entrypoint.py` so guardrails/tracing stay uniform). Core owns everything else.
- **Orchestration modes:** `single`, `sequential`, `supervisor`. An adapter declares which it supports; `validate_config` rejects unsupported combinations. Multi-agent teams come from `ProjectConfig.agents` (list of `AgentSpec`) — adapters must build orchestration from that list, never hardcode agent names.
- **Tier 1 (fully supported, tested):**
  - `plain` → explicit loop (`agents/base.py`) + `agents/pipeline.py` orchestrator (reference implementation)
  - `langgraph` → `agents/graph.py` (StateGraph, checkpointer, supervisor routing via conditional edges)
  - `crewai` → `agents/crew.py` (Agent/Task per spec, `Process.sequential` or `hierarchical`, LLM wired via LiteLLM prefix)
- **Tier 2 (maintained, reduced surface):** `agentscript` (Salesforce Agentforce DX; single-agent only — orchestration happens inside Salesforce topics).
- **LlamaIndex is a retrieval-layer option, not a framework adapter:** exposed as `--retrieval llamaindex`, pairable with any orchestrator, implementing the same `retrieve`/`format_context` interface as the custom pipeline. Do not model it as a peer of LangGraph/CrewAI.
- **Version floors:** dependency floors live in `generators/requirements.py` and must match the APIs the generated code actually calls (e.g. langfuse is pinned `<3` because the generated tracer targets the v2 SDK). When an upstream API changes, update the generated code and the floor together.
- When working on framework-specific generated code, do not guess APIs from memory — check the pinned version's docs; these frameworks ship breaking changes frequently.

---

**These guidelines are working if:** diffs contain fewer unnecessary changes, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
