"""Shared builder for the generated agents/__init__.py entrypoint module.

Every framework adapter exposes the same interface in generated projects:

    from agents import agent
    result = agent.run(user_message, session_id)   # -> {"response": str, ...}
    agent.reset(session_id)

Guardrails (ingress/egress) and observability tracing are wired here, in one
place, so every framework gets identical behavior regardless of how the
inner orchestration works.
"""
from textwrap import indent


def build_entrypoint(
    config,
    inner_import: str,
    reset_body: str,
    framework_note: str,
) -> str:
    """Render the agents/__init__.py content for a framework adapter.

    Args:
        config: ProjectConfig.
        inner_import: import statement binding the framework's run function
            as ``_run`` (signature: ``_run(user_message, session_id) -> dict``).
        reset_body: statements for AgentEntrypoint.reset (unindented).
        framework_note: one-line description of the inner orchestration.
    """
    guardrail_import = ""
    if config["include_guardrails"]:
        guardrail_import = (
            "from guardrails.input_filters import check_input\n"
            "from guardrails.output_filters import check_output\n"
        )

    obs = config.get("observability", "none")
    tracer_import = ""
    if obs in ("langfuse", "langsmith", "opentelemetry"):
        tracer_import = "from observability.tracer import tracer\n"

    if obs == "langfuse":
        traced_run = '''def _traced_run(user_message: str, session_id: str) -> dict:
    trace = tracer.trace_agent_run(user_message, session_id)
    result = _run(user_message, session_id)
    response = result["response"] if isinstance(result, dict) else str(result)
    trace.set_output({"response": response})
    tracer.flush()
    return result'''
    elif obs == "langsmith":
        traced_run = '''@tracer.traced("agent_run")
def _traced_run(user_message: str, session_id: str) -> dict:
    return _run(user_message, session_id)'''
    elif obs == "opentelemetry":
        traced_run = '''def _traced_run(user_message: str, session_id: str) -> dict:
    with tracer.trace_agent_run(user_message, session_id):
        return _run(user_message, session_id)'''
    else:
        traced_run = '''def _traced_run(user_message: str, session_id: str) -> dict:
    return _run(user_message, session_id)'''

    # Assemble the run() method body with explicit 8-space indentation
    run_body_lines = []
    if config["include_guardrails"]:
        run_body_lines += [
            "input_check = check_input(user_message)",
            'if not input_check["safe"]:',
            "    return {",
            '        "response": input_check["message"],',
            '        "blocked": True,',
            '        "session_id": session_id,',
            "    }",
        ]
    run_body_lines += [
        "result = _traced_run(user_message, session_id)",
        'response_text = result["response"] if isinstance(result, dict) else str(result)',
    ]
    if config["include_guardrails"]:
        run_body_lines += [
            "output_check = check_output(response_text)",
            'if not output_check["safe"]:',
            '    response_text = output_check["filtered_text"]',
        ]
    run_body_lines += [
        "out = result if isinstance(result, dict) else {}",
        'out["response"] = response_text',
        'out.setdefault("session_id", session_id)',
        "return out",
    ]
    run_body = indent("\n".join(run_body_lines), "        ")
    reset_indented = indent(reset_body.strip(), "        ")

    return f'''"""Agent entrypoint — the uniform interface used by the UI, API, CLI, and eval layers.

{framework_note}
Guardrails and tracing are applied here so ingress/egress behavior is
identical for every framework.
"""
{inner_import}
{guardrail_import}{tracer_import}

{traced_run}


class AgentEntrypoint:
    """Uniform agent interface: run(user_message, session_id) -> dict."""

    def run(self, user_message: str, session_id: str = "default") -> dict:
{run_body}

    def reset(self, session_id: str = "default"):
        """Clear conversation state for a session, where the framework supports it."""
{reset_indented}


agent = AgentEntrypoint()
'''
