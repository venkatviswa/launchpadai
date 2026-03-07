"""Generate the guardrails layer — input/output safety."""
from pathlib import Path


def generate_guardrails_layer(config: dict, project_path: Path):
    if not config["include_guardrails"]:
        return

    base = project_path / "guardrails"
    _write(base / "__init__.py", "")

    _write(base / "input_filters.py", '''"""Input filters — check user messages before processing.

Catches prompt injection, PII in input, and blocked topics.
"""


# Simple blocklist — replace with more sophisticated detection
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your instructions",
    "you are now",
    "act as if",
    "pretend you are",
]


def check_input(text: str) -> dict:
    """Check user input for safety issues.

    Returns:
        {"safe": True/False, "message": str, "flags": list}
    """
    flags = []
    text_lower = text.lower()

    # Check for prompt injection attempts
    for pattern in INJECTION_PATTERNS:
        if pattern in text_lower:
            flags.append(f"prompt_injection: {pattern}")

    if flags:
        return {
            "safe": False,
            "message": "I'm not able to process that request. Please rephrase your question.",
            "flags": flags,
        }

    return {"safe": True, "message": "", "flags": []}


def detect_pii(text: str) -> list[dict]:
    """Detect PII in text using Presidio (if available)."""
    try:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        return [{"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results]
    except ImportError:
        return []
''')

    _write(base / "output_filters.py", '''"""Output filters — check agent responses before sending to user."""


def check_output(text: str) -> dict:
    """Check agent output for safety and quality.

    Returns:
        {"safe": True/False, "filtered_text": str, "flags": list}
    """
    flags = []

    # Check for hallucination markers
    hallucination_phrases = [
        "as an ai language model",
        "i don't have access to real-time",
        "my training data",
    ]

    for phrase in hallucination_phrases:
        if phrase in text.lower():
            flags.append(f"potential_hallucination_marker: {phrase}")

    # Check for PII leakage in output
    # TODO: Add PII detection on output

    return {
        "safe": len(flags) == 0,
        "filtered_text": text,
        "flags": flags,
    }
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
