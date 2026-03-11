"""Generate the guardrails layer — input/output safety."""
from pathlib import Path


def generate_guardrails_layer(config: dict, project_path: Path):
    if not config["include_guardrails"]:
        return

    base = project_path / "guardrails"
    _write(base / "__init__.py", "")

    _write(base / "input_filters.py", '''"""Input filters — check user messages before processing.

Catches prompt injection, PII in input, and blocked topics.

OWASP Agentic AI reference:
- Prevents direct prompt injection attacks (OWASP LLM01)
- Detects PII to prevent sensitive data exposure (OWASP LLM06)
"""
import re
import logging

logger = logging.getLogger(__name__)


# Blocklist patterns — expanded set with regex support
INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\\s+(all\\s+)?(previous|prior|above|system)\\s+instructions?",
    r"disregard\\s+(all\\s+)?(your|prior|previous|above)\\s+instructions?",
    r"forget\\s+(all\\s+)?(your|prior|previous|above)\\s+instructions?",
    r"override\\s+(all\\s+)?(your|prior|previous|above)\\s+instructions?",
    # Role manipulation
    r"you\\s+are\\s+now\\b",
    r"act\\s+as\\s+if\\b",
    r"pretend\\s+(you\\s+are|to\\s+be)\\b",
    r"from\\s+now\\s+on\\s+you\\s+are\\b",
    r"i\\s+want\\s+you\\s+to\\s+act\\s+as\\b",
    r"switch\\s+to\\s+.*\\s+mode",
    r"enter\\s+.*\\s+mode",
    # System prompt extraction
    r"(print|show|reveal|display|output|repeat)\\s+(your|the)\\s+(system|initial|original)\\s+(prompt|instructions?|message)",
    r"what\\s+(are|is|were)\\s+your\\s+(system|original|initial)\\s+(prompt|instructions?)",
    # Delimiter injection
    r"\\[/?INST\\]",
    r"\\[/?SYS(TEM)?\\]",
    r"<\\|?(system|im_start|im_end|endof)\\|?>",
    # Encoding evasion markers
    r"base64\\s*:\\s*[A-Za-z0-9+/=]{20,}",
]

# Pre-compile patterns for performance
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_input(text: str) -> dict:
    """Check user input for safety issues.

    Returns:
        {"safe": True/False, "message": str, "flags": list}
    """
    flags = []

    # Check for prompt injection attempts using regex patterns
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            flags.append(f"prompt_injection: matched pattern near '{match.group()[:50]}'")

    # Check for excessively long input (potential resource exhaustion)
    if len(text) > 50000:
        flags.append("excessive_length: input exceeds 50000 characters")

    # Check for high ratio of special characters (potential encoding attack)
    if len(text) > 10:
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_ratio > 0.5:
            flags.append("suspicious_encoding: high special character ratio")

    if flags:
        logger.warning(f"Input filter triggered: {flags}")
        return {
            "safe": False,
            "message": "I\'m not able to process that request. Please rephrase your question.",
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

    _write(base / "output_filters.py", '''"""Output filters — check agent responses before sending to user.

OWASP Agentic AI reference:
- Prevents sensitive information disclosure (OWASP LLM06)
- Detects hallucination markers for quality (OWASP LLM03)
- Checks for PII leakage in LLM output
"""
import re
import logging

logger = logging.getLogger(__name__)


def check_output(text: str) -> dict:
    """Check agent output for safety and quality.

    Returns:
        {"safe": True/False, "filtered_text": str, "flags": list}
    """
    flags = []
    filtered = text

    # Check for hallucination markers
    hallucination_phrases = [
        "as an ai language model",
        "i don\'t have access to real-time",
        "my training data",
        "i was trained by",
        "my knowledge cutoff",
    ]

    for phrase in hallucination_phrases:
        if phrase in text.lower():
            flags.append(f"potential_hallucination_marker: {phrase}")

    # Check for PII leakage in output using regex patterns
    pii_patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
        "phone": r"\\b(?:\\+?1[-.\\s]?)?\\(?[0-9]{3}\\)?[-.\\s]?[0-9]{3}[-.\\s]?[0-9]{4}\\b",
        "ssn": r"\\b\\d{3}-\\d{2}-\\d{4}\\b",
        "credit_card": r"\\b(?:\\d{4}[-.\\s]?){3}\\d{4}\\b",
    }

    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, filtered)
        if matches:
            flags.append(f"potential_pii_leakage: {pii_type} detected ({len(matches)} instance(s))")
            # Redact PII from output
            filtered = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", filtered)

    # Check for leaked system prompt fragments
    system_leak_patterns = [
        r"my\\s+system\\s+prompt\\s+(is|says|reads)",
        r"here\\s+(is|are)\\s+my\\s+instructions?",
    ]
    for pattern in system_leak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append("potential_system_prompt_leak")

    if flags:
        logger.warning(f"Output filter triggered: {flags}")

    return {
        "safe": len(flags) == 0,
        "filtered_text": filtered,
        "flags": flags,
    }
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
