"""Generate the memory layer — conversation history and state."""
from pathlib import Path


def generate_memory_layer(config: dict, project_path: Path):
    base = project_path / "memory"
    _write(base / "__init__.py", "")

    _write(base / "conversation.py", '''"""Conversation Memory — maintains chat history per session.

This provides short-term memory within a conversation.
For long-term memory across sessions, integrate with a database.
"""
from collections import defaultdict


class ConversationMemory:
    """In-memory conversation history manager."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._history: dict[str, list[dict]] = defaultdict(list)

    def add(self, session_id: str, role: str, content: str):
        """Add a message to the conversation history."""
        self._history[session_id].append({"role": role, "content": content})

        # Trim to max turns (keep system messages)
        history = self._history[session_id]
        if len(history) > self.max_turns * 2:
            self._history[session_id] = history[-(self.max_turns * 2):]

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        return self._history.get(session_id, [])

    def clear(self, session_id: str):
        """Clear history for a session."""
        self._history.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._history.keys())
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
