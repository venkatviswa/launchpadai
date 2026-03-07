"""Generate CLI runner for non-UI agent interaction."""
from pathlib import Path


def generate_cli_runner(config: dict, project_path: Path):
    _write(project_path / "agents" / "cli_runner.py", f'''"""CLI Runner — interact with the agent from the terminal.

Run with: python -m agents.cli_runner
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import agent


def main():
    """Interactive CLI chat with the agent."""
    print("🤖 {config['project_name']}")
    print("{config['agent_description']}")
    print("Type 'quit' to exit, 'clear' to reset conversation.\\n")

    session_id = "cli"

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            agent.memory.clear(session_id)
            print("Conversation cleared.\\n")
            continue

        try:
            result = agent.run(user_input, session_id=session_id)
            response = result["response"] if isinstance(result, dict) else str(result)
            print(f"\\nAgent: {{response}}\\n")
        except Exception as e:
            print(f"\\nError: {{e}}\\n")


if __name__ == "__main__":
    main()
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
