"""Generate the prompts layer — system prompts and templates."""
from pathlib import Path


def generate_prompts_layer(config: dict, project_path: Path):
    """Generate prompt template files."""
    base = project_path / "prompts"

    # System prompt
    _write(base / "system" / "default.md", f"""You are a helpful AI assistant for {config['project_name']}.

## Your Role
{config['agent_description']}

## Instructions
- Answer questions accurately based on the provided context
- If you don't know something, say so honestly — do not make up information
- When using tools, explain what you're doing and why
- Be concise but thorough
- Cite sources when referencing specific documents

## Context
Use the following retrieved documents to answer the user's question.
If the documents don't contain relevant information, say so.

{{context}}
""")

    # Few-shot examples
    _write(base / "few_shot" / "examples.yaml", """# Few-shot examples — teach the agent by demonstration
# Add input/output pairs that show desired behavior

examples:
  - input: "What is our return policy?"
    output: "Based on our policy documents, items can be returned within 30 days of purchase with original packaging. Would you like more details about specific product categories?"

  - input: "I need help with something not in the docs"
    output: "I don't have specific information about that in my knowledge base. Let me connect you with a team member who can help. Is there anything else I can assist with?"
""")

    # Template assembly
    _write(base / "__init__.py", "")
    _write(base / "templates.py", '''"""Prompt template assembly — builds the final prompt sent to the LLM.

This is where user input, system prompt, retrieved context,
and few-shot examples are combined into the complete prompt.
"""
from pathlib import Path
import yaml


PROMPTS_DIR = Path(__file__).parent


def load_system_prompt(name: str = "default") -> str:
    """Load a system prompt by name from prompts/system/."""
    path = PROMPTS_DIR / "system" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text()


def load_few_shot_examples(name: str = "examples") -> list[dict]:
    """Load few-shot examples from YAML."""
    path = PROMPTS_DIR / "few_shot" / f"{name}.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("examples", [])


def build_messages(
    user_message: str,
    context: str = "",
    system_prompt_name: str = "default",
    include_few_shot: bool = True,
    conversation_history: list[dict] = None,
) -> list[dict]:
    """Build the complete message list for the LLM.

    This assembles:
    1. System prompt (with context injected)
    2. Few-shot examples (optional)
    3. Conversation history (optional)
    4. Current user message
    """
    # Load and fill system prompt
    system = load_system_prompt(system_prompt_name)
    system = system.replace("{{context}}", context or "No context provided.")

    messages = [{"role": "system", "content": system}]

    # Add few-shot examples
    if include_few_shot:
        examples = load_few_shot_examples()
        for ex in examples:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})

    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    return messages
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
