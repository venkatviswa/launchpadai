"""Generate the evaluation framework."""
from pathlib import Path


def generate_eval_layer(config: dict, project_path: Path):
    if not config["include_eval"]:
        return

    base = project_path / "eval"
    _write(base / "__init__.py", "")

    _write(base / "datasets" / "test_cases.yaml", '''# Evaluation test cases
# Each case has an input, expected behavior, and optional ground truth

test_cases:
  - name: "basic_greeting"
    input: "Hello!"
    expected_behavior: "Responds with a friendly greeting"
    should_contain: ["hello", "hi", "hey"]

  - name: "out_of_scope"
    input: "What's the meaning of life?"
    expected_behavior: "Acknowledges the question without making up domain-specific info"
    should_not_contain: ["according to our documents"]

  - name: "rag_retrieval"
    input: "What is our return policy?"
    expected_behavior: "Retrieves and references actual policy documents"
    requires_context: true
''')

    _write(base / "run_eval.py", '''"""Evaluation runner — test agent against evaluation datasets."""
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_cases(path: str = "eval/datasets/test_cases.yaml") -> list[dict]:
    """Load test cases from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("test_cases", [])


def run_evaluation():
    """Run all test cases and report results."""
    from agents.base import agent

    cases = load_test_cases()
    results = []

    print(f"Running {len(cases)} test cases...\\n")

    for case in cases:
        print(f"  Testing: {case['name']}")
        try:
            response = agent.run(case["input"], session_id=f"eval_{case['name']}")
            text = response["response"].lower()

            passed = True
            notes = []

            # Check should_contain
            for term in case.get("should_contain", []):
                if term.lower() not in text:
                    passed = False
                    notes.append(f"Missing: '{term}'")

            # Check should_not_contain
            for term in case.get("should_not_contain", []):
                if term.lower() in text:
                    passed = False
                    notes.append(f"Unexpected: '{term}'")

            status = "PASS" if passed else "FAIL"
            results.append({"name": case["name"], "status": status, "notes": notes})
            print(f"    [{status}] {', '.join(notes) if notes else 'OK'}")

        except Exception as e:
            results.append({"name": case["name"], "status": "ERROR", "notes": [str(e)]})
            print(f"    [ERROR] {e}")

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\\nResults: {passed}/{total} passed")

    return results


if __name__ == "__main__":
    run_evaluation()
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
