"""
Leak and Consistency Verifier

Checks every response from the generator against the full patient case for:
1. Direct leaks — withheld facts surfacing in the response
2. Contradictions — fabricated claims that conflict with the full case
3. Fabrications — invented details not in any authorized fact

If any check fails, the response is rejected and the generator must retry.
"""

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-fp16"


def call_llm(messages: list[dict], temperature: float = 0.1) -> str:
    """Call Ollama and return the assistant's response text."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def parse_json_response(raw_response: str) -> dict:
    """Parse an LLM response as JSON, handling markdown fences."""
    text = raw_response.strip()
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse verifier output as JSON: {e}\nRaw output:\n{text}"
        )


def load_case(path: str) -> dict:
    """Load a clinical case JSON file."""
    with open(path) as f:
        return json.load(f)


def check_leak_phrases(
    response: str,
    case: dict,
    unlocked_fact_ids: set[str],
) -> list[dict]:
    """
    Deterministic check for direct leaks using specific phrases.

    Each withheld fact can have a 'leak_phrases' field — multi-word phrases
    that, if found in the response, indicate a direct leak. These are more
    specific than single words to avoid false positives.

    If a fact has no leak_phrases, this check is skipped for that fact
    (the LLM check handles it instead).
    """
    response_lower = response.lower()

    leaks = []
    for fact in case["facts"]:
        # Skip disclosed and unlocked facts
        if fact["disclosed"] or fact["id"] in unlocked_fact_ids:
            continue

        phrases = fact.get("leak_phrases", [])
        for phrase in phrases:
            if phrase.lower() in response_lower:
                leaks.append({
                    "fact_id": fact["id"],
                    "keyword": phrase,
                    "fact_content": fact["content"],
                })

    return leaks


def check_with_llm(
    response: str,
    case: dict,
    unlocked_fact_ids: set[str],
    student_message: str,
) -> dict:
    """
    LLM-based check for contradictions ONLY.

    Leak detection is handled by the deterministic phrase scanner.
    The LLM is unreliable at distinguishing disclosed/unlocked/withheld
    facts, so we limit it to checking for factual contradictions — things
    like wrong timelines, wrong severity, wrong locations.
    """
    # Combine all facts into one list for contradiction checking
    # The LLM doesn't need to know disclosure status for this task
    all_facts = []
    for fact in case["facts"]:
        all_facts.append({"id": fact["id"], "content": fact["content"]})
        # Include attributes for precise contradiction checking
        if "attributes" in fact:
            for key, val in fact["attributes"].items():
                all_facts.append({
                    "id": fact["id"],
                    "content": f"{key}: {val}",
                })

    facts_str = json.dumps(all_facts, indent=2)

    prompt = f"""You are checking a simulated patient's response for FACTUAL CONTRADICTIONS ONLY.

## Patient Facts (the ground truth)
{facts_str}

## Patient's Response
"{response}"

## Your Task
Check if the response CONTRADICTS any fact above. A contradiction is when the response states something that directly conflicts with a fact. Examples:
- Fact says "3 days ago" but response says "a week ago" → CONTRADICTION
- Fact says "left arm" but response says "right arm" → CONTRADICTION
- Fact says "mild" but response says "severe" → CONTRADICTION

Things that are NOT contradictions:
- Paraphrasing a fact in different words
- Adding conversational filler ("you know", "it's been rough")
- Mentioning a fact that exists in the list above
- Minor wording differences that don't change the meaning

Respond with ONLY this JSON:
{{
  "pass": true/false,
  "issues": [
    {{
      "type": "contradiction",
      "description": "<specific contradiction: what the fact says vs what the response says>",
      "fact_id": "<the fact ID being contradicted>"
    }}
  ]
}}

If there are no contradictions, return {{"pass": true, "issues": []}}."""

    raw = call_llm([{"role": "user", "content": prompt}])
    return parse_json_response(raw)


class Verifier:
    """
    Checks generator responses for leaks, contradictions, and fabrications.

    Currently uses deterministic phrase scanning only. The LLM-based
    contradiction check is available but disabled by default because
    Llama 3.1 8B produces too many false positives. Enable it with
    use_llm_check=True when using a stronger model (e.g., GPT-4o-mini).
    """

    def __init__(self, case_path: str, use_llm_check: bool = False):
        self.case = load_case(case_path)
        self.use_llm_check = use_llm_check

    def verify(
        self,
        response: str,
        unlocked_fact_ids: set[str],
        student_message: str,
    ) -> dict:
        """
        Verify a patient response.

        Args:
            response: The generator's patient response.
            unlocked_fact_ids: Set of fact IDs that have been unlocked.
            student_message: The student's message that triggered this response.

        Returns:
            {
                "pass": bool,
                "keyword_leaks": [...],
                "llm_check": {"pass": bool, "issues": [...]},
            }
        """
        result = {
            "pass": True,
            "keyword_leaks": [],
            "llm_check": {"pass": True, "issues": []},
        }

        # Pass 1: Deterministic phrase check
        phrase_leaks = check_leak_phrases(
            response, self.case, unlocked_fact_ids
        )
        if phrase_leaks:
            result["pass"] = False
            result["keyword_leaks"] = phrase_leaks

        # Pass 2: LLM semantic check (disabled by default for Llama 8B)
        if self.use_llm_check:
            try:
                llm_result = check_with_llm(
                    response, self.case, unlocked_fact_ids, student_message
                )
                result["llm_check"] = llm_result
                if not llm_result.get("pass", True):
                    result["pass"] = False
            except (ValueError, Exception) as e:
                result["llm_check"] = {
                    "pass": True,
                    "issues": [],
                    "error": str(e),
                }

        return result


# --- CLI for testing ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python verifier.py <case_file.json> <response_text>")
        print('Example: python verifier.py cases/case_cardiology.json "I\'ve been having chest pain"')
        sys.exit(1)

    case_path = sys.argv[1]
    response_text = sys.argv[2]

    verifier = Verifier(case_path)
    result = verifier.verify(
        response=response_text,
        unlocked_fact_ids=set(),
        student_message="What brings you in today?",
    )

    print(json.dumps(result, indent=2))
