"""
Response Generator

Takes a content brief from the disclosure planner and produces natural
patient dialogue. This component NEVER sees the full patient case —
only the content brief.

This is the core of the information isolation architecture: because the
generator has no access to withheld facts, it cannot leak them regardless
of how the student phrases their questions.
"""

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-fp16"


def call_llm(messages: list[dict], temperature: float = 0.7) -> str:
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


def build_generator_prompt(
    brief: dict,
    conversation_history: list[dict],
) -> list[dict]:
    """
    Build the prompt for the response generator.

    The generator sees ONLY:
    - The content brief (authorized facts, tone notes, hedging rule)
    - The conversation history
    - The student's current message

    It does NOT see the full case, withheld facts, or disclosure conditions.
    """
    # Format authorized positives by relevance
    directly_relevant = [
        f for f in brief["authorized_positives"]
        if f.get("relevance") == "directly_relevant"
    ]
    tangentially_relevant = [
        f for f in brief["authorized_positives"]
        if f.get("relevance") == "tangentially_relevant"
    ]
    not_relevant = [
        f for f in brief["authorized_positives"]
        if f.get("relevance") == "not_relevant"
    ]

    def format_facts(facts):
        return "\n".join(f"- {f['content']}" for f in facts) if facts else "(none)"

    # Format newly unlocked facts
    if brief["newly_unlocked"]:
        newly_unlocked_str = "\n".join(
            f"- {f['content']}" for f in brief["newly_unlocked"]
        )
    else:
        newly_unlocked_str = "(none)"

    # Format authorized negatives
    if brief["authorized_negatives"]:
        negatives_str = "\n".join(
            f"- {f['content']}" for f in brief["authorized_negatives"]
        )
    else:
        negatives_str = "(none this turn)"

    # Format conversation history
    if conversation_history:
        conv_str = "\n".join(
            f"{'Student' if msg['role'] == 'student' else 'Patient'}: {msg['content']}"
            for msg in conversation_history
        )
    else:
        conv_str = "(This is the first turn.)"

    tone = brief.get("tone_notes", {})
    hedging = brief.get("hedging_rule", {})

    system_prompt = f"""You are a simulated patient in a medical interview with a student doctor.

## Your Persona
- {tone.get('persona', 'Cooperative patient')}
- Current emotional state: {tone.get('emotional_state', 'neutral')}
- Speech style: {tone.get('speech_style', 'casual')}

## Facts You Can Share

DIRECTLY RELEVANT to the student's question (use these to answer):
{format_facts(directly_relevant)}

TANGENTIALLY RELEVANT (mention only if it flows naturally):
{format_facts(tangentially_relevant)}

BACKGROUND FACTS (you know these but don't volunteer them):
{format_facts(not_relevant)}

## New Information to Introduce This Turn
{newly_unlocked_str}
If new information is listed above, include it in your response — the student's question triggered it.

## Things You Can Deny
{negatives_str}

## CRITICAL RULES — READ CAREFULLY

1. ONLY use facts listed above. Do NOT invent, embellish, or add ANY details not explicitly listed. No made-up timelines, no extra symptoms, no imagined scenarios. If a fact says "chest tightness for 3 days" you say 3 days, not "a month" or "a while."

2. Keep responses to 1-2 sentences. Real patients answer the question asked and stop talking.

3. Use the EXACT details from the facts: correct durations, correct severity numbers, correct onset times. Do not paraphrase numbers or timeframes.

4. {hedging.get('instruction', 'If asked about something not listed above, say you are not sure.')}

5. Do NOT use medical jargon. You are a patient, not a doctor.

6. Do NOT volunteer information the student hasn't asked about.

7. If new information is listed above but the response would be longer than 2 sentences, prioritize the new information over repeating old facts."""

    user_prompt = f"""## Conversation So Far
{conv_str}

## Student's Current Message
"{brief['student_message']}"

Respond as the patient:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_response(
    brief: dict,
    conversation_history: list[dict],
) -> str:
    """
    Generate a patient response given a content brief and conversation history.

    Args:
        brief: Content brief from the disclosure planner.
        conversation_history: List of prior messages [{"role": "student"|"patient", "content": "..."}]

    Returns:
        The patient's response as a string.
    """
    messages = build_generator_prompt(brief, conversation_history)
    response = call_llm(messages)

    # Clean up: remove any quotation marks wrapping the entire response
    response = response.strip()
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]

    return response


# --- CLI for testing ---

if __name__ == "__main__":
    import sys

    # Accept a content brief JSON file for testing
    if len(sys.argv) < 2:
        print("Usage: python generator.py <brief.json>")
        print("Or pipe a brief via stdin: echo '{...}' | python generator.py -")
        sys.exit(1)

    if sys.argv[1] == "-":
        brief = json.load(sys.stdin)
    else:
        with open(sys.argv[1]) as f:
            brief = json.load(f)

    # Generate with empty conversation history
    response = generate_response(brief, [])
    print(f"Patient: {response}")
