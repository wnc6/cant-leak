"""
Disclosure Planner

Takes the full patient case + conversation history and outputs a content brief
for the response generator. The planner is the ONLY component that sees the
full case including withheld facts.

The generator receives only the content brief and can never leak withheld facts
because it never sees them.

Architecture note: The LLM is used ONLY to evaluate disclosure conditions and
determine tone. All brief assembly (authorized_positives, authorized_negatives,
newly_unlocked) is done deterministically in Python to avoid LLM errors.
"""

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-fp16"


def call_llm(messages: list[dict], temperature: float = 0.3) -> str:
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


def load_case(path: str) -> dict:
    """Load a clinical case JSON file."""
    with open(path) as f:
        return json.load(f)


def build_disclosure_eval_prompt(
    case: dict,
    conversation_history: list[dict],
    turn_number: int,
    unevaluated_withheld: list[dict],
) -> list[dict]:
    """
    Build a prompt that asks the LLM ONLY to evaluate disclosure conditions
    for withheld facts that haven't been unlocked yet, and to provide tone notes.

    The LLM does NOT assemble the full brief — that happens in Python.
    """
    if conversation_history:
        conv_str = "\n".join(
            f"{'Student' if msg['role'] == 'student' else 'Patient'}: {msg['content']}"
            for msg in conversation_history
        )
    else:
        conv_str = "(No prior conversation — this is the first turn.)"

    student_message = ""
    if conversation_history:
        for msg in reversed(conversation_history):
            if msg["role"] == "student":
                student_message = msg["content"]
                break

    withheld_str = json.dumps(
        [
            {
                "id": f["id"],
                "content": f["content"],
                "disclosure_condition": f["disclosure_condition"],
            }
            for f in unevaluated_withheld
        ],
        indent=2,
    )

    system_prompt = f"""You are the disclosure evaluator for a simulated patient system.

## Patient
- Name: {case['patient']['name']}, Age: {case['patient']['age']}, Sex: {case['patient']['sex']}
- Chief Complaint: {case['chief_complaint']}

## Your Task

You will be given a list of WITHHELD facts, each with a disclosure_condition. Based on the conversation history and the student's current message, decide which facts should be UNLOCKED this turn.

A fact should be unlocked ONLY if its disclosure_condition is clearly and specifically met by what the student has said. Read the disclosure_condition literally — if it requires the student to ask about a specific topic (e.g., "heart disease"), a vague or general question (e.g., "tell me more" or "any medical conditions?") does NOT satisfy it. When in doubt, do NOT unlock. It is better to withhold a fact that should have been disclosed than to disclose one that should have been withheld.

Also provide tone_notes for how the patient should sound this turn.

## Withheld Facts to Evaluate
{withheld_str}

## Output Format

Respond with ONLY this JSON, no other text:

{{
  "unlock": [
    {{
      "fact_id": "<id of fact to unlock>",
      "trigger": "<what the student said that met the condition>"
    }}
  ],
  "tone_notes": {{
    "persona": "<one-line patient persona>",
    "emotional_state": "<how the patient feels THIS turn based on what just happened>",
    "speech_style": "<how the patient talks>"
  }}
}}

If NO facts should be unlocked, return an empty list for "unlock".
Output ONLY valid JSON. No explanation, no markdown, no commentary."""

    user_prompt = f"""## Conversation History
{conv_str}

## Current Turn: {turn_number}
Student's message: "{student_message}"

Evaluate which withheld facts (if any) should be unlocked."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_relevance_prompt(
    case: dict,
    student_message: str,
    authorized_fact_ids: list[str],
) -> list[dict]:
    """
    Ask the LLM to tag each authorized fact with relevance to the student's
    current question. This is a lightweight call — just classification.
    """
    fact_lookup = {f["id"]: f for f in case["facts"]}
    facts_for_tagging = [
        {"id": fid, "content": fact_lookup[fid]["content"]}
        for fid in authorized_fact_ids
    ]
    facts_str = json.dumps(facts_for_tagging, indent=2)

    system_prompt = """You are a relevance tagger. Given a student's question and a list of patient facts, tag each fact with its relevance to the student's current question.

Tags:
- "directly_relevant": the fact directly answers or relates to what the student is asking
- "tangentially_relevant": the fact is somewhat related but not what the student is asking about
- "not_relevant": the fact has nothing to do with the current question

Output ONLY a JSON object mapping fact_id to relevance tag:
{
  "C01": "directly_relevant",
  "C07": "not_relevant"
}

No other text."""

    user_prompt = f"""Student's question: "{student_message}"

Facts to tag:
{facts_str}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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
            f"Failed to parse LLM output as JSON: {e}\nRaw output:\n{text}"
        )


def add_hedging_rule(brief: dict) -> dict:
    """Add the fixed hedging rule to the brief. This is always the same."""
    brief["hedging_rule"] = {
        "instruction": (
            "If the student asks about a topic not listed in authorized_positives "
            "or authorized_negatives, the patient should hedge: express uncertainty, "
            "redirect, say 'I'm not sure', 'I haven't really thought about that', "
            "or 'I don't know'. The patient should NOT make up information and "
            "should NOT make definitive claims about unlisted topics."
        ),
        "examples": [
            "I'm not really sure about that.",
            "Hmm, I haven't really noticed.",
            "I don't think so, but I'm not certain.",
            "That's a good question, I'd have to think about it.",
        ],
    }
    return brief


class DisclosurePlanner:
    """
    Manages disclosure state across a conversation and produces
    content briefs each turn.

    Key design: The LLM only evaluates disclosure conditions and tags
    relevance. The brief is assembled deterministically in Python:
    - authorized_positives = all disclosed facts + all previously unlocked
      + newly unlocked this turn (excluding negative facts)
    - authorized_negatives = subset of authorized facts that are
      explicitly negative statements AND relevant to the current question
    - newly_unlocked = only facts unlocked THIS turn (enforced in code)
    """

    NEGATIVE_PREFIXES = ("No ", "Non-", "Never ", "no ", "non-", "never ")

    def __init__(self, case_path: str):
        self.case = load_case(case_path)
        self.conversation_history: list[dict] = []
        self.turn_number: int = 0
        self.unlocked_fact_ids: set[str] = set()

        # Pre-compute fact lookups
        self.fact_lookup: dict[str, dict] = {
            f["id"]: f for f in self.case["facts"]
        }
        self.disclosed_ids: set[str] = {
            f["id"] for f in self.case["facts"] if f["disclosed"]
        }
        self.withheld_ids: set[str] = {
            f["id"] for f in self.case["facts"] if not f["disclosed"]
        }

        # Identify negative facts (facts that deny something)
        self.negative_ids: set[str] = set()
        for f in self.case["facts"]:
            if f["content"].startswith(self.NEGATIVE_PREFIXES):
                self.negative_ids.add(f["id"])

    def process_turn(self, student_message: str) -> dict:
        """
        Process a student message and return a content brief.

        Steps:
        1. Ask LLM which withheld facts to unlock (if any)
        2. Assemble authorized facts deterministically
        3. Ask LLM to tag relevance of each authorized fact
        4. Build and return the complete brief
        """
        self.turn_number += 1
        self.conversation_history.append({
            "role": "student",
            "content": student_message,
        })

        # --- Step 1: Evaluate disclosure conditions ---
        unevaluated = [
            self.fact_lookup[fid]
            for fid in sorted(self.withheld_ids - self.unlocked_fact_ids)
        ]

        newly_unlocked_this_turn: list[dict] = []
        tone_notes = {
            "persona": "Cooperative patient",
            "emotional_state": "neutral",
            "speech_style": "casual",
        }

        if unevaluated:
            eval_messages = build_disclosure_eval_prompt(
                self.case,
                self.conversation_history,
                self.turn_number,
                unevaluated,
            )
            raw_eval = call_llm(eval_messages)
            eval_result = parse_json_response(raw_eval)

            # Process unlocks — LLM proposes, keyword gate confirms
            for unlock in eval_result.get("unlock", []):
                fid = unlock.get("fact_id", "")
                if fid in self.withheld_ids and fid not in self.unlocked_fact_ids:
                    fact = self.fact_lookup[fid]
                    if self._keyword_gate(fact):
                        self.unlocked_fact_ids.add(fid)
                        newly_unlocked_this_turn.append({
                            "fact_id": fid,
                            "content": fact["content"],
                            "trigger": unlock.get("trigger", ""),
                        })

            tone_notes = eval_result.get("tone_notes", tone_notes)

        # --- Step 2: Assemble authorized facts deterministically ---
        authorized_ids = sorted(self.disclosed_ids | self.unlocked_fact_ids)

        # --- Step 3: Tag relevance ---
        relevance_tags = {}
        try:
            rel_messages = build_relevance_prompt(
                self.case, student_message, authorized_ids
            )
            raw_rel = call_llm(rel_messages, temperature=0.1)
            relevance_tags = parse_json_response(raw_rel)
        except (ValueError, Exception):
            # If relevance tagging fails, default everything to not_relevant
            pass

        # --- Step 4: Build the brief ---
        authorized_positives = []
        for fid in authorized_ids:
            if fid not in self.negative_ids:
                authorized_positives.append({
                    "fact_id": fid,
                    "content": self.fact_lookup[fid]["content"],
                    "relevance": relevance_tags.get(fid, "not_relevant"),
                })

        # Authorized negatives: only negative facts that are authorized
        # AND relevant to the current question
        authorized_negatives = []
        for fid in authorized_ids:
            if fid in self.negative_ids:
                rel = relevance_tags.get(fid, "not_relevant")
                if rel in ("directly_relevant", "tangentially_relevant"):
                    authorized_negatives.append({
                        "fact_id": fid,
                        "content": self.fact_lookup[fid]["content"],
                    })

        brief = {
            "turn_number": self.turn_number,
            "student_message": student_message,
            "authorized_positives": authorized_positives,
            "authorized_negatives": authorized_negatives,
            "newly_unlocked": newly_unlocked_this_turn,
            "tone_notes": tone_notes,
        }

        brief = add_hedging_rule(brief)
        return brief

    def _keyword_gate(self, fact: dict) -> bool:
        """
        Deterministic check: at least one keyword from the fact's
        unlock_keywords must appear in the conversation history.

        This prevents the LLM from unlocking facts on vague questions
        like "tell me more" that don't actually mention the relevant topic.

        If the fact has no unlock_keywords, the LLM's judgment is trusted.
        """
        keywords = fact.get("unlock_keywords", [])
        if not keywords:
            # No keyword gate defined — trust the LLM
            return True

        # Search all student messages in conversation history
        student_text = " ".join(
            msg["content"].lower()
            for msg in self.conversation_history
            if msg["role"] == "student"
        )

        return any(kw.lower() in student_text for kw in keywords)

    def record_patient_response(self, patient_response: str):
        """Record the patient's response in conversation history."""
        self.conversation_history.append({
            "role": "patient",
            "content": patient_response,
        })

    def get_state(self) -> dict:
        """Return current planner state for debugging/logging."""
        return {
            "turn_number": self.turn_number,
            "unlocked_fact_ids": sorted(self.unlocked_fact_ids),
            "conversation_length": len(self.conversation_history),
        }


# --- CLI for testing ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python planner.py <case_file.json>")
        print("Then type student messages interactively.")
        sys.exit(1)

    case_path = sys.argv[1]
    planner = DisclosurePlanner(case_path)

    print(f"Loaded case: {planner.case['case_id']}")
    print(f"Patient: {planner.case['patient']['name']}")
    print(f"Chief complaint: {planner.case['chief_complaint']}")
    print(f"Total facts: {len(planner.case['facts'])}")
    print(f"Withheld: {len([f for f in planner.case['facts'] if not f['disclosed']])}")
    print(f"Negative facts: {sorted(planner.negative_ids)}")
    print("\nType student messages below. Press Ctrl+C to quit.\n")

    while True:
        try:
            student_msg = input("Student: ").strip()
            if not student_msg:
                continue

            brief = planner.process_turn(student_msg)
            print("\n--- Content Brief ---")
            print(json.dumps(brief, indent=2))
            print(f"\n--- Planner State ---")
            print(json.dumps(planner.get_state(), indent=2))
            print()

        except KeyboardInterrupt:
            print("\nDone.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
