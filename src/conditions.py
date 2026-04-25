"""
Experimental Conditions

Six conditions with a unified interface:
    1. naive_prompting — full case + basic withholding instructions
    2. structured_prompting — detailed rules, few-shot examples, chain-of-thought
    3. self_monitoring — generate then self-check and fix leaks
    4. isolated_architecture — the proposed system (planner + generator + verifier)
    5. no_isolation_ablation — same architecture but generator sees full case
    6. no_verifier_ablation — isolated architecture without the verifier

Each condition implements:
    - process_turn(student_message) -> patient_response
    - get_state() -> dict with turn count, unlocked facts, etc.
"""

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-fp16"


def call_llm(messages: list[dict], temperature: float = 0.7) -> str:
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
    with open(path) as f:
        return json.load(f)


def format_conversation(history: list[dict]) -> str:
    if not history:
        return "(No prior conversation.)"
    return "\n".join(
        f"{'Student' if m['role'] == 'student' else 'Patient'}: {m['content']}"
        for m in history
    )


# ═══════════════════════════════════════════════
# Condition 1: Naive Prompting
# Full case + basic "don't reveal these" instruction
# ═══════════════════════════════════════════════

class NaivePrompting:
    def __init__(self, case_path: str):
        self.case = load_case(case_path)
        self.conversation_history: list[dict] = []
        self.turn_number = 0

        self.withheld_facts = [f for f in self.case["facts"] if not f["disclosed"]]
        self.all_facts_str = json.dumps(
            [{"id": f["id"], "content": f["content"]} for f in self.case["facts"]],
            indent=2,
        )
        self.withheld_str = json.dumps(
            [{"id": f["id"], "content": f["content"]} for f in self.withheld_facts],
            indent=2,
        )

    def process_turn(self, student_message: str) -> str:
        self.turn_number += 1
        self.conversation_history.append({"role": "student", "content": student_message})

        system_prompt = f"""You are a simulated patient named {self.case['patient']['name']}, age {self.case['patient']['age']}, {self.case['patient']['sex']}.

Your chief complaint is: {self.case['chief_complaint']}

Here is your complete medical case:
{self.all_facts_str}

IMPORTANT: Do NOT reveal the following facts unless the student specifically asks about them:
{self.withheld_str}

If asked about a withheld fact, you may reveal it. If not asked, do not volunteer it.
Keep responses to 1-2 sentences. Respond naturally as a patient would."""

        conv_str = format_conversation(self.conversation_history)

        response = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation so far:\n{conv_str}\n\nRespond as the patient:"},
        ])

        response = response.strip().strip('"')
        self.conversation_history.append({"role": "patient", "content": response})
        return response

    def get_state(self) -> dict:
        return {"turn_number": self.turn_number, "condition": "naive_prompting"}


# ═══════════════════════════════════════════════
# Condition 2: Structured Prompting
# Detailed disclosure rules, few-shot examples, CoT
# ═══════════════════════════════════════════════

class StructuredPrompting:
    def __init__(self, case_path: str):
        self.case = load_case(case_path)
        self.conversation_history: list[dict] = []
        self.turn_number = 0

        self.disclosed_str = json.dumps(
            [{"id": f["id"], "content": f["content"]}
             for f in self.case["facts"] if f["disclosed"]],
            indent=2,
        )
        self.withheld_str = json.dumps(
            [{"id": f["id"], "content": f["content"],
              "disclosure_condition": f.get("disclosure_condition", "")}
             for f in self.case["facts"] if not f["disclosed"]],
            indent=2,
        )

    def process_turn(self, student_message: str) -> str:
        self.turn_number += 1
        self.conversation_history.append({"role": "student", "content": student_message})

        system_prompt = f"""You are a simulated patient named {self.case['patient']['name']}, age {self.case['patient']['age']}, {self.case['patient']['sex']}.
Chief complaint: {self.case['chief_complaint']}

## FREELY DISCLOSED FACTS (share these when relevant):
{self.disclosed_str}

## WITHHELD FACTS (share ONLY if the disclosure condition is met):
{self.withheld_str}

## DISCLOSURE RULES
1. Before responding, think step by step:
   a. What is the student asking about?
   b. Which disclosed facts are relevant to their question?
   c. For each withheld fact, does the student's question meet the disclosure condition?
   d. If no withheld fact's condition is met, use ONLY disclosed facts.
2. If the student's question does NOT match a disclosure condition, HEDGE: say "I'm not sure" or redirect.
3. NEVER volunteer withheld information the student hasn't asked about.
4. Keep responses to 1-2 sentences.

## EXAMPLES
Student: "What brings you in today?"
Reasoning: Student is asking about chief complaint. No withheld fact conditions are met.
Patient: "I've been having chest tightness for the past 3 days, feels like pressure."

Student: "Any sweating when the chest pain happens?"
Reasoning: Student asks about sweating — this matches the disclosure condition for diaphoresis. Reveal it.
Patient: "Yeah, I do get pretty sweaty during those episodes."

Student: "Tell me more about that."
Reasoning: Vague question. No specific disclosure condition is met. Use disclosed facts only.
Patient: "It comes and goes, usually lasts about 10-15 minutes each time."

Now respond to the student's latest message. Think through the disclosure rules, then respond as the patient ONLY (do not show your reasoning)."""

        conv_str = format_conversation(self.conversation_history)

        response = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation:\n{conv_str}\n\nRespond as the patient:"},
        ])

        response = response.strip().strip('"')
        # Strip any reasoning the model might have included
        if "Patient:" in response:
            response = response.split("Patient:")[-1].strip()
        if "Reasoning:" in response:
            response = response.split("Reasoning:")[0].strip()

        self.conversation_history.append({"role": "patient", "content": response})
        return response

    def get_state(self) -> dict:
        return {"turn_number": self.turn_number, "condition": "structured_prompting"}


# ═══════════════════════════════════════════════
# Condition 3: Self-Monitoring
# Generate then check for leaks and fix
# ═══════════════════════════════════════════════

class SelfMonitoring:
    def __init__(self, case_path: str):
        self.case = load_case(case_path)
        self.conversation_history: list[dict] = []
        self.turn_number = 0

        self.all_facts_str = json.dumps(
            [{"id": f["id"], "content": f["content"]} for f in self.case["facts"]],
            indent=2,
        )
        self.withheld_str = json.dumps(
            [{"id": f["id"], "content": f["content"],
              "disclosure_condition": f.get("disclosure_condition", "")}
             for f in self.case["facts"] if not f["disclosed"]],
            indent=2,
        )

    def process_turn(self, student_message: str) -> str:
        self.turn_number += 1
        self.conversation_history.append({"role": "student", "content": student_message})

        conv_str = format_conversation(self.conversation_history)

        # Step 1: Generate
        gen_prompt = f"""You are a simulated patient named {self.case['patient']['name']}, age {self.case['patient']['age']}, {self.case['patient']['sex']}.
Chief complaint: {self.case['chief_complaint']}

Your complete case:
{self.all_facts_str}

WITHHELD facts (do NOT reveal unless the student specifically asks):
{self.withheld_str}

Keep responses to 1-2 sentences. Respond naturally.

Conversation:
{conv_str}

Respond as the patient:"""

        draft = call_llm([{"role": "user", "content": gen_prompt}])
        draft = draft.strip().strip('"')

        # Step 2: Self-check
        check_prompt = f"""You are a leak checker. A simulated patient generated a response. Check if it reveals any WITHHELD information that shouldn't have been disclosed.

WITHHELD facts:
{self.withheld_str}

Student's message: "{student_message}"

Patient's draft response: "{draft}"

Does the draft reveal any withheld fact whose disclosure condition was NOT met by the student's question? If yes, rewrite the response to remove the leak while keeping it natural. If no leaks, return the response unchanged.

Respond with ONLY the final patient response (no explanation):"""

        fixed = call_llm([{"role": "user", "content": check_prompt}], temperature=0.3)
        fixed = fixed.strip().strip('"')

        self.conversation_history.append({"role": "patient", "content": fixed})
        return fixed

    def get_state(self) -> dict:
        return {"turn_number": self.turn_number, "condition": "self_monitoring"}


# ═══════════════════════════════════════════════
# Condition 4: Isolated Architecture (main system)
# Imported from existing modules
# ═══════════════════════════════════════════════

class IsolatedArchitecture:
    def __init__(self, case_path: str):
        from src.planner import DisclosurePlanner
        from src.generator import generate_response as _gen
        from src.verifier import Verifier

        self.planner = DisclosurePlanner(case_path)
        self.verifier = Verifier(case_path)
        self._generate = _gen
        self.case = self.planner.case
        self.max_retries = 2
        self.total_retries = 0

    def process_turn(self, student_message: str) -> str:
        brief = self.planner.process_turn(student_message)

        # Generate + verify loop
        for attempt in range(self.max_retries + 1):
            response = self._generate(
                brief, self.planner.conversation_history[:-1]
            )
            ver = self.verifier.verify(
                response=response,
                unlocked_fact_ids=self.planner.unlocked_fact_ids,
                student_message=student_message,
            )
            if ver["pass"]:
                break
            self.total_retries += 1

        self.planner.record_patient_response(response)
        return response

    def get_state(self) -> dict:
        state = self.planner.get_state()
        state["condition"] = "isolated_architecture"
        state["total_retries"] = self.total_retries
        return state


# ═══════════════════════════════════════════════
# Condition 5: No-Isolation Ablation
# Same planner/generator/verifier, but generator sees full case
# ═══════════════════════════════════════════════

class NoIsolationAblation:
    def __init__(self, case_path: str):
        from src.planner import DisclosurePlanner
        from src.verifier import Verifier

        self.case_path = case_path
        self.planner = DisclosurePlanner(case_path)
        self.verifier = Verifier(case_path)
        self.case = self.planner.case
        self.max_retries = 2
        self.total_retries = 0

        # Full case for the generator (breaks isolation)
        self.all_facts_str = json.dumps(
            [{"id": f["id"], "content": f["content"]} for f in self.case["facts"]],
            indent=2,
        )
        self.withheld_str = json.dumps(
            [{"id": f["id"], "content": f["content"]}
             for f in self.case["facts"] if not f["disclosed"]],
            indent=2,
        )

    def _generate_with_full_case(self, brief: dict, conversation_history: list[dict]) -> str:
        """Generator that sees the full case — isolation is broken."""
        conv_str = format_conversation(conversation_history)
        tone = brief.get("tone_notes", {})

        # Generator sees everything — full case + brief
        prompt = f"""You are a simulated patient named {self.case['patient']['name']}.

## Your Persona
- {tone.get('persona', 'Cooperative patient')}
- Emotional state: {tone.get('emotional_state', 'neutral')}
- Speech style: {tone.get('speech_style', 'casual')}

## YOUR COMPLETE CASE (you know all of this):
{self.all_facts_str}

## WITHHELD facts — do NOT reveal these unless the content brief authorizes them:
{self.withheld_str}

## Content Brief (from the planner — follow this):
Authorized positives: {json.dumps([f['content'] for f in brief['authorized_positives']])}
Newly unlocked: {json.dumps([f['content'] for f in brief['newly_unlocked']])}

## Rules
- Only share facts listed in the authorized positives or newly unlocked.
- Do NOT reveal withheld facts not in the brief.
- Keep responses to 1-2 sentences.
- Use exact details from the facts.

Conversation:
{conv_str}

Student: "{brief['student_message']}"
Respond as the patient:"""

        response = call_llm([{"role": "user", "content": prompt}])
        return response.strip().strip('"')

    def process_turn(self, student_message: str) -> str:
        brief = self.planner.process_turn(student_message)

        for attempt in range(self.max_retries + 1):
            response = self._generate_with_full_case(
                brief, self.planner.conversation_history[:-1]
            )
            ver = self.verifier.verify(
                response=response,
                unlocked_fact_ids=self.planner.unlocked_fact_ids,
                student_message=student_message,
            )
            if ver["pass"]:
                break
            self.total_retries += 1

        self.planner.record_patient_response(response)
        return response

    def get_state(self) -> dict:
        state = self.planner.get_state()
        state["condition"] = "no_isolation_ablation"
        state["total_retries"] = self.total_retries
        return state


# ═══════════════════════════════════════════════
# Condition 6: No-Verifier Ablation
# Full isolated architecture without the verifier
# ═══════════════════════════════════════════════

class NoVerifierAblation:
    def __init__(self, case_path: str):
        from src.planner import DisclosurePlanner
        from src.generator import generate_response as _gen

        self.planner = DisclosurePlanner(case_path)
        self._generate = _gen
        self.case = self.planner.case

    def process_turn(self, student_message: str) -> str:
        brief = self.planner.process_turn(student_message)
        response = self._generate(
            brief, self.planner.conversation_history[:-1]
        )
        self.planner.record_patient_response(response)
        return response

    def get_state(self) -> dict:
        state = self.planner.get_state()
        state["condition"] = "no_verifier_ablation"
        return state


# ═══════════════════════════════════════════════
# Condition registry
# ═══════════════════════════════════════════════

CONDITIONS = {
    "naive_prompting": NaivePrompting,
    "structured_prompting": StructuredPrompting,
    "self_monitoring": SelfMonitoring,
    "isolated_architecture": IsolatedArchitecture,
    "no_isolation_ablation": NoIsolationAblation,
    "no_verifier_ablation": NoVerifierAblation,
}


def get_condition(name: str, case_path: str):
    """Instantiate a condition by name."""
    if name not in CONDITIONS:
        raise ValueError(
            f"Unknown condition '{name}'. "
            f"Available: {list(CONDITIONS.keys())}"
        )
    return CONDITIONS[name](case_path)


def get_condition_names() -> list[str]:
    return list(CONDITIONS.keys())


# --- CLI ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print("Available conditions:")
        for name in get_condition_names():
            print(f"  {name}")
        sys.exit(0)

    print(f"Condition '{sys.argv[1]}' is available.")
