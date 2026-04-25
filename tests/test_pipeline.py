"""
End-to-End Pipeline Test: Student Agent + Planner + Generator + Verifier

Runs a full 20-turn conversation using a selected pressure strategy.

Usage:
    python3 tests/test_pipeline.py                      # defaults to direct_questioning
    python3 tests/test_pipeline.py emotional_appeal      # specific strategy
    python3 tests/test_pipeline.py --list                # list all strategies
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.planner import DisclosurePlanner
from src.generator import generate_response
from src.verifier import Verifier
from src.student_agent import get_strategy, get_strategy_names


CASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cases",
    "case_cardiology.json",
)

MAX_RETRIES = 2

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def run_conversation(strategy_name: str, max_turns: int = 20):
    """Run a full conversation using the specified strategy."""

    planner = DisclosurePlanner(CASE_PATH)
    verifier = Verifier(CASE_PATH)
    student_messages = get_strategy(strategy_name, "cardiology")

    turns = min(max_turns, len(student_messages))

    print(f"\n{BOLD}═══ Full Pipeline Test ═══{RESET}")
    print(f"Case: {planner.case['case_id']} — {planner.case['patient']['name']}")
    print(f"Strategy: {strategy_name}")
    print(f"Turns: {turns}")
    print(f"Max retries per turn: {MAX_RETRIES}\n")

    total_retries = 0
    total_time = 0

    for i in range(turns):
        student_msg = student_messages[i]
        turn_num = i + 1

        print(f"{BOLD}--- Turn {turn_num} ---{RESET}")
        print(f"{BLUE}Student:{RESET} {student_msg}")

        turn_start = time.time()

        # Planner
        brief = planner.process_turn(student_msg)

        if brief["newly_unlocked"]:
            unlocked = [f["fact_id"] for f in brief["newly_unlocked"]]
            print(f"{YELLOW}  [Unlocked: {unlocked}]{RESET}")

        # Generate + Verify loop
        accepted = False
        attempt = 0

        while not accepted and attempt <= MAX_RETRIES:
            attempt += 1
            patient_response = generate_response(
                brief, planner.conversation_history[:-1]
            )
            ver_result = verifier.verify(
                response=patient_response,
                unlocked_fact_ids=planner.unlocked_fact_ids,
                student_message=student_msg,
            )

            if ver_result["pass"]:
                accepted = True
                print(f"{GREEN}Patient:{RESET} {patient_response}")
            else:
                total_retries += 1
                issues = []
                for leak in ver_result["keyword_leaks"]:
                    issues.append(f"leak: '{leak['keyword']}' from {leak['fact_id']}")
                print(f"{RED}  [REJECTED attempt {attempt}]{RESET} {'; '.join(issues)}")

                if attempt > MAX_RETRIES:
                    print(f"{GREEN}Patient:{RESET} {patient_response}")

        planner.record_patient_response(patient_response)

        turn_time = time.time() - turn_start
        total_time += turn_time
        print()

    # Summary
    print(f"{BOLD}═══ Summary ═══{RESET}")
    state = planner.get_state()
    print(f"Strategy: {strategy_name}")
    print(f"Turns: {state['turn_number']}")
    print(f"Unlocked: {state['unlocked_fact_ids']}")
    print(f"Regenerations: {total_retries}")
    print(f"Total time: {total_time:.0f}s ({total_time/turns:.1f}s per turn)")

    withheld_remaining = [
        f["id"] for f in planner.case["facts"]
        if not f["disclosed"] and f["id"] not in state["unlocked_fact_ids"]
    ]
    print(f"Still withheld: {withheld_remaining}")

    # Leak check
    patient_text = " ".join(
        msg["content"] for msg in planner.conversation_history
        if msg["role"] == "patient"
    ).lower()

    with open(CASE_PATH) as f:
        case = json.load(f)
    leak_found = False
    for fact in case["facts"]:
        if fact["disclosed"] or fact["id"] in state["unlocked_fact_ids"]:
            continue
        for phrase in fact.get("leak_phrases", []):
            if phrase.lower() in patient_text:
                print(f"{RED}⚠ LEAK: '{phrase}' from {fact['id']}{RESET}")
                leak_found = True
    if not leak_found:
        print(f"{GREEN}✓ No leaks detected{RESET}")


if __name__ == "__main__":
    # Handle args
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print("Available strategies:")
        for name in get_strategy_names():
            print(f"  {name}")
        sys.exit(0)

    strategy = sys.argv[1] if len(sys.argv) > 1 else "direct_questioning"

    # Check Ollama
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    run_conversation(strategy)
