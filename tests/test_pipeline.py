"""
End-to-End Pipeline Test: Planner + Generator + Verifier

Runs a short scripted conversation through the full architecture.
The verifier checks each response; failures trigger regeneration.

Run with: python3 tests/test_pipeline.py
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.planner import DisclosurePlanner
from src.generator import generate_response
from src.verifier import Verifier


CASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cases",
    "case_cardiology.json",
)

MAX_RETRIES = 2  # Max regeneration attempts per turn

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def run_conversation():
    """Run a scripted 5-turn conversation through the full pipeline."""

    planner = DisclosurePlanner(CASE_PATH)
    verifier = Verifier(CASE_PATH)

    student_messages = [
        # Turn 1: Opening
        "Hi there, I'm a medical student. What brings you in today?",

        # Turn 2: Should unlock C03 (sweating)
        "I see, chest tightness for three days. When the tightness happens, "
        "do you notice any other symptoms? Any sweating or dizziness?",

        # Turn 3: Vague — should NOT unlock anything
        "Okay, tell me more about that.",

        # Turn 4: Should unlock C05 (arm tingling)
        "Does the pain spread anywhere else? Like to your arms or shoulders?",

        # Turn 5: Should unlock C18 (father's heart attack)
        "Has anyone in your family had heart disease or heart attacks?",
    ]

    print(f"\n{BOLD}═══ Full Pipeline Test: Planner + Generator + Verifier ═══{RESET}")
    print(f"Case: {planner.case['case_id']} — {planner.case['patient']['name']}")
    print(f"Chief complaint: {planner.case['chief_complaint']}")
    print(f"Max retries per turn: {MAX_RETRIES}\n")

    total_retries = 0
    all_issues = []

    for i, student_msg in enumerate(student_messages, 1):
        print(f"{BOLD}--- Turn {i} ---{RESET}")
        print(f"{BLUE}Student:{RESET} {student_msg}")

        # Step 1: Planner
        start = time.time()
        brief = planner.process_turn(student_msg)
        planner_time = time.time() - start

        if brief["newly_unlocked"]:
            unlocked = [f["fact_id"] for f in brief["newly_unlocked"]]
            print(f"{YELLOW}  [Unlocked: {unlocked}]{RESET}")
        else:
            print(f"{YELLOW}  [No new unlocks]{RESET}")

        # Step 2: Generate + Verify loop
        accepted = False
        attempt = 0

        while not accepted and attempt <= MAX_RETRIES:
            attempt += 1

            # Generate
            gen_start = time.time()
            patient_response = generate_response(
                brief, planner.conversation_history[:-1]
            )
            gen_time = time.time() - gen_start

            # Verify
            ver_start = time.time()
            ver_result = verifier.verify(
                response=patient_response,
                unlocked_fact_ids=planner.unlocked_fact_ids,
                student_message=student_msg,
            )
            ver_time = time.time() - ver_start

            if ver_result["pass"]:
                accepted = True
                print(f"{GREEN}Patient:{RESET} {patient_response}")
                print(
                    f"  (planner: {planner_time:.1f}s, "
                    f"generator: {gen_time:.1f}s, "
                    f"verifier: {ver_time:.1f}s"
                    f"{f', attempt {attempt}' if attempt > 1 else ''})"
                )
            else:
                total_retries += 1
                issues = []
                if ver_result["keyword_leaks"]:
                    for leak in ver_result["keyword_leaks"]:
                        issues.append(f"keyword leak: '{leak['keyword']}' from {leak['fact_id']}")
                if ver_result["llm_check"].get("issues"):
                    for issue in ver_result["llm_check"]["issues"]:
                        issues.append(f"{issue['type']}: {issue['description']}")

                all_issues.append({
                    "turn": i,
                    "attempt": attempt,
                    "response": patient_response,
                    "issues": issues,
                })

                print(f"{RED}  [REJECTED attempt {attempt}]{RESET} {patient_response}")
                for iss in issues:
                    print(f"{RED}    → {iss}{RESET}")

                if attempt > MAX_RETRIES:
                    # Use last response despite issues
                    print(f"{RED}  [MAX RETRIES — using last response]{RESET}")
                    print(f"{GREEN}Patient:{RESET} {patient_response}")

        # Record accepted response
        planner.record_patient_response(patient_response)
        print()

    # Summary
    print(f"{BOLD}═══ Final State ═══{RESET}")
    state = planner.get_state()
    print(f"Turns: {state['turn_number']}")
    print(f"Unlocked facts: {state['unlocked_fact_ids']}")
    print(f"Total regenerations: {total_retries}")

    if all_issues:
        print(f"\n{BOLD}═══ Rejected Responses ═══{RESET}")
        for issue in all_issues:
            print(f"  Turn {issue['turn']}, attempt {issue['attempt']}:")
            print(f"    Response: {issue['response'][:80]}...")
            for iss in issue["issues"]:
                print(f"    → {iss}")
    else:
        print(f"{GREEN}No responses were rejected by the verifier.{RESET}")

    # Leak check
    print(f"\n{BOLD}═══ Final Leak Check ═══{RESET}")
    withheld_not_unlocked = [
        f["id"] for f in planner.case["facts"]
        if not f["disclosed"] and f["id"] not in state["unlocked_fact_ids"]
    ]
    print(f"Still withheld: {withheld_not_unlocked}")

    patient_text = " ".join(
        msg["content"] for msg in planner.conversation_history
        if msg["role"] == "patient"
    ).lower()

    leak_keywords = {
        "C04": ["nausea"],
        "C06": ["ankle", "swelling", "edema"],
        "C09": ["cholesterol", "statin", "atorvastatin", "lipid"],
        "C13": ["sedentary", "exercise"],
        "C14": ["fast food", "diet"],
    }

    leaks_found = []
    for fid, keywords in leak_keywords.items():
        if fid in withheld_not_unlocked:
            for kw in keywords:
                if kw in patient_text:
                    leaks_found.append(f"{fid} (keyword: '{kw}')")

    if leaks_found:
        print(f"{RED}⚠ POSSIBLE LEAKS: {leaks_found}{RESET}")
    else:
        print(f"{GREEN}✓ No leaks of withheld facts detected{RESET}")


if __name__ == "__main__":
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    run_conversation()
