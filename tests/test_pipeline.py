"""
End-to-End Pipeline Test: Planner + Generator

Runs a short scripted conversation through the planner and generator
to verify they work together. Prints the full exchange so you can
judge response quality.

Run with: python tests/test_pipeline.py
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.planner import DisclosurePlanner
from src.generator import generate_response


CASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cases",
    "case_cardiology.json",
)

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def run_conversation():
    """Run a scripted 5-turn conversation and print everything."""

    planner = DisclosurePlanner(CASE_PATH)

    # Scripted student messages that test different scenarios
    student_messages = [
        # Turn 1: Opening — should get chief complaint, no unlocks
        "Hi there, I'm a medical student. What brings you in today?",

        # Turn 2: Follow-up on symptoms — should unlock C03 (sweating)
        "I see, chest tightness for three days. When the tightness happens, "
        "do you notice any other symptoms? Any sweating or dizziness?",

        # Turn 3: Vague follow-up — should NOT unlock anything
        "Okay, tell me more about that.",

        # Turn 4: Ask about radiation — should unlock C05 (arm tingling)
        "Does the pain spread anywhere else? Like to your arms or shoulders?",

        # Turn 5: Ask about family heart history — should unlock C18
        "Has anyone in your family had heart disease or heart attacks?",
    ]

    print(f"\n{BOLD}═══ End-to-End Pipeline Test ═══{RESET}")
    print(f"Case: {planner.case['case_id']} — {planner.case['patient']['name']}")
    print(f"Chief complaint: {planner.case['chief_complaint']}\n")

    for i, student_msg in enumerate(student_messages, 1):
        print(f"{BOLD}--- Turn {i} ---{RESET}")
        print(f"{BLUE}Student:{RESET} {student_msg}")

        start = time.time()

        # Step 1: Planner produces content brief
        brief = planner.process_turn(student_msg)

        planner_time = time.time() - start

        # Show what was unlocked
        if brief["newly_unlocked"]:
            unlocked = [f["fact_id"] for f in brief["newly_unlocked"]]
            print(f"{YELLOW}  [Unlocked: {unlocked}]{RESET}")
        else:
            print(f"{YELLOW}  [No new unlocks]{RESET}")

        # Step 2: Generator produces patient response
        gen_start = time.time()
        patient_response = generate_response(brief, planner.conversation_history[:-1])
        gen_time = time.time() - gen_start

        print(f"{GREEN}Patient:{RESET} {patient_response}")
        print(f"  (planner: {planner_time:.1f}s, generator: {gen_time:.1f}s)")

        # Record patient response in planner's history
        planner.record_patient_response(patient_response)
        print()

    # Final state
    print(f"{BOLD}═══ Final State ═══{RESET}")
    state = planner.get_state()
    print(f"Turns: {state['turn_number']}")
    print(f"Unlocked facts: {state['unlocked_fact_ids']}")
    print(f"Conversation length: {state['conversation_length']} messages")

    # Check for obvious problems
    print(f"\n{BOLD}═══ Quick Checks ═══{RESET}")

    # Load full case to check for leaks in patient responses
    with open(CASE_PATH) as f:
        case = json.load(f)

    withheld_not_unlocked = {
        f["id"]: f["content"]
        for f in case["facts"]
        if not f["disclosed"] and f["id"] not in state["unlocked_fact_ids"]
    }

    print(f"Withheld facts that were NOT unlocked: {list(withheld_not_unlocked.keys())}")

    # Crude leak check: look for withheld fact content in patient responses
    patient_responses = " ".join(
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
                if kw in patient_responses:
                    leaks_found.append(f"{fid} (keyword: '{kw}')")

    if leaks_found:
        print(f"⚠ POSSIBLE LEAKS DETECTED: {leaks_found}")
    else:
        print(f"✓ No obvious leaks of withheld facts in patient responses")


if __name__ == "__main__":
    # Check Ollama
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    run_conversation()
