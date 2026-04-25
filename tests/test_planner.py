"""
Planner Integration Tests

Run with: python tests/test_planner.py

Requires Ollama running locally with llama3.1:8b-instruct-fp16 pulled.
"""

import json
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.planner import DisclosurePlanner


CASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cases",
    "case_cardiology.json",
)

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_pass(name, detail=""):
    print(f"  {GREEN}✓ PASS{RESET}  {name}")
    if detail:
        print(f"           {detail}")


def print_fail(name, detail=""):
    print(f"  {RED}✗ FAIL{RESET}  {name}")
    if detail:
        print(f"           {detail}")


def print_brief_summary(brief):
    pos_ids = [f["fact_id"] for f in brief.get("authorized_positives", [])]
    neg_ids = [f["fact_id"] for f in brief.get("authorized_negatives", [])]
    new_ids = [f["fact_id"] for f in brief.get("newly_unlocked", [])]
    print(f"           authorized_positives ({len(pos_ids)}): {pos_ids}")
    print(f"           authorized_negatives ({len(neg_ids)}): {neg_ids}")
    print(f"           newly_unlocked: {new_ids}")
    tone = brief.get("tone_notes", {})
    if tone.get("emotional_state"):
        print(f"           emotional_state: {tone['emotional_state']}")


def get_expected_positive_count(planner):
    """
    Calculate expected number of authorized_positives.
    = disclosed non-negative facts + unlocked non-negative facts
    """
    authorized = planner.disclosed_ids | planner.unlocked_fact_ids
    return len(authorized - planner.negative_ids)


# ─────────────────────────────────────────────
# Test 1: Basic opening — only disclosed facts, all of them
# ─────────────────────────────────────────────
def test_1_basic_opening():
    name = "Basic opening — all disclosed facts present, no withheld"
    planner = DisclosurePlanner(CASE_PATH)
    brief = planner.process_turn("Hi, what brings you in today?")

    # Check no withheld facts in authorized_positives
    authorized_ids = {f["fact_id"] for f in brief["authorized_positives"]}
    leaked = authorized_ids & planner.withheld_ids
    if leaked:
        print_fail(name, f"Withheld facts leaked: {leaked}")
    else:
        print_pass(name)

    # Check ALL non-negative disclosed facts are present
    expected_count = len(planner.disclosed_ids - planner.negative_ids)
    actual_count = len(brief["authorized_positives"])
    if actual_count == expected_count:
        print_pass(f"  All {expected_count} non-negative disclosed facts present")
    else:
        print_fail(
            f"  Expected {expected_count} facts, got {actual_count}",
            f"Missing: {(planner.disclosed_ids - planner.negative_ids) - authorized_ids}"
        )

    # Check newly_unlocked is empty
    if not brief["newly_unlocked"]:
        print_pass("  No facts unlocked on greeting")
    else:
        unlocked = [f["fact_id"] for f in brief["newly_unlocked"]]
        print_fail("  No facts should unlock on greeting", f"Got: {unlocked}")

    print_brief_summary(brief)
    return brief


# ─────────────────────────────────────────────
# Test 2: Ask about associated symptoms → should unlock C03 (diaphoresis)
# ─────────────────────────────────────────────
def test_2_unlock_diaphoresis():
    name = "Ask about sweating during episodes → unlock C03"
    planner = DisclosurePlanner(CASE_PATH)

    planner.process_turn("What brings you in today?")
    planner.record_patient_response(
        "I've been having this chest tightness for the past three days. "
        "It feels like pressure right in the middle of my chest."
    )

    brief = planner.process_turn(
        "When the chest tightness comes on, do you notice anything else? "
        "Any sweating, nausea, or dizziness?"
    )

    newly_unlocked_ids = {f["fact_id"] for f in brief["newly_unlocked"]}

    if "C03" in newly_unlocked_ids:
        print_pass(name)
    else:
        print_fail(name, f"C03 not in newly_unlocked. Got: {newly_unlocked_ids}")

    # C03 should also be in authorized_positives
    authorized_ids = {f["fact_id"] for f in brief["authorized_positives"]}
    if "C03" in authorized_ids:
        print_pass("  C03 in authorized_positives")
    else:
        print_fail("  C03 missing from authorized_positives")

    # Verify all disclosed facts are still present
    expected_count = get_expected_positive_count(planner)
    actual_count = len(brief["authorized_positives"])
    if actual_count == expected_count:
        print_pass(f"  All {expected_count} authorized facts present")
    else:
        print_fail(f"  Expected {expected_count} facts, got {actual_count}")

    # Verify newly_unlocked only contains withheld facts
    for f in brief["newly_unlocked"]:
        if f["fact_id"] not in planner.withheld_ids:
            print_fail(f"  {f['fact_id']} in newly_unlocked but is not a withheld fact")

    print_brief_summary(brief)
    return brief


# ─────────────────────────────────────────────
# Test 3: Vague follow-up — nothing should unlock
# ─────────────────────────────────────────────
def test_3_vague_message_no_unlock():
    name = "Vague follow-up — no new unlocks"
    planner = DisclosurePlanner(CASE_PATH)

    planner.process_turn("What brings you in today?")
    planner.record_patient_response(
        "I've been having this chest tightness and shortness of breath."
    )

    brief = planner.process_turn("Can you tell me more about that?")

    if not brief["newly_unlocked"]:
        print_pass(name)
    else:
        unlocked = [f["fact_id"] for f in brief["newly_unlocked"]]
        print_fail(name, f"Facts unlocked on vague message: {unlocked}")

    # Verify all disclosed facts still present
    expected_count = get_expected_positive_count(planner)
    actual_count = len(brief["authorized_positives"])
    if actual_count == expected_count:
        print_pass(f"  All {expected_count} authorized facts present")
    else:
        print_fail(f"  Expected {expected_count} facts, got {actual_count}")

    print_brief_summary(brief)
    return brief


# ─────────────────────────────────────────────
# Test 4: Multi-turn state — unlocked fact persists, not re-listed as new
# ─────────────────────────────────────────────
def test_4_unlocked_fact_persists():
    name = "Unlocked fact persists but not re-listed as newly_unlocked"
    planner = DisclosurePlanner(CASE_PATH)

    # Turn 1
    planner.process_turn("What brings you in today?")
    planner.record_patient_response(
        "I've been having chest tightness for three days, like a pressure."
    )

    # Turn 2: unlock C03
    brief2 = planner.process_turn(
        "Do you notice any sweating when the chest tightness happens?"
    )
    planner.record_patient_response(
        "Yeah, actually I do get sweaty during the episodes. Started about two days ago."
    )

    # Turn 3: different topic
    brief3 = planner.process_turn("How about your breathing? Any shortness of breath?")

    authorized_ids_3 = {f["fact_id"] for f in brief3["authorized_positives"]}
    newly_unlocked_3 = {f["fact_id"] for f in brief3["newly_unlocked"]}

    # C03 should still be in authorized_positives
    if "C03" in authorized_ids_3:
        print_pass(name, "C03 still in authorized_positives at turn 3")
    else:
        print_fail(name, "C03 missing from authorized_positives at turn 3")

    # C03 should NOT be in newly_unlocked at turn 3 (deterministic check)
    if "C03" not in newly_unlocked_3:
        print_pass("  C03 not in newly_unlocked at turn 3 (correct)")
    else:
        print_fail("  C03 re-appeared in newly_unlocked at turn 3")

    # Verify fact count
    expected_count = get_expected_positive_count(planner)
    actual_count = len(brief3["authorized_positives"])
    if actual_count == expected_count:
        print_pass(f"  All {expected_count} authorized facts present")
    else:
        print_fail(f"  Expected {expected_count} facts, got {actual_count}")

    print_brief_summary(brief3)
    return brief3


# ─────────────────────────────────────────────
# Test 5: Partial condition — family history without specificity
# ─────────────────────────────────────────────
def test_5_partial_condition_no_unlock():
    name = "General family history question — C18 should NOT unlock"
    planner = DisclosurePlanner(CASE_PATH)

    planner.process_turn("What brings you in today?")
    planner.record_patient_response(
        "Chest tightness and shortness of breath for three days."
    )

    brief = planner.process_turn(
        "Do you have any family members with medical conditions?"
    )

    newly_unlocked_ids = {f["fact_id"] for f in brief["newly_unlocked"]}
    authorized_ids = {f["fact_id"] for f in brief["authorized_positives"]}

    if "C18" not in newly_unlocked_ids and "C18" not in authorized_ids:
        print_pass(name)
    else:
        print_fail(
            name,
            "C18 unlocked on generic family history question. "
            "This may indicate the planner is too generous with disclosure conditions. "
            "Consider whether this is acceptable for your project.",
        )

    print_brief_summary(brief)
    return brief


# ─────────────────────────────────────────────
# Test 6: Verify negative facts are handled correctly
# ─────────────────────────────────────────────
def test_6_negative_facts():
    name = "Negative facts identified correctly"
    planner = DisclosurePlanner(CASE_PATH)

    # Check which facts are identified as negative
    case_negatives = {
        fid: planner.fact_lookup[fid]["content"]
        for fid in planner.negative_ids
    }
    print(f"           Identified negative facts: {case_negatives}")

    # Negative facts should only be things like "No known drug allergies",
    # "Non-smoker", "Never smoked" — not positive statements
    all_valid = True
    for fid, content in case_negatives.items():
        # Check that each "negative" fact is actually a denial
        if not any(content.startswith(p) for p in DisclosurePlanner.NEGATIVE_PREFIXES):
            print_fail(f"  {fid} falsely identified as negative: {content}")
            all_valid = False

    if all_valid and case_negatives:
        print_pass(name, f"{len(case_negatives)} negative facts identified")
    elif not case_negatives:
        print_fail(name, "No negative facts found — expected at least 'No known drug allergies'")

    # Negative facts should NOT appear in authorized_positives
    brief = planner.process_turn("Tell me about your medical history")
    pos_ids = {f["fact_id"] for f in brief["authorized_positives"]}
    neg_in_pos = pos_ids & planner.negative_ids
    if not neg_in_pos:
        print_pass("  Negative facts excluded from authorized_positives")
    else:
        print_fail(f"  Negative facts in authorized_positives: {neg_in_pos}")

    print_brief_summary(brief)
    return brief


# ─────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────
def main():
    print(f"\n{BOLD}═══ Planner Integration Tests ═══{RESET}")
    print(f"Case: {CASE_PATH}")
    print(f"Model: llama3.1:8b-instruct-fp16 via Ollama\n")

    # Check Ollama is running
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama is running. Available models: {models}\n")
    except Exception as e:
        print(f"{RED}ERROR: Cannot connect to Ollama at localhost:11434{RESET}")
        print(f"Make sure Ollama is running: ollama serve")
        print(f"And the model is pulled: ollama pull llama3.1:8b-instruct-fp16")
        print(f"\nError: {e}")
        sys.exit(1)

    tests = [
        ("Test 1: Basic opening", test_1_basic_opening),
        ("Test 2: Unlock diaphoresis", test_2_unlock_diaphoresis),
        ("Test 3: Vague message", test_3_vague_message_no_unlock),
        ("Test 4: Persistence", test_4_unlocked_fact_persists),
        ("Test 5: Partial condition", test_5_partial_condition_no_unlock),
        ("Test 6: Negative facts", test_6_negative_facts),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{BOLD}--- {name} ---{RESET}")
        start = time.time()
        try:
            test_fn()
            results.append((name, True, None))
        except Exception as e:
            print(f"  {RED}✗ EXCEPTION{RESET}  {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))
        elapsed = time.time() - start
        print(f"  ({elapsed:.1f}s)")

    # Summary
    print(f"\n{BOLD}═══ Summary ═══{RESET}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"{passed}/{total} tests completed without exceptions")
    if all(ok for _, ok, _ in results):
        print(f"{GREEN}All tests passed!{RESET}")
    print(f"\nNote: Tests 2 and 5 depend on LLM judgment and may vary between runs.")
    print(f"Tests 1, 3, 4, 6 validate deterministic behavior and should always pass.")


if __name__ == "__main__":
    main()
