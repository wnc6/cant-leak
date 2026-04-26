"""
Smoke Test + Meaningfulness Check

Runs all 6 conditions with the authority_claim strategy for 10 turns.
Compares leakage, disclosure behavior, and response quality across conditions.

Usage: python3 tests/smoke_test.py
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiment import run_experiment

CASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cases",
    "case_cardiology.json",
)

CONDITIONS = [
    "naive_prompting",
    "structured_prompting",
    "self_monitoring",
    "isolated_architecture",
    "no_isolation_ablation",
    "no_verifier_ablation",
]

STRATEGY = "authority_claim"
TURNS = 10

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():
    # Check Ollama
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    print(f"\n{BOLD}═══ Smoke Test: All 6 Conditions × authority_claim × 10 turns ═══{RESET}\n")

    # Load case for analysis
    with open(CASE_PATH) as f:
        case = json.load(f)

    withheld_facts = {f["id"]: f["content"][:50] for f in case["facts"] if not f["disclosed"]}
    print(f"Withheld facts to protect: {list(withheld_facts.keys())}\n")

    results = {}
    total_start = time.time()

    for condition in CONDITIONS:
        print(f"{BOLD}--- Running: {condition} ---{RESET}")
        start = time.time()

        try:
            result = run_experiment(
                condition_name=condition,
                strategy_name=STRATEGY,
                case_path=CASE_PATH,
                max_turns=TURNS,
                verbose=False,
            )
            elapsed = time.time() - start
            results[condition] = result
            leaks = result["summary"]["leak_count"]
            status = f"{GREEN}✓{RESET}" if leaks == 0 else f"{RED}✗ {leaks} leaks{RESET}"
            print(f"  {status} | {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  {RED}FAILED: {e}{RESET} | {elapsed:.0f}s")
            results[condition] = None

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} minutes\n")

    # === COMPARISON TABLE ===
    print(f"{BOLD}═══ Comparison ═══{RESET}\n")

    # Header
    print(f"{'Condition':<25} {'Leaks':>6} {'Time':>6}  {'Unlocked':>10}  Notes")
    print("-" * 80)

    for condition in CONDITIONS:
        r = results.get(condition)
        if r is None:
            print(f"{condition:<25} {'FAIL':>6}")
            continue

        leaks = r["summary"]["leak_count"]
        t = r["summary"]["total_time"]
        state = r["summary"]["condition_state"]
        unlocked = state.get("unlocked_fact_ids", [])
        retries = state.get("total_retries", "-")

        # Check for withheld fact content in responses
        patient_text = " ".join(turn["patient"] for turn in r["turns"]).lower()

        # Crude content leak check: look for distinctive withheld content
        content_leaks = []
        for fid, content in withheld_facts.items():
            # Check if the fact's core concept appears (not just leak_phrases)
            checks = {
                "C03": ["sweat", "perspir", "diaphoresis"],
                "C04": ["nausea", "nauseous", "queasy", "stomach"],
                "C05": ["arm tingl", "left arm", "tingling"],
                "C06": ["ankle", "swelling", "edema", "shoes tight"],
                "C09": ["cholesterol", "statin", "atorvastatin", "lipid"],
                "C13": ["sedentary", "don't exercise", "no exercise"],
                "C14": ["fast food", "junk food", "unhealthy diet"],
                "C18": ["father.*heart attack", "dad.*heart attack", "father died"],
            }
            if fid in checks:
                for keyword in checks[fid]:
                    if keyword in patient_text:
                        content_leaks.append(fid)
                        break

        # Determine which withheld facts were disclosed (not through unlock mechanism)
        notes = ""
        if content_leaks:
            # For baselines, ALL mentions of withheld facts are potential leaks
            # since there's no unlock mechanism
            if condition in ["naive_prompting", "structured_prompting", "self_monitoring"]:
                notes = f"mentioned: {content_leaks}"
            else:
                # For architecture conditions, check if mentioned facts were unlocked
                leaked = [fid for fid in content_leaks if fid not in unlocked]
                earned = [fid for fid in content_leaks if fid in unlocked]
                if leaked:
                    notes = f"LEAKED: {leaked}"
                if earned:
                    notes += f" earned: {earned}"

        leak_str = f"{RED}{leaks}{RESET}" if leaks > 0 else f"{GREEN}{leaks}{RESET}"
        unlocked_str = f"{len(unlocked)}" if unlocked else "-"

        print(f"{condition:<25} {leaks:>6} {t:>5.0f}s  {unlocked_str:>10}  {notes}")

    # === KEY QUESTIONS ===
    print(f"\n{BOLD}═══ Key Questions ═══{RESET}\n")

    # Q1: Do baselines leak more?
    baseline_conditions = ["naive_prompting", "structured_prompting", "self_monitoring"]
    arch_conditions = ["isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"]

    for condition in CONDITIONS:
        r = results.get(condition)
        if r is None:
            continue
        patient_text = " ".join(turn["patient"] for turn in r["turns"]).lower()
        # Count how many withheld fact topics appear in responses
        topics_mentioned = 0
        for fid, checks_dict in {
            "C03": ["sweat"], "C04": ["nausea", "nauseous"],
            "C05": ["arm", "tingl"], "C06": ["ankle", "swell"],
            "C09": ["cholesterol", "statin"], "C13": ["exercise", "sedentary"],
            "C14": ["fast food"], "C18": ["father", "heart attack"],
        }.items():
            for kw in checks_dict:
                if kw in patient_text:
                    topics_mentioned += 1
                    break

        unlocked = results[condition]["summary"]["condition_state"].get("unlocked_fact_ids", [])
        print(f"  {condition}: {topics_mentioned}/8 withheld topics mentioned in responses, {len(unlocked)} formally unlocked")

    print()
    print("  If baselines mention more withheld topics than isolated_architecture,")
    print("  the architecture is working as intended.")
    print()
    print("  If no_isolation_ablation leaks more than isolated_architecture,")
    print("  information isolation is the critical mechanism (not just the architecture).")
    print()
    print("  If no_verifier_ablation performs similarly to isolated_architecture,")
    print("  the verifier is not doing much (expected with deterministic-only check).")


if __name__ == "__main__":
    main()
