"""
Experiment Runner

Runs a single conversation: one condition × one strategy × one case.
Saves results as JSON for later analysis.

Usage:
    python3 run_experiment.py <condition> <strategy> [--case <case_path>] [--turns <n>]

Examples:
    python3 run_experiment.py naive_prompting direct_questioning
    python3 run_experiment.py isolated_architecture emotional_appeal
    python3 run_experiment.py --list
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.conditions import get_condition, get_condition_names
from src.student_agent import get_strategy, get_strategy_names


DEFAULT_CASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cases",
    "case_cardiology.json",
)

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def run_experiment(
    condition_name: str,
    strategy_name: str,
    case_path: str = DEFAULT_CASE,
    max_turns: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Run a single experiment and return results.

    Returns:
        {
            "condition": str,
            "strategy": str,
            "case_id": str,
            "turns": [...],
            "summary": {...}
        }
    """
    condition = get_condition(condition_name, case_path)

    # Derive case_name from case file for case-specific strategies
    with open(case_path) as f:
        case = json.load(f)
    case_name_map = {
        "CARDIO-001": "cardiology",
        "RESP-001": "respiratory",
        "GI-001": "gi",
    }
    case_name = case_name_map.get(case["case_id"], "cardiology")
    student_messages = get_strategy(strategy_name, case_name)
    turns = min(max_turns, len(student_messages))

    if verbose:
        print(f"\n{BOLD}═══ Experiment ═══{RESET}")
        print(f"Condition: {condition_name}")
        print(f"Strategy: {strategy_name}")
        print(f"Case: {case['case_id']}")
        print(f"Turns: {turns}\n")

    results = {
        "condition": condition_name,
        "strategy": strategy_name,
        "case_id": case["case_id"],
        "case_path": case_path,
        "turns": [],
        "summary": {},
    }

    total_time = 0

    for i in range(turns):
        student_msg = student_messages[i]
        turn_num = i + 1

        if verbose:
            print(f"{BOLD}--- Turn {turn_num} ---{RESET}")
            print(f"{BLUE}Student:{RESET} {student_msg}")

        start = time.time()
        patient_response = condition.process_turn(student_msg)
        elapsed = time.time() - start
        total_time += elapsed

        if verbose:
            print(f"{GREEN}Patient:{RESET} {patient_response}")
            print(f"  ({elapsed:.1f}s)\n")

        results["turns"].append({
            "turn": turn_num,
            "student": student_msg,
            "patient": patient_response,
            "time": round(elapsed, 1),
        })

    # Summary
    state = condition.get_state()
    results["summary"] = {
        "total_turns": turns,
        "total_time": round(total_time, 1),
        "avg_time_per_turn": round(total_time / turns, 1),
        "condition_state": state,
    }

    # Leak check against full case
    patient_text = " ".join(t["patient"] for t in results["turns"]).lower()
    unlocked = set(state.get("unlocked_fact_ids", []))

    leaks = []
    for fact in case["facts"]:
        if fact["disclosed"] or fact["id"] in unlocked:
            continue
        for phrase in fact.get("leak_phrases", []):
            if phrase.lower() in patient_text:
                leaks.append({"fact_id": fact["id"], "phrase": phrase})

    results["summary"]["leaks"] = leaks
    results["summary"]["leak_count"] = len(leaks)

    if verbose:
        print(f"{BOLD}═══ Summary ═══{RESET}")
        print(f"Condition: {condition_name}")
        print(f"Strategy: {strategy_name}")
        print(f"Time: {total_time:.0f}s ({total_time/turns:.1f}s/turn)")
        print(f"State: {json.dumps(state, indent=2)}")
        if leaks:
            print(f"{RED}Leaks: {leaks}{RESET}")
        else:
            print(f"{GREEN}✓ No leaks detected{RESET}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("condition", nargs="?", help="Condition name")
    parser.add_argument("strategy", nargs="?", help="Strategy name")
    parser.add_argument("--case", default=DEFAULT_CASE, help="Path to case JSON")
    parser.add_argument("--turns", type=int, default=20, help="Max turns")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--list", action="store_true", help="List conditions and strategies")
    parser.add_argument("--quiet", action="store_true", help="Suppress turn-by-turn output")

    args = parser.parse_args()

    if args.list:
        print("Conditions:")
        for name in get_condition_names():
            print(f"  {name}")
        print("\nStrategies:")
        for name in get_strategy_names():
            print(f"  {name}")
        sys.exit(0)

    if not args.condition or not args.strategy:
        parser.print_help()
        sys.exit(1)

    # Check Ollama
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    results = run_experiment(
        condition_name=args.condition,
        strategy_name=args.strategy,
        case_path=args.case,
        max_turns=args.turns,
        verbose=not args.quiet,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
