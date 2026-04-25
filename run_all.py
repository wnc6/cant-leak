#!/usr/bin/env python3
"""
Run All Experiments

Runs all combinations of conditions × strategies × cases × runs.
Saves results as JSON files in results/<case>/<condition>/<strategy>/run_<n>.json

Usage:
    python3 run_all.py                     # run everything
    python3 run_all.py --case cardiology   # one case only
    python3 run_all.py --condition naive_prompting --strategy direct_questioning  # one combo
    python3 run_all.py --dry-run           # show what would run without running
    python3 run_all.py --runs 1            # single run instead of 3
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.conditions import get_condition_names
from src.student_agent import get_strategy_names
from run_experiment import run_experiment


CASES = {
    "cardiology": "cases/case_cardiology.json",
    "respiratory": "cases/case_respiratory.json",
    "gi": "cases/case_gi.json",
}

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--case", choices=list(CASES.keys()), help="Run only this case")
    parser.add_argument("--condition", choices=get_condition_names(), help="Run only this condition")
    parser.add_argument("--strategy", choices=get_strategy_names(), help="Run only this strategy")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per combo (default: 3)")
    parser.add_argument("--turns", type=int, default=20, help="Turns per conversation (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="List experiments without running")
    parser.add_argument("--resume", action="store_true", help="Skip experiments with existing results")
    args = parser.parse_args()

    # Build experiment list
    cases = {args.case: CASES[args.case]} if args.case else CASES
    conditions = [args.condition] if args.condition else get_condition_names()
    strategies = [args.strategy] if args.strategy else get_strategy_names()

    experiments = []
    for case_name, case_path in cases.items():
        for condition in conditions:
            for strategy in strategies:
                for run in range(1, args.runs + 1):
                    output_path = os.path.join(
                        RESULTS_DIR, case_name, condition, strategy, f"run_{run}.json"
                    )
                    experiments.append({
                        "case_name": case_name,
                        "case_path": case_path,
                        "condition": condition,
                        "strategy": strategy,
                        "run": run,
                        "output": output_path,
                    })

    # Filter out completed experiments if resuming
    if args.resume:
        before = len(experiments)
        experiments = [e for e in experiments if not os.path.exists(e["output"])]
        skipped = before - len(experiments)
        if skipped:
            print(f"Resuming: skipping {skipped} completed experiments\n")

    total = len(experiments)
    print(f"Total experiments: {total}")
    print(f"Cases: {list(cases.keys())}")
    print(f"Conditions: {conditions}")
    print(f"Strategies: {strategies}")
    print(f"Runs per combo: {args.runs}")

    # Estimate time
    # ~20s per turn for isolated/ablation conditions (2 LLM calls)
    # ~10s per turn for baseline conditions (1 LLM call)
    # ~15s average × 20 turns = 300s per experiment
    est_seconds = total * 300
    est_hours = est_seconds / 3600
    print(f"Estimated time: {est_hours:.1f} hours\n")

    if args.dry_run:
        print("Experiments that would run:")
        for i, exp in enumerate(experiments, 1):
            status = "EXISTS" if os.path.exists(exp["output"]) else "PENDING"
            print(f"  {i:3d}. [{status}] {exp['case_name']}/{exp['condition']}/{exp['strategy']}/run_{exp['run']}")
        return

    # Check Ollama
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    # Run experiments
    completed = 0
    failed = 0
    start_time = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{total}: {exp['case_name']}/{exp['condition']}/{exp['strategy']}/run_{exp['run']}")
        print(f"{'='*60}")

        try:
            results = run_experiment(
                condition_name=exp["condition"],
                strategy_name=exp["strategy"],
                case_path=exp["case_path"],
                max_turns=args.turns,
                verbose=False,
            )

            # Add run metadata
            results["run"] = exp["run"]
            results["case_name"] = exp["case_name"]

            # Save
            os.makedirs(os.path.dirname(exp["output"]), exist_ok=True)
            with open(exp["output"], "w") as f:
                json.dump(results, f, indent=2)

            # Print summary
            state = results["summary"]["condition_state"]
            leaks = results["summary"]["leak_count"]
            t = results["summary"]["total_time"]
            unlocked = state.get("unlocked_fact_ids", [])

            status = f"{GREEN}✓{RESET}" if leaks == 0 else f"{RED}✗ {leaks} leaks{RESET}"
            print(f"  {status} | {len(unlocked)} unlocked | {t:.0f}s | saved to {exp['output']}")

            completed += 1

        except Exception as e:
            print(f"  {RED}FAILED: {e}{RESET}")
            failed += 1

        # Progress
        elapsed = time.time() - start_time
        rate = elapsed / i
        remaining = rate * (total - i)
        print(f"  Progress: {i}/{total} | Elapsed: {elapsed/60:.0f}m | ETA: {remaining/60:.0f}m")

    # Final summary
    print(f"\n{'='*60}")
    print(f"DONE: {completed} completed, {failed} failed, {total - completed - failed} skipped")
    print(f"Total time: {(time.time() - start_time)/60:.0f} minutes")
    print(f"Results in: {RESULTS_DIR}/")


# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


if __name__ == "__main__":
    main()
