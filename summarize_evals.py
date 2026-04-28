"""
Summarize GPT-4o-mini evaluation metrics across all runs.

Run after evaluate.py finishes for all 3 runs to get 3-run averaged numbers
for the report's supplementary metrics tables (Tables 4 and 7).

Note: this script intentionally does NOT report leakage. The report's
primary leakage measure is the runner's deterministic phrase matching
(summary.leak_count in the result files), which is computed during the
experiment and does not depend on this evaluation pipeline.

Usage:
    python3 summarize_evals.py evals/
"""

import json
import os
import sys
import glob
from collections import defaultdict


def mean_sd(values):
    n = len(values)
    if n == 0:
        return 0, 0
    m = sum(values) / n
    if n == 1:
        return m, 0
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, variance ** 0.5


def load_evals(evals_dir):
    evals = []
    for filepath in sorted(glob.glob(os.path.join(evals_dir, "**", "*_eval.json"), recursive=True)):
        with open(filepath) as f:
            data = json.load(f)
        parts = os.path.relpath(filepath, evals_dir).split(os.sep)
        if len(parts) < 4:
            continue
        evals.append({
            "case": parts[0],
            "condition": parts[1],
            "strategy": parts[2],
            "data": data,
        })
    return evals


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 summarize_evals.py evals/")
        sys.exit(1)

    evals_dir = sys.argv[1]
    evals = load_evals(evals_dir)
    print(f"Loaded {len(evals)} eval files\n")

    if not evals:
        print("No eval files found!")
        sys.exit(1)

    condition_order = [
        "naive_prompting", "structured_prompting", "self_monitoring",
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]
    condition_labels = {
        "naive_prompting": "Naive Prompting",
        "structured_prompting": "Structured Prompting",
        "self_monitoring": "Self-Monitoring",
        "isolated_architecture": "Isolated Architecture",
        "no_isolation_ablation": "No-Isolation Abl.",
        "no_verifier_ablation": "No-Verifier Abl.",
    }

    # Collect supplementary metrics by condition
    by_condition = defaultdict(lambda: {
        "contradictions": [], "l2l": [], "p2l": [], "naturalness": [],
        "generator_errors": 0, "planner_errors": 0, "verifier_misses": 0,
    })

    for e in evals:
        d = e["data"]
        c = e["condition"]

        by_condition[c]["contradictions"].append(d.get("total_contradictions", 0))
        by_condition[c]["l2l"].append(d.get("line_to_line_inconsistencies", 0))
        by_condition[c]["p2l"].append(d.get("prompt_to_line_inconsistencies", 0))

        nat = d.get("avg_naturalness", 0)
        if nat > 0:
            by_condition[c]["naturalness"].append(nat)

        # Failure attributions are at the top level. Each entry is a dict with
        # `attribution` -> {attribution: "generator_error" | "planner_error" | "verifier_miss" | ..., explanation: ...}
        for item in d.get("failure_attributions", []):
            if not isinstance(item, dict):
                continue
            inner = item.get("attribution", {})
            if isinstance(inner, dict):
                attr = inner.get("attribution", "")
            else:
                attr = inner if isinstance(inner, str) else ""
            attr_lower = attr.lower()
            if "generator" in attr_lower:
                by_condition[c]["generator_errors"] += 1
            elif "planner" in attr_lower:
                by_condition[c]["planner_errors"] += 1
            elif "verifier" in attr_lower:
                by_condition[c]["verifier_misses"] += 1

    # === Table 4: Consistency Metrics ===
    print("=" * 70)
    print("TABLE 4: Consistency Metrics (mean ± SD across all 3 runs)")
    print("=" * 70)
    print(f"{'Condition':<25} {'Contrad.':>14} {'L2L':>14} {'P2L':>14}")
    print("-" * 70)
    for c in condition_order:
        contrad = by_condition[c]["contradictions"]
        l2l = by_condition[c]["l2l"]
        p2l = by_condition[c]["p2l"]
        if not contrad:
            continue
        cm, csd = mean_sd(contrad)
        lm, lsd = mean_sd(l2l)
        pm, psd = mean_sd(p2l)
        label = condition_labels.get(c, c)
        print(f"{label:<25} {cm:>5.1f} ± {csd:<5.1f}  {lm:>5.1f} ± {lsd:<5.1f}  {pm:>5.1f} ± {psd:<5.1f}")
    print(f"\nN = {len(evals) // len(condition_order)} per condition")

    # === Naturalness ===
    print(f"\n{'=' * 70}")
    print("NATURALNESS (mean ± SD across all 3 runs)")
    print("=" * 70)
    print(f"{'Condition':<25} {'Mean':>8} {'SD':>8} {'N':>5}")
    print("-" * 70)
    for c in condition_order:
        vals = by_condition[c]["naturalness"]
        if not vals:
            continue
        m, sd = mean_sd(vals)
        label = condition_labels.get(c, c)
        print(f"{label:<25} {m:>8.2f} {sd:>8.2f} {len(vals):>5}")

    # === Table 7: Failure Attribution ===
    print(f"\n{'=' * 70}")
    print("TABLE 7: Failure Attribution (totals across all 3 runs)")
    print("=" * 70)

    total_gen = sum(by_condition[c]["generator_errors"] for c in condition_order)
    total_plan = sum(by_condition[c]["planner_errors"] for c in condition_order)
    total_ver = sum(by_condition[c]["verifier_misses"] for c in condition_order)
    total_all = total_gen + total_plan + total_ver

    if total_all == 0:
        print("No failure attribution data found in eval files.")
        print("Expected: top-level 'failure_attributions' list, each item having")
        print("an 'attribution' dict with an inner 'attribution' string.")
        return

    print(f"Generator errors:  {total_gen:>5} ({total_gen/total_all*100:.0f}%)")
    print(f"Planner errors:    {total_plan:>5} ({total_plan/total_all*100:.0f}%)")
    print(f"Verifier misses:   {total_ver:>5} ({total_ver/total_all*100:.0f}%)")
    print(f"Total:             {total_all:>5}")


if __name__ == "__main__":
    main()
