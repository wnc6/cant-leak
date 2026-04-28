"""
Quick summary of all experiment results across runs.
Run this to see mean ± SD before designing charts.

Usage:
    python3 summarize_runs.py results/
"""

import json
import os
import sys
import glob
from collections import defaultdict

def load_results(results_dir):
    """Load all result files and extract key metrics."""
    results = []
    for filepath in sorted(glob.glob(os.path.join(results_dir, "**", "run_*.json"), recursive=True)):
        with open(filepath) as f:
            data = json.load(f)
        
        # Extract from path: case/condition/strategy/run_N.json
        parts = os.path.relpath(filepath, results_dir).split(os.sep)
        if len(parts) < 4:
            continue
        
        case = parts[0]
        condition = parts[1]
        strategy = parts[2]
        run = parts[3].replace(".json", "")
        
        # Extract from summary
        summary = data.get("summary", {})
        det_leaks = summary.get("leak_count", 0)
        
        # Count unlocked facts
        unlocked = summary.get("unlocked_count", 0)
        # Try condition_state for architecture conditions
        cond_state = summary.get("condition_state", {})
        if unlocked == 0 and "unlocked_fact_ids" in cond_state:
            unlocked = len(cond_state["unlocked_fact_ids"])
        
        results.append({
            "case": case,
            "condition": condition,
            "strategy": strategy,
            "run": run,
            "leaks": det_leaks,
            "unlocked": unlocked,
            "filepath": filepath,
        })
    
    return results

def mean_sd(values):
    n = len(values)
    if n == 0:
        return 0, 0
    m = sum(values) / n
    if n == 1:
        return m, 0
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, variance ** 0.5

def mann_whitney_u(x, y):
    """Simple Mann-Whitney U. Returns U statistic and approximate p-value."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0, 1.0
    
    combined = [(v, 'x') for v in x] + [(v, 'y') for v in y]
    combined.sort(key=lambda t: t[0])
    
    # Assign ranks with tie handling
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed
        for k in range(i, j):
            ranks.append((combined[k][0], combined[k][1], avg_rank))
        i = j
    
    r1 = sum(r for _, g, r in ranks if g == 'x')
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    
    # Normal approximation for p-value
    mu = n1 * n2 / 2
    sigma = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
    if sigma == 0:
        return u, 1.0
    z = abs((u - mu) / sigma)
    
    # Approximate two-tailed p-value using z
    import math
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return u, p

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 summarize_runs.py results/")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    results = load_results(results_dir)
    
    print(f"Loaded {len(results)} result files\n")
    
    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r["leaks"])
    
    condition_order = [
        "naive_prompting", "structured_prompting", "self_monitoring",
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]
    condition_labels = {
        "naive_prompting": "Naive Prompting",
        "structured_prompting": "Structured Prompting",
        "self_monitoring": "Self-Monitoring",
        "isolated_architecture": "Isolated Architecture",
        "no_isolation_ablation": "No-Isolation Ablation",
        "no_verifier_ablation": "No-Verifier Ablation",
    }
    
    # === Table 1: Leakage by condition ===
    print("=" * 65)
    print("LEAKAGE BY CONDITION (mean ± SD across all runs)")
    print("=" * 65)
    print(f"{'Condition':<25} {'N':>4} {'Mean':>6} {'SD':>6} {'Min':>4} {'Max':>4}")
    print("-" * 65)
    
    isolated_leaks = by_condition.get("isolated_architecture", [])
    
    for c in condition_order:
        vals = by_condition.get(c, [])
        if not vals:
            continue
        m, sd = mean_sd(vals)
        label = condition_labels.get(c, c)
        print(f"{label:<25} {len(vals):>4} {m:>6.2f} {sd:>6.2f} {min(vals):>4} {max(vals):>4}")
    
    # === Table 2: Leakage by condition × strategy ===
    print(f"\n{'=' * 85}")
    print("LEAKAGE BY CONDITION × STRATEGY (mean ± SD)")
    print("=" * 85)
    
    by_cond_strat = defaultdict(list)
    for r in results:
        by_cond_strat[(r["condition"], r["strategy"])].append(r["leaks"])
    
    strategies = [
        "direct_questioning", "rephrasing", "emotional_appeal",
        "authority_claim", "gradual_escalation", "logical_inference",
    ]
    strat_labels = {
        "direct_questioning": "Direct",
        "rephrasing": "Rephrase",
        "emotional_appeal": "Emotion",
        "authority_claim": "Authority",
        "gradual_escalation": "Escalate",
        "logical_inference": "Logic",
    }
    
    header = f"{'Condition':<25}"
    for s in strategies:
        header += f" {strat_labels[s]:>10}"
    print(header)
    print("-" * 85)
    
    for c in condition_order:
        label = condition_labels.get(c, c)
        row = f"{label:<25}"
        for s in strategies:
            vals = by_cond_strat.get((c, s), [])
            if vals:
                m, sd = mean_sd(vals)
                row += f" {m:>4.1f}±{sd:<4.1f}"
            else:
                row += f" {'--':>10}"
        print(row)
    
    # === Table 3: Leakage by condition × case ===
    print(f"\n{'=' * 65}")
    print("LEAKAGE BY CONDITION × CASE (mean ± SD)")
    print("=" * 65)
    
    by_cond_case = defaultdict(list)
    for r in results:
        by_cond_case[(r["condition"], r["case"])].append(r["leaks"])
    
    cases = ["cardiology", "respiratory", "gi"]
    
    header = f"{'Condition':<25} {'Cardio':>10} {'Resp':>10} {'GI':>10}"
    print(header)
    print("-" * 65)
    
    for c in condition_order:
        label = condition_labels.get(c, c)
        row = f"{label:<25}"
        for case in cases:
            vals = by_cond_case.get((c, case), [])
            if vals:
                m, sd = mean_sd(vals)
                row += f" {m:>4.1f}±{sd:<4.1f}"
            else:
                row += f" {'--':>10}"
        print(row)
    
    # === Mann-Whitney: each baseline vs isolated ===
    print(f"\n{'=' * 65}")
    print("MANN-WHITNEY U: Each condition vs Isolated Architecture")
    print("=" * 65)
    print("Note: the isolated condition has zero variance by construction,")
    print("so these tests are degenerate. The magnitude of separation is")
    print("the more meaningful comparison (see report Table 2 caption).")
    print("p-values shown for reference only.")
    print()
    
    for c in condition_order:
        if c == "isolated_architecture":
            continue
        vals = by_condition.get(c, [])
        if not vals:
            continue
        label = condition_labels.get(c, c)
        u, p = mann_whitney_u(vals, isolated_leaks)
        m, sd = mean_sd(vals)
        print(f"{label:<25} {m:.2f}±{sd:.2f} vs 0.00±0.00  U={u:.0f}  p={p:.6f}  {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    
    # === Disclosure rate for architecture conditions ===
    print(f"\n{'=' * 65}")
    print("STRUCTURED vs NAIVE (suggestive only — see report §2.2)")
    print("=" * 65)
    print("The report flags this as a suggestive direction, not an")
    print("established finding. Caveats: borderline p-value that won't")
    print("survive multiple-comparisons correction; structured bundles")
    print("three changes from naive; direction inconsistent across cases.")
    print()

    naive_vals = by_condition.get("naive_prompting", [])
    struct_vals = by_condition.get("structured_prompting", [])

    nm, nsd = mean_sd(naive_vals)
    sm, ssd = mean_sd(struct_vals)

    print(f"Naive:      {nm:.2f} ± {nsd:.2f} (n={len(naive_vals)})")
    print(f"Structured: {sm:.2f} ± {ssd:.2f} (n={len(struct_vals)})")
    u, p = mann_whitney_u(naive_vals, struct_vals)
    print(f"Mann-Whitney U: {u:.0f}, p = {p:.4f} (uncorrected)")

    # === Disclosure rate for architecture conditions ===
    print(f"\n{'=' * 65}")
    print("DISCLOSURE RATE (architecture conditions only)")
    print("=" * 65)
    
    by_cond_unlock = defaultdict(list)
    for r in results:
        if r["condition"] in ["isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"]:
            by_cond_unlock[r["condition"]].append(r["unlocked"])
    
    # Get max withheld per case
    withheld_counts = {"cardiology": 8, "respiratory": 7, "gi": 8}
    
    by_cond_rate = defaultdict(list)
    for r in results:
        if r["condition"] in ["isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"]:
            max_w = withheld_counts.get(r["case"], 8)
            rate = r["unlocked"] / max_w * 100 if max_w > 0 else 0
            by_cond_rate[r["condition"]].append(rate)
    
    for c in ["isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"]:
        vals = by_cond_rate.get(c, [])
        if vals:
            m, sd = mean_sd(vals)
            label = condition_labels.get(c, c)
            print(f"{label:<25} {m:.1f}% ± {sd:.1f}%")


if __name__ == "__main__":
    main()
