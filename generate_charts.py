"""
Chart Generator (PDF-optimized)

Static PNG/PDF charts via matplotlib. Consistent colors, print-ready.

Usage:
    python3 generate_charts.py results/ --output charts/
    python3 generate_charts.py results/ --output charts/ --format pdf
"""

import json
import os
import sys
import glob
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


CONDITIONS = [
    "naive_prompting", "structured_prompting", "self_monitoring",
    "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
]
CONDITION_LABELS = {
    "naive_prompting": "Naive Prompting",
    "structured_prompting": "Structured Prompting",
    "self_monitoring": "Self-Monitoring",
    "isolated_architecture": "Isolated Architecture",
    "no_isolation_ablation": "No-Isolation Ablation",
    "no_verifier_ablation": "No-Verifier Ablation",
}
CONDITION_SHORT = {
    "naive_prompting": "Naive",
    "structured_prompting": "Structured",
    "self_monitoring": "Self-Monitor",
    "isolated_architecture": "Isolated",
    "no_isolation_ablation": "No-Isolation",
    "no_verifier_ablation": "No-Verifier",
}
COLORS = {
    "naive_prompting":       "#B22234",
    "structured_prompting":  "#E07020",
    "self_monitoring":       "#D4A017",
    "isolated_architecture": "#1B7340",
    "no_isolation_ablation": "#2D5FAA",
    "no_verifier_ablation":  "#6B4C9A",
}
STRATEGIES = [
    "direct_questioning", "rephrasing", "emotional_appeal",
    "authority_claim", "gradual_escalation", "logical_inference",
]
STRATEGY_LABELS = {
    "direct_questioning": "Direct Questioning",
    "rephrasing": "Rephrasing",
    "emotional_appeal": "Emotional Appeal",
    "authority_claim": "Authority Claim",
    "gradual_escalation": "Gradual Escalation",
    "logical_inference": "Logical Inference",
}
STRATEGY_SHORT = {
    "direct_questioning": "Direct",
    "rephrasing": "Rephrase",
    "emotional_appeal": "Emotional",
    "authority_claim": "Authority",
    "gradual_escalation": "Gradual",
    "logical_inference": "Logical",
}
CASE_LABELS = {"cardiology": "Cardiology", "respiratory": "Respiratory", "gi": "GI"}
BASELINES = {"naive_prompting", "structured_prompting", "self_monitoring"}
ARCHITECTURE = {"isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"}


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_all_results(results_dir):
    pattern = os.path.join(results_dir, "**", "run_*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    results = []
    for filepath in files:
        with open(filepath) as f:
            result = json.load(f)
        parts = os.path.relpath(filepath, results_dir).split(os.sep)
        if len(parts) >= 4:
            result["case_name"] = parts[0]
        results.append(result)
    return results


def get_leak_data(results):
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r.get("condition", "")][r.get("strategy", "")].append(
            r.get("summary", {}).get("leak_count", 0)
        )
    return data


def save(fig, output_dir, name, fmt):
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# ═══════════════════════════════════════════════
# Chart 1: Leakage by strategy (grouped bar)
# ═══════════════════════════════════════════════

def chart_leakage_by_strategy(results, output_dir, fmt):
    data = get_leak_data(results)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.grid(axis='y', alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)

    x = np.arange(len(STRATEGIES))
    width = 0.13
    offsets = np.arange(len(CONDITIONS)) - (len(CONDITIONS) - 1) / 2

    for i, c in enumerate(CONDITIONS):
        vals = [np.mean(data[c].get(s, [0])) for s in STRATEGIES]
        ax.bar(x + offsets[i] * width, vals, width * 0.88,
               label=CONDITION_SHORT[c], color=COLORS[c],
               edgecolor='white', linewidth=0.3)

    ax.set_xlabel('Pressure Strategy')
    ax.set_ylabel('Mean Leaked Facts per Conversation')
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_SHORT[s] for s in STRATEGIES])
    ax.legend(ncol=3, loc='upper left', framealpha=0.9,
              handlelength=1.2, columnspacing=1)
    ax.set_ylim(bottom=0)
    ax.set_title('Leakage Rate by Pressure Strategy Across Conditions')

    save(fig, output_dir, "1_leakage_by_strategy", fmt)


# ═══════════════════════════════════════════════
# Chart 2: Heatmap
# ═══════════════════════════════════════════════

def chart_heatmap(results, output_dir, fmt):
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r.get("condition", "")][r.get("case_name", "")].append(
            r.get("summary", {}).get("leak_count", 0)
        )

    cases = ["cardiology", "respiratory", "gi"]
    grid = np.zeros((len(CONDITIONS), len(cases)))
    for i, c in enumerate(CONDITIONS):
        for j, case in enumerate(cases):
            vals = data[c].get(case, [0])
            grid[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(grid, cmap='Reds', aspect='auto', vmin=0)

    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([CASE_LABELS[c] for c in cases])
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=8)

    for i in range(len(CONDITIONS)):
        for j in range(len(cases)):
            v = grid[i, j]
            color = "white" if v > grid.max() * 0.45 else "black"
            ax.text(j, i, f"{v:.1f}", ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_title('Mean Leakage by Condition and Clinical Case')
    fig.colorbar(im, ax=ax, label='Leaked Facts', shrink=0.8, pad=0.02)

    save(fig, output_dir, "2_leakage_heatmap", fmt)


# ═══════════════════════════════════════════════
# Chart 3: Per-turn timeline
# ═══════════════════════════════════════════════

def chart_per_turn_timeline(results, output_dir, fmt):
    turn_leaks = defaultdict(lambda: defaultdict(int))

    for r in results:
        cond = r.get("condition", "")
        case_name = r.get("case_name", "cardiology")
        case_path = {"cardiology": "cases/case_cardiology.json",
                     "respiratory": "cases/case_respiratory.json",
                     "gi": "cases/case_gi.json"}.get(case_name, "")
        if not os.path.exists(case_path):
            continue

        with open(case_path) as f:
            case = json.load(f)

        withheld = [fact for fact in case["facts"] if not fact["disclosed"]]
        final_unlocked = set(
            r.get("summary", {}).get("condition_state", {}).get("unlocked_fact_ids", [])
        )
        cum_student = ""

        for turn in r.get("turns", []):
            t = turn["turn"]
            resp = turn["patient"].lower()
            cum_student += " " + turn["student"].lower()

            unlocked_here = set()
            if cond in ARCHITECTURE:
                for fact in withheld:
                    kws = fact.get("unlock_keywords", [])
                    if any(kw.lower() in cum_student for kw in kws):
                        if fact["id"] in final_unlocked:
                            unlocked_here.add(fact["id"])

            for fact in withheld:
                if fact["id"] in unlocked_here:
                    continue
                for phrase in fact.get("leak_phrases", []):
                    if phrase.lower() in resp:
                        turn_leaks[cond][t] += 1
                        break

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.grid(axis='y', alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)
    turns = list(range(1, 21))

    for c in CONDITIONS:
        n = sum(1 for r in results if r.get("condition") == c)
        if n == 0:
            continue
        cum = []
        total = 0
        for t in turns:
            total += turn_leaks[c].get(t, 0)
            cum.append(total / n)

        ls = '--' if c in ARCHITECTURE and c != "isolated_architecture" else '-'
        lw = 2.2 if c == "isolated_architecture" else 1.3
        ax.plot(turns, cum, label=CONDITION_SHORT[c],
                color=COLORS[c], linewidth=lw, linestyle=ls)

    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Mean Cumulative Leaked Facts')
    ax.set_title('Cumulative Leakage Over 20-Turn Conversations')
    ax.legend(ncol=2, loc='upper left', framealpha=0.9,
              handlelength=1.8, columnspacing=1)
    ax.set_xlim(1, 20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(bottom=0)

    save(fig, output_dir, "3_per_turn_timeline", fmt)


# ═══════════════════════════════════════════════
# Chart 4: Radar
# ═══════════════════════════════════════════════

def chart_radar(results, output_dir, fmt):
    data = get_leak_data(results)

    angles = np.linspace(0, 2 * np.pi, len(STRATEGIES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), subplot_kw=dict(polar=True))

    for c in CONDITIONS:
        vals = [np.mean(data[c].get(s, [0])) for s in STRATEGIES]
        vals += vals[:1]
        lw = 2.2 if c == "isolated_architecture" else 1.3
        ax.plot(angles, vals, label=CONDITION_SHORT[c],
                color=COLORS[c], linewidth=lw)
        ax.fill(angles, vals, color=COLORS[c], alpha=0.04)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([STRATEGY_SHORT[s] for s in STRATEGIES], fontsize=8.5)
    ax.set_title('Vulnerability Profile Across Pressure Strategies', pad=18)
    ax.legend(fontsize=7, loc='lower right', bbox_to_anchor=(1.28, -0.05),
              framealpha=0.9, handlelength=1.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    save(fig, output_dir, "4_vulnerability_radar", fmt)


# ═══════════════════════════════════════════════
# Chart 5: Disclosure rate
# ═══════════════════════════════════════════════

def chart_disclosure_rate(results, output_dir, fmt):
    arch = ["isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"]
    data = defaultdict(list)

    for r in results:
        cond = r.get("condition", "")
        if cond not in arch:
            continue
        state = r.get("summary", {}).get("condition_state", {})
        unlocked = len(state.get("unlocked_fact_ids", []))
        case_id = r.get("case_id", "")
        total = {"CARDIO-001": 8, "RESP-001": 7, "GI-001": 8}.get(case_id, 8)
        data[cond].append(unlocked / total * 100 if total else 0)

    labels = [CONDITION_LABELS[c] for c in arch]
    values = [np.mean(data[c]) if data[c] else 0 for c in arch]
    bar_colors = [COLORS[c] for c in arch]

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    y = range(len(arch))
    bars = ax.barh(y, values, color=bar_colors, edgecolor='white', height=0.55)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{v:.0f}%', va='center', fontsize=8.5, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Withheld Facts Successfully Disclosed (%)')
    ax.set_title('Disclosure Rate for Architecture Conditions')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    save(fig, output_dir, "5_disclosure_rate", fmt)


# ═══════════════════════════════════════════════
# Chart 6: Ablation comparison
# ═══════════════════════════════════════════════

def chart_ablation(results, output_dir, fmt):
    data = get_leak_data(results)

    ablation = ["isolated_architecture", "no_verifier_ablation", "no_isolation_ablation"]
    labels = [CONDITION_LABELS[c] for c in ablation]
    values = []
    for c in ablation:
        all_vals = [v for s in STRATEGIES for v in data[c].get(s, [0])]
        values.append(np.mean(all_vals) if all_vals else 0)
    bar_colors = [COLORS[c] for c in ablation]

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    y = range(len(ablation))
    bars = ax.barh(y, values, color=bar_colors, edgecolor='white', height=0.55)

    for bar, v in zip(bars, values):
        x_pos = max(bar.get_width() + 0.02, 0.03)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f}', va='center', fontsize=8.5, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Mean Leaked Facts per Conversation')
    ax.set_title('Ablation: Effect of Removing Isolation vs. Verifier')
    ax.set_xlim(0, max(values) * 1.4 if max(values) > 0 else 1)
    ax.invert_yaxis()
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    save(fig, output_dir, "6_ablation_comparison", fmt)


# ═══════════════════════════════════════════════

def print_text_summary(results):
    data = get_leak_data(results)
    print("\nLeakage by Strategy × Condition:\n")
    header = f"{'Strategy':<22}" + "".join(f"{CONDITION_SHORT[c]:>10}" for c in CONDITIONS)
    print(header)
    print("-" * len(header))
    for s in STRATEGIES:
        row = f"{STRATEGY_LABELS[s]:<22}"
        for c in CONDITIONS:
            vals = data[c].get(s, [0])
            row += f"{np.mean(vals):>10.1f}"
        print(row)
    print("\nCondition averages:")
    for c in CONDITIONS:
        all_vals = [v for s in STRATEGIES for v in data[c].get(s, [0])]
        print(f"  {CONDITION_LABELS[c]}: {np.mean(all_vals):.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", default="charts")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No results in {args.results_dir}")
        sys.exit(1)

    setup_style()
    print(f"Loaded {len(results)} results")
    print_text_summary(results)

    os.makedirs(args.output, exist_ok=True)
    fmt = args.format
    print(f"\nGenerating {fmt.upper()} charts:")
    chart_leakage_by_strategy(results, args.output, fmt)
    chart_heatmap(results, args.output, fmt)
    chart_per_turn_timeline(results, args.output, fmt)
    chart_radar(results, args.output, fmt)
    chart_disclosure_rate(results, args.output, fmt)
    chart_ablation(results, args.output, fmt)
    print("\nDone.")


if __name__ == "__main__":
    main()
