"""
Generate publication-quality charts from experiment results.
Auto-detects single or multiple runs and adapts accordingly.

Usage:
    python3 generate_charts.py results/ --output charts/ --format pdf
"""

import json
import os
import sys
import glob
import argparse
from collections import defaultdict
import math

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Install matplotlib: pip3 install matplotlib")
    sys.exit(1)


# === Color scheme ===
COLORS = {
    "naive_prompting": "#E07B39",
    "structured_prompting": "#B5283D",
    "self_monitoring": "#D4A843",
    "isolated_architecture": "#1B7340",
    "no_isolation_ablation": "#2D5FAA",
    "no_verifier_ablation": "#6B4C9A",
}

CONDITION_ORDER = [
    "naive_prompting", "structured_prompting", "self_monitoring",
    "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
]

CONDITION_LABELS = {
    "naive_prompting": "Naive\nPrompting",
    "structured_prompting": "Structured\nPrompting",
    "self_monitoring": "Self-\nMonitoring",
    "isolated_architecture": "Isolated\nArchitecture",
    "no_isolation_ablation": "No-Isolation\nAblation",
    "no_verifier_ablation": "No-Verifier\nAblation",
}

CONDITION_LABELS_SHORT = {
    "naive_prompting": "Naive",
    "structured_prompting": "Structured",
    "self_monitoring": "Self-Monitoring",
    "isolated_architecture": "Isolated",
    "no_isolation_ablation": "No-Isolation Abl.",
    "no_verifier_ablation": "No-Verifier Abl.",
}

STRATEGY_ORDER = [
    "direct_questioning", "rephrasing", "emotional_appeal",
    "authority_claim", "gradual_escalation", "logical_inference",
]

STRATEGY_LABELS = {
    "direct_questioning": "Direct",
    "rephrasing": "Rephrasing",
    "emotional_appeal": "Emotional\nAppeal",
    "authority_claim": "Authority\nClaim",
    "gradual_escalation": "Gradual\nEscalation",
    "logical_inference": "Logical\nInference",
}

CASE_ORDER = ["cardiology", "respiratory", "gi"]
CASE_LABELS = {"cardiology": "Cardiology", "respiratory": "Respiratory", "gi": "GI"}

WITHHELD_COUNTS = {"cardiology": 8, "respiratory": 7, "gi": 8}


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.fontsize': 7.5,
        'legend.framealpha': 0.9,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


def mean_sd(values):
    n = len(values)
    if n == 0:
        return 0, 0
    m = sum(values) / n
    if n == 1:
        return m, 0
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, variance ** 0.5


def load_results(results_dir):
    results = []
    for filepath in sorted(glob.glob(os.path.join(results_dir, "**", "run_*.json"), recursive=True)):
        with open(filepath) as f:
            data = json.load(f)

        parts = os.path.relpath(filepath, results_dir).split(os.sep)
        if len(parts) < 4:
            continue

        case = parts[0]
        condition = parts[1]
        strategy = parts[2]
        run = parts[3].replace(".json", "")

        summary = data.get("summary", {})
        det_leaks = summary.get("leak_count", 0)

        cond_state = summary.get("condition_state", {})
        unlocked = 0
        if "unlocked_fact_ids" in cond_state:
            unlocked = len(cond_state["unlocked_fact_ids"])

        results.append({
            "case": case,
            "condition": condition,
            "strategy": strategy,
            "run": run,
            "leaks": det_leaks,
            "unlocked": unlocked,
            "turns": data.get("turns", []),
            "summary": summary,
        })

    runs = set(r["run"] for r in results)
    n_runs = len(runs)
    print(f"Loaded {len(results)} results across {n_runs} run(s)")

    return results, n_runs


def mann_whitney_p(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 1.0

    combined = [(v, 'x') for v in x] + [(v, 'y') for v in y]
    combined.sort(key=lambda t: t[0])

    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks.append((combined[k][0], combined[k][1], avg_rank))
        i = j

    r1 = sum(r for _, g, r in ranks if g == 'x')
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)

    mu = n1 * n2 / 2
    sigma = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
    if sigma == 0:
        return 1.0
    z = abs((u - mu) / sigma)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return p


# ============================================================
# Chart 1: Leakage by Condition (bar + error bars)
# ============================================================
def chart_leakage_by_condition(results, output_dir, fmt, n_runs):
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r["leaks"])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(CONDITION_ORDER))
    means = []
    sds = []
    colors = []

    for c in CONDITION_ORDER:
        vals = by_condition.get(c, [0])
        m, sd = mean_sd(vals)
        means.append(m)
        sds.append(sd)
        colors.append(COLORS[c])

    bars = ax.bar(x, means, yerr=sds if n_runs > 1 else None,
                  capsize=4, color=colors, edgecolor='white', width=0.65,
                  error_kw={'linewidth': 1.2, 'color': '#333'})

    for i, (m, sd) in enumerate(zip(means, sds)):
        if n_runs > 1 and sd > 0:
            label = f'{m:.1f}\u00b1{sd:.1f}'
        else:
            label = f'{m:.1f}'
        ax.text(i, m + sd + 0.15 if sd > 0 else m + 0.15, label,
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Annotate zero bar — most important result, least visible
    iso_idx = CONDITION_ORDER.index("isolated_architecture")
    ax.annotate('zero leaks', xy=(iso_idx, 0), xytext=(iso_idx, 1.2),
                ha='center', fontsize=7, color='#1B7340', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#1B7340', lw=1.2))

    if n_runs > 1:
        naive_vals = by_condition.get("naive_prompting", [])
        struct_vals = by_condition.get("structured_prompting", [])
        p = mann_whitney_p(naive_vals, struct_vals)
        if p < 0.05:
            y_max = max(means[0] + sds[0], means[1] + sds[1]) + 0.8
            ax.plot([0, 0, 1, 1], [y_max - 0.2, y_max, y_max, y_max - 0.2],
                    color='#333', linewidth=1)
            ax.text(0.5, y_max + 0.1, f'p={p:.3f}', ha='center', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=7.5)
    ax.set_ylabel('Mean Leaked Facts per Conversation')
    ax.set_title('Leakage by Condition' + (f' (n={n_runs} runs)' if n_runs > 1 else ''))

    plt.tight_layout()
    path = os.path.join(output_dir, f'1_leakage_by_condition.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Chart 2: Leakage by Strategy (grouped bar + error bars)
# ============================================================
def chart_leakage_by_strategy(results, output_dir, fmt, n_runs):
    by_cond_strat = defaultdict(list)
    for r in results:
        by_cond_strat[(r["condition"], r["strategy"])].append(r["leaks"])

    fig, ax = plt.subplots(figsize=(8, 4))

    plot_conditions = [
        "naive_prompting", "structured_prompting", "self_monitoring",
    ]

    # Sort strategies by ascending mean leakage (averaged across baselines)
    strat_means = {}
    for s in STRATEGY_ORDER:
        all_vals = []
        for c in plot_conditions:
            all_vals.extend(by_cond_strat.get((c, s), [0]))
        strat_means[s] = sum(all_vals) / len(all_vals) if all_vals else 0
    sorted_strategies = sorted(STRATEGY_ORDER, key=lambda s: strat_means[s])

    n_conds = len(plot_conditions)
    n_strats = len(sorted_strategies)
    bar_width = 0.22
    x = np.arange(n_strats)

    for i, c in enumerate(plot_conditions):
        means = []
        sds = []
        for s in sorted_strategies:
            vals = by_cond_strat.get((c, s), [0])
            m, sd = mean_sd(vals)
            means.append(m)
            sds.append(sd)

        offset = (i - n_conds / 2 + 0.5) * bar_width
        ax.bar(x + offset, means,
               yerr=sds if n_runs > 1 else None,
               width=bar_width, capsize=2,
               color=COLORS[c], edgecolor='white',
               label=CONDITION_LABELS_SHORT[c],
               error_kw={'linewidth': 0.8, 'color': '#333'})

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in sorted_strategies], fontsize=8)
    ax.set_ylabel('Mean Leaked Facts per Conversation')
    ax.set_title('Baseline Leakage by Pressure Strategy')
    ax.text(0.5, -0.15, 'Isolated architecture: 0.0 across all strategies (not shown)',
            transform=ax.transAxes, ha='center', fontsize=7.5, fontstyle='italic', color='#1B7340')
    ax.legend(loc='upper left', ncol=3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'2_leakage_by_strategy.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Chart 3: Heatmap (condition x case)
# ============================================================
def chart_leakage_heatmap(results, output_dir, fmt, n_runs):
    by_cond_case = defaultdict(list)
    for r in results:
        by_cond_case[(r["condition"], r["case"])].append(r["leaks"])

    fig, ax = plt.subplots(figsize=(5, 4))

    data_matrix = []
    for c in CONDITION_ORDER:
        row = []
        for case in CASE_ORDER:
            vals = by_cond_case.get((c, case), [0])
            m, _ = mean_sd(vals)
            row.append(m)
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    im = ax.imshow(data_matrix, cmap='Reds', aspect='auto', vmin=0)

    for i in range(len(CONDITION_ORDER)):
        for j in range(len(CASE_ORDER)):
            vals = by_cond_case.get((CONDITION_ORDER[i], CASE_ORDER[j]), [0])
            m, sd = mean_sd(vals)
            text_color = 'white' if m > 4 else 'black'
            if n_runs > 1:
                cell_text = f'{m:.1f}\n\u00b1{sd:.1f}'
            else:
                cell_text = f'{m:.1f}'
            ax.text(j, i, cell_text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(CASE_ORDER)))
    ax.set_xticklabels([CASE_LABELS[c] for c in CASE_ORDER])
    ax.set_yticks(range(len(CONDITION_ORDER)))
    ax.set_yticklabels([CONDITION_LABELS_SHORT[c] for c in CONDITION_ORDER], fontsize=8)
    ax.set_title('Mean Leakage by Condition and Case')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Leaks', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f'3_leakage_heatmap.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Chart 4: Per-Turn Timeline (cumulative leakage)
# ============================================================
def chart_per_turn_timeline(results, output_dir, fmt, n_runs):
    # Load ALL leak phrases from case files (withheld facts)
    case_leak_phrases = {}
    for case in CASE_ORDER:
        case_path = f"cases/case_{case}.json"
        if os.path.exists(case_path):
            with open(case_path) as f2:
                case_data = json.load(f2)
            phrases = []
            for fact in case_data.get("facts", []):
                if not fact.get("disclosed", True):
                    for phrase in fact.get("leak_phrases", []):
                        phrases.append(phrase.lower())
            case_leak_phrases[case] = phrases

    # Also collect additional phrases from result summaries
    for r in results:
        case = r["case"]
        if case not in case_leak_phrases:
            case_leak_phrases[case] = []
        for leak in r["summary"].get("leaks", []):
            phrase = leak.get("phrase", "").lower()
            if phrase and phrase not in case_leak_phrases[case]:
                case_leak_phrases[case].append(phrase)

    by_cond_turn = defaultdict(lambda: defaultdict(list))

    for r in results:
        phrases = case_leak_phrases.get(r["case"], [])
        cumulative = 0
        leaked_phrases = set()
        for turn in r["turns"]:
            patient = turn.get("patient", "").lower()
            for phrase in phrases:
                if phrase in patient and phrase not in leaked_phrases:
                    cumulative += 1
                    leaked_phrases.add(phrase)
            turn_num = turn.get("turn", 0)
            if turn_num > 0:
                by_cond_turn[r["condition"]][turn_num].append(cumulative)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Only plot baselines — different markers to distinguish where lines overlap
    baseline_styles = [
        ("naive_prompting", 'o'),       # circle
        ("structured_prompting", 's'),  # square
        ("self_monitoring", 'D'),       # diamond
    ]

    for c, marker in baseline_styles:
        turn_data = by_cond_turn.get(c, {})
        turns = sorted(turn_data.keys())
        if not turns:
            continue
        means = [mean_sd(turn_data[t])[0] for t in turns]
        ax.plot(turns, means, color=COLORS[c], linewidth=2,
                label=CONDITION_LABELS_SHORT[c], marker=marker, markersize=4)

    # Isolated architecture: flat zero line — thicker + triangle marker to stand out from x-axis
    ax.plot(range(1, 21), [0] * 20, color=COLORS["isolated_architecture"], linewidth=3,
            label=CONDITION_LABELS_SHORT["isolated_architecture"], marker='^', markersize=4)

    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Cumulative Leaked Facts')
    ax.set_title('Cumulative Leakage Over 20-Turn Conversations')
    ax.legend(loc='upper left', fontsize=7)
    ax.set_xlim(1, 20)
    ax.set_xticks(range(1, 21, 2))

    plt.tight_layout()
    path = os.path.join(output_dir, f'4_per_turn_timeline.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Chart 5: Disclosure Rate by Strategy (isolated architecture)
# ============================================================
def chart_disclosure_by_strategy(results, output_dir, fmt, n_runs):
    by_strat = defaultdict(list)
    for r in results:
        if r["condition"] != "isolated_architecture":
            continue
        max_w = WITHHELD_COUNTS.get(r["case"], 8)
        rate = r["unlocked"] / max_w * 100 if max_w > 0 else 0
        by_strat[r["strategy"]].append(rate)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(STRATEGY_ORDER))
    means = []
    sds = []

    for s in STRATEGY_ORDER:
        vals = by_strat.get(s, [0])
        m, sd = mean_sd(vals)
        means.append(m)
        sds.append(sd)

    base_color = '#8FBC8F'
    highlight_color = '#1B7340'
    max_idx = means.index(max(means))
    bar_colors = [highlight_color if i == max_idx else base_color for i in range(len(means))]

    bars = ax.bar(x, means, yerr=sds if n_runs > 1 else None,
                  capsize=4, color=bar_colors, edgecolor='white', width=0.6,
                  error_kw={'linewidth': 1.2, 'color': '#333'})

    for i, (m, sd) in enumerate(zip(means, sds)):
        label = f'{m:.0f}%'
        y_pos = m + sd + 1.5 if sd > 0 else m + 1.5
        ax.text(i, y_pos, label, ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color=highlight_color if i == max_idx else '#333')

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=8)
    ax.set_ylabel('Withheld Facts Disclosed (%)')
    ax.set_title('Disclosure Rate by Strategy (Isolated Architecture)')
    ax.set_ylim(0, 105)

    # Average line with label below the line on the left where there's clear space
    avg_val = np.mean(means)
    ax.axhline(y=avg_val, color='#666', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(0.02, avg_val - 4, f'avg: {avg_val:.0f}%',
            fontsize=7, color='#666', ha='left', va='top')

    plt.tight_layout()
    path = os.path.join(output_dir, f'5_disclosure_by_strategy.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Chart 6: Ablation + Disclosure (combined)
# ============================================================
def chart_ablation_disclosure(results, output_dir, fmt, n_runs):
    arch_conditions = [
        "isolated_architecture", "no_verifier_ablation", "no_isolation_ablation",
    ]
    arch_labels = [
        "Isolated\nArchitecture", "No-Verifier\nAblation", "No-Isolation\nAblation",
    ]
    arch_colors = [COLORS[c] for c in arch_conditions]

    by_condition = defaultdict(list)
    by_cond_rate = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r["leaks"])
        if r["condition"] in arch_conditions:
            max_w = WITHHELD_COUNTS.get(r["case"], 8)
            rate = r["unlocked"] / max_w * 100 if max_w > 0 else 0
            by_cond_rate[r["condition"]].append(rate)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5),
                                    gridspec_kw={'width_ratios': [1, 1]})

    y = range(len(arch_conditions))

    leak_means = []
    leak_sds = []
    for c in arch_conditions:
        m, sd = mean_sd(by_condition.get(c, [0]))
        leak_means.append(m)
        leak_sds.append(sd)

    bars1 = ax1.barh(y, leak_means,
                      xerr=leak_sds if n_runs > 1 else None,
                      capsize=3, color=arch_colors, edgecolor='white', height=0.55,
                      error_kw={'linewidth': 1, 'color': '#333'})
    for i, (m, sd) in enumerate(zip(leak_means, leak_sds)):
        if n_runs > 1:
            label = f'{m:.2f}\u00b1{sd:.2f}'
        else:
            label = f'{m:.1f}'
        ax1.text(max(m + sd + 0.02, 0.05), i, label,
                 va='center', fontsize=8, fontweight='bold')

    ax1.set_yticks(y)
    ax1.set_yticklabels(arch_labels, fontsize=8)
    ax1.set_xlabel('Mean Leaked Facts')
    ax1.set_title('Leakage')
    ax1.set_xlim(0, max(leak_means) + max(leak_sds) + 0.5)
    ax1.invert_yaxis()
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(axis='y', length=0)

    disc_means = []
    disc_sds = []
    for c in arch_conditions:
        m, sd = mean_sd(by_cond_rate.get(c, [0]))
        disc_means.append(m)
        disc_sds.append(sd)

    bars2 = ax2.barh(y, disc_means,
                      xerr=disc_sds if n_runs > 1 else None,
                      capsize=3, color=arch_colors, edgecolor='white', height=0.55,
                      error_kw={'linewidth': 1, 'color': '#333'})
    for i, (m, sd) in enumerate(zip(disc_means, disc_sds)):
        if n_runs > 1:
            label = f'{m:.0f}%\u00b1{sd:.0f}%'
        else:
            label = f'{m:.0f}%'
        ax2.text(m + sd + 1.5, i, label,
                 va='center', fontsize=8, fontweight='bold')

    ax2.set_yticks(y)
    ax2.set_yticklabels([''] * len(arch_conditions))
    ax2.set_xlabel('Withheld Facts Disclosed (%)')
    ax2.set_title('Disclosure Rate')
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', length=0)

    plt.tight_layout()
    path = os.path.join(output_dir, f'6_ablation_disclosure.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--output", default="charts/", help="Output directory")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"], help="Output format")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    setup_style()

    results, n_runs = load_results(args.results_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"\nGenerating charts ({args.format})...\n")

    chart_leakage_by_condition(results, args.output, args.format, n_runs)
    chart_leakage_by_strategy(results, args.output, args.format, n_runs)
    chart_leakage_heatmap(results, args.output, args.format, n_runs)
    chart_per_turn_timeline(results, args.output, args.format, n_runs)
    chart_disclosure_by_strategy(results, args.output, args.format, n_runs)
    chart_ablation_disclosure(results, args.output, args.format, n_runs)

    if n_runs > 1:
        print("\n" + "=" * 65)
        print("STATISTICAL SUMMARY")
        print("=" * 65)

        by_condition = defaultdict(list)
        for r in results:
            by_condition[r["condition"]].append(r["leaks"])

        isolated = by_condition.get("isolated_architecture", [])

        for c in CONDITION_ORDER:
            if c == "isolated_architecture":
                continue
            vals = by_condition.get(c, [])
            m, sd = mean_sd(vals)
            p = mann_whitney_p(vals, isolated)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            label = CONDITION_LABELS_SHORT[c]
            print(f"  {label:<20} vs Isolated: {m:.2f}\u00b1{sd:.2f} vs 0.00\u00b10.00  p={p:.6f} {sig}")

        naive = by_condition.get("naive_prompting", [])
        struct = by_condition.get("structured_prompting", [])
        p = mann_whitney_p(naive, struct)
        nm, nsd = mean_sd(naive)
        sm, ssd = mean_sd(struct)
        print(f"\n  Structured vs Naive: {sm:.2f}\u00b1{ssd:.2f} vs {nm:.2f}\u00b1{nsd:.2f}  p={p:.4f} {'*' if p < 0.05 else 'ns'}")

    print(f"\nDone. Charts saved to {args.output}")


if __name__ == "__main__":
    main()
