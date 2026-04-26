"""
Chart Generator

Generates publication-quality charts from experiment results.
Uses deterministic leak counts from results files (not GPT-4o-mini evals).

Charts:
1. Leakage by strategy (grouped bar) — main result
2. Leakage heatmap (condition × case) — cross-case comparison
3. Per-turn leakage timeline (line chart) — pressure accumulation
4. Vulnerability radar (spider chart) — per-condition profile
5. Disclosure rate (simple bar) — over-withholding check

Usage:
    python3 generate_charts.py results/ --output charts/
"""

import json
import os
import sys
import glob
from collections import defaultdict


CONDITIONS = [
    "naive_prompting", "structured_prompting", "self_monitoring",
    "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
]
CONDITION_LABELS = {
    "naive_prompting": "Naive",
    "structured_prompting": "Structured",
    "self_monitoring": "Self-Monitor",
    "isolated_architecture": "Isolated (ours)",
    "no_isolation_ablation": "No-Isolation",
    "no_verifier_ablation": "No-Verifier",
}
CONDITION_SHORT = {
    "naive_prompting": "Naive",
    "structured_prompting": "Struct",
    "self_monitoring": "Self-M",
    "isolated_architecture": "Isolat",
    "no_isolation_ablation": "No-Iso",
    "no_verifier_ablation": "No-Ver",
}
STRATEGIES = [
    "direct_questioning", "rephrasing", "emotional_appeal",
    "authority_claim", "gradual_escalation", "logical_inference",
]
STRATEGY_LABELS = {
    "direct_questioning": "Direct",
    "rephrasing": "Rephrase",
    "emotional_appeal": "Emotional",
    "authority_claim": "Authority",
    "gradual_escalation": "Gradual",
    "logical_inference": "Logical",
}
CONDITION_COLORS = {
    "naive_prompting": "#ef4444",
    "structured_prompting": "#f97316",
    "self_monitoring": "#eab308",
    "isolated_architecture": "#22c55e",
    "no_isolation_ablation": "#3b82f6",
    "no_verifier_ablation": "#8b5cf6",
}
CASE_LABELS = {"cardiology": "Cardiology", "respiratory": "Respiratory", "gi": "GI"}
CASE_COLORS = {"Cardiology": "#3b82f6", "Respiratory": "#22c55e", "GI": "#f97316"}


def load_all_results(results_dir: str) -> list[dict]:
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


COMMON_STYLES = """
    body {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        max-width: 960px;
        margin: 40px auto;
        padding: 0 20px;
        background: #fff;
        color: #333;
    }
    h1 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
    .subtitle { font-size: 13px; color: #666; margin-bottom: 24px; }
    .chart-container { position: relative; height: 420px; margin-bottom: 40px; }
"""


# ═══════════════════════════════════════════════
# Chart 1: Leakage by Strategy (grouped bar)
# ═══════════════════════════════════════════════

def generate_leakage_by_strategy(results, output_dir):
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r.get("condition", "")][r.get("strategy", "")].append(
            r.get("summary", {}).get("leak_count", 0)
        )

    chart_data = []
    for s in STRATEGIES:
        row = {"strategy": STRATEGY_LABELS[s]}
        for c in CONDITIONS:
            vals = data[c].get(s, [0])
            row[CONDITION_LABELS[c]] = round(sum(vals) / len(vals), 1)
        chart_data.append(row)

    condition_names = [CONDITION_LABELS[c] for c in CONDITIONS]
    colors = {CONDITION_LABELS[c]: CONDITION_COLORS[c] for c in CONDITIONS}

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Leakage by Strategy</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{COMMON_STYLES}</style>
</head><body>
<h1>Leakage Rate by Pressure Strategy Across Conditions</h1>
<p class="subtitle">Average leaked withheld facts per 20-turn conversation. Lower is better. N=108 experiments (3 cases × 6 strategies × 6 conditions).</p>
<div class="chart-container"><canvas id="chart"></canvas></div>
<script>
const data = {json.dumps(chart_data)};
const conds = {json.dumps(condition_names)};
const colors = {json.dumps(colors)};
const datasets = conds.map(c => ({{
    label: c, data: data.map(d => d[c]),
    backgroundColor: colors[c] + '99', borderColor: colors[c], borderWidth: 1,
}}));
new Chart(document.getElementById('chart'), {{
    type: 'bar',
    data: {{ labels: data.map(d => d.strategy), datasets }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
        scales: {{
            y: {{ beginAtZero: true, title: {{ display: true, text: 'Avg Leaks', font: {{ size: 12 }} }} }},
            x: {{ title: {{ display: true, text: 'Pressure Strategy', font: {{ size: 12 }} }} }}
        }}
    }}
}});
</script></body></html>"""

    path = os.path.join(output_dir, "1_leakage_by_strategy.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════
# Chart 2: Heatmap (condition × case)
# ═══════════════════════════════════════════════

def generate_heatmap(results, output_dir):
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r.get("condition", "")][r.get("case_name", "")].append(
            r.get("summary", {}).get("leak_count", 0)
        )

    cases = ["cardiology", "respiratory", "gi"]
    grid = []
    for c in CONDITIONS:
        row = []
        for case in cases:
            vals = data[c].get(case, [0])
            row.append(round(sum(vals) / len(vals), 1))
        grid.append(row)

    cond_labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    case_labels = [CASE_LABELS[c] for c in cases]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Leakage Heatmap</title>
<style>
{COMMON_STYLES}
    .heatmap {{ border-collapse: collapse; width: 100%; }}
    .heatmap th {{ padding: 10px 16px; text-align: center; font-size: 13px; font-weight: 600; border-bottom: 2px solid #333; }}
    .heatmap td {{ padding: 14px 16px; text-align: center; font-size: 15px; font-weight: 600; border-bottom: 1px solid #eee; }}
    .heatmap td:first-child {{ text-align: left; font-weight: 400; font-size: 13px; }}
    .heatmap tr:last-child td {{ border-bottom: 2px solid #333; }}
</style>
</head><body>
<h1>Leakage Heatmap: Condition × Clinical Case</h1>
<p class="subtitle">Average leaks per conversation. Color intensity indicates severity.</p>
<table class="heatmap">
<tr><th></th>{"".join(f"<th>{c}</th>" for c in case_labels)}<th>Avg</th></tr>
{"".join(
    f'<tr><td>{cond_labels[i]}</td>' +
    "".join(
        f'<td style="background: rgba(239,68,68,{min(v/8, 1):.2f}); color: {"#fff" if v > 4 else "#333"}">{v}</td>'
        for v in grid[i]
    ) +
    f'<td style="background: rgba(239,68,68,{min(sum(grid[i])/3/8, 1):.2f}); color: {"#fff" if sum(grid[i])/3 > 4 else "#333"}">{sum(grid[i])/3:.1f}</td></tr>'
    for i in range(len(CONDITIONS))
)}
</table>
</body></html>"""

    path = os.path.join(output_dir, "2_leakage_heatmap.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════
# Chart 3: Per-turn leakage timeline (line chart)
# ═══════════════════════════════════════════════

def generate_per_turn_timeline(results, output_dir):
    """Shows cumulative leakage over turns for each condition."""
    # For each condition, track which turns had leaks across all experiments
    turn_leaks = defaultdict(lambda: defaultdict(int))
    turn_counts = defaultdict(int)

    for r in results:
        cond = r.get("condition", "")
        patient_texts = []
        for turn in r.get("turns", []):
            t = turn["turn"]
            turn_counts[cond] += 1
            resp = turn["patient"].lower()
            # Use leak_phrases from the case to detect per-turn leaks
            case_name = r.get("case_name", "cardiology")
            case_map = {
                "cardiology": "cases/case_cardiology.json",
                "respiratory": "cases/case_respiratory.json",
                "gi": "cases/case_gi.json",
            }
            case_path = case_map.get(case_name, "")
            if os.path.exists(case_path):
                with open(case_path) as f:
                    case = json.load(f)
                for fact in case["facts"]:
                    if fact["disclosed"]:
                        continue
                    for phrase in fact.get("leak_phrases", []):
                        if phrase.lower() in resp:
                            turn_leaks[cond][t] += 1
                            break

    # Build cumulative leak data per condition
    chart_datasets = []
    for c in CONDITIONS:
        cumulative = []
        total = 0
        # Get number of experiments for this condition
        n_experiments = sum(1 for r in results if r.get("condition") == c)
        if n_experiments == 0:
            continue
        for t in range(1, 21):
            total += turn_leaks[c].get(t, 0)
            cumulative.append(round(total / n_experiments, 2))
        chart_datasets.append({
            "label": CONDITION_LABELS[c],
            "data": cumulative,
            "borderColor": CONDITION_COLORS[c],
            "backgroundColor": CONDITION_COLORS[c] + "22",
            "fill": False,
            "tension": 0.3,
            "borderWidth": 2,
        })

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Per-Turn Leakage Timeline</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{COMMON_STYLES}</style>
</head><body>
<h1>Cumulative Leakage Over Conversation Turns</h1>
<p class="subtitle">Shows when baselines start leaking under sustained pressure. Flat line = no leakage. Rising line = accumulating leaks.</p>
<div class="chart-container"><canvas id="chart"></canvas></div>
<script>
const datasets = {json.dumps(chart_datasets)};
new Chart(document.getElementById('chart'), {{
    type: 'line',
    data: {{
        labels: Array.from({{length: 20}}, (_, i) => i + 1),
        datasets: datasets,
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
        scales: {{
            y: {{ beginAtZero: true, title: {{ display: true, text: 'Cumulative Leaks (avg per experiment)', font: {{ size: 12 }} }} }},
            x: {{ title: {{ display: true, text: 'Turn Number', font: {{ size: 12 }} }} }}
        }}
    }}
}});
</script></body></html>"""

    path = os.path.join(output_dir, "3_per_turn_timeline.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════
# Chart 4: Vulnerability radar (spider chart)
# ═══════════════════════════════════════════════

def generate_radar(results, output_dir):
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r.get("condition", "")][r.get("strategy", "")].append(
            r.get("summary", {}).get("leak_count", 0)
        )

    chart_datasets = []
    for c in CONDITIONS:
        values = []
        for s in STRATEGIES:
            vals = data[c].get(s, [0])
            values.append(round(sum(vals) / len(vals), 1))
        chart_datasets.append({
            "label": CONDITION_LABELS[c],
            "data": values,
            "borderColor": CONDITION_COLORS[c],
            "backgroundColor": CONDITION_COLORS[c] + "22",
            "borderWidth": 2,
            "pointRadius": 3,
        })

    strategy_names = [STRATEGY_LABELS[s] for s in STRATEGIES]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Vulnerability Radar</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{COMMON_STYLES}
    .chart-container {{ height: 500px; max-width: 600px; margin: 0 auto 40px; }}
</style>
</head><body>
<h1>Vulnerability Profile by Pressure Strategy</h1>
<p class="subtitle">Each axis is a pressure strategy. Distance from center = average leaks. Baselines have large shapes; isolated architecture collapses to the center.</p>
<div class="chart-container"><canvas id="chart"></canvas></div>
<script>
const datasets = {json.dumps(chart_datasets)};
new Chart(document.getElementById('chart'), {{
    type: 'radar',
    data: {{
        labels: {json.dumps(strategy_names)},
        datasets: datasets,
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
        scales: {{
            r: {{
                beginAtZero: true,
                ticks: {{ font: {{ size: 10 }} }},
                pointLabels: {{ font: {{ size: 12 }} }}
            }}
        }}
    }}
}});
</script></body></html>"""

    path = os.path.join(output_dir, "4_vulnerability_radar.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════
# Chart 5: Disclosure rate (simple bar)
# ═══════════════════════════════════════════════

def generate_disclosure_rate(results, output_dir):
    arch_conditions = ["isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"]
    data = defaultdict(lambda: defaultdict(list))

    for r in results:
        cond = r.get("condition", "")
        strat = r.get("strategy", "")
        if cond not in arch_conditions:
            continue
        state = r.get("summary", {}).get("condition_state", {})
        unlocked = len(state.get("unlocked_fact_ids", []))
        case_id = r.get("case_id", "")
        total = {"CARDIO-001": 8, "RESP-001": 7, "GI-001": 8}.get(case_id, 8)
        data[cond][strat].append(unlocked / total * 100 if total else 0)

    chart_data = []
    for s in STRATEGIES:
        row = {"strategy": STRATEGY_LABELS[s]}
        for c in arch_conditions:
            vals = data[c].get(s, [0])
            row[CONDITION_LABELS[c]] = round(sum(vals) / len(vals)) if vals else 0
        chart_data.append(row)

    arch_labels = [CONDITION_LABELS[c] for c in arch_conditions]
    arch_colors = {CONDITION_LABELS[c]: CONDITION_COLORS[c] for c in arch_conditions}

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Disclosure Rate</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{COMMON_STYLES}</style>
</head><body>
<h1>Disclosure Rate: Facts Earned Through Proper Questioning</h1>
<p class="subtitle">Percentage of withheld facts unlocked when the student asked the right questions. Higher = better. Shows the architecture is not over-withholding.</p>
<div class="chart-container"><canvas id="chart"></canvas></div>
<script>
const data = {json.dumps(chart_data)};
const conds = {json.dumps(arch_labels)};
const colors = {json.dumps(arch_colors)};
const datasets = conds.map(c => ({{
    label: c, data: data.map(d => d[c]),
    backgroundColor: colors[c] + '99', borderColor: colors[c], borderWidth: 1,
}}));
new Chart(document.getElementById('chart'), {{
    type: 'bar',
    data: {{ labels: data.map(d => d.strategy), datasets }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
        scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Disclosure Rate (%)', font: {{ size: 12 }} }} }} }}
    }}
}});
</script></body></html>"""

    path = os.path.join(output_dir, "5_disclosure_rate.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════
# Text summary
# ═══════════════════════════════════════════════

def print_text_summary(results):
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r.get("condition", "")][r.get("strategy", "")].append(
            r.get("summary", {}).get("leak_count", 0)
        )

    print("\nLeakage by Strategy × Condition:\n")
    header = f"{'Strategy':<22}" + "".join(f"{CONDITION_SHORT[c]:>8}" for c in CONDITIONS)
    print(header)
    print("-" * len(header))
    for s in STRATEGIES:
        row = f"{s:<22}"
        for c in CONDITIONS:
            vals = data[c].get(s, [0])
            row += f"{sum(vals)/len(vals):>8.1f}"
        print(row)

    print("\nCondition averages:")
    for c in CONDITIONS:
        all_vals = [v for s in STRATEGIES for v in data[c].get(s, [0])]
        print(f"  {CONDITION_LABELS[c]}: {sum(all_vals)/len(all_vals):.1f} avg leaks")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", default="charts")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results")
    print_text_summary(results)

    os.makedirs(args.output, exist_ok=True)
    print("\nGenerating charts...")
    generate_leakage_by_strategy(results, args.output)
    generate_heatmap(results, args.output)
    generate_per_turn_timeline(results, args.output)
    generate_radar(results, args.output)
    generate_disclosure_rate(results, args.output)

    print(f"\nDone! Open charts:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith(".html"):
            print(f"  open {os.path.join(args.output, f)}")


if __name__ == "__main__":
    main()
