"""
Chart Generator

Generates publication-quality charts from experiment results.
Uses the deterministic leak counts from results files (not GPT-4o-mini evals).

Usage:
    python3 generate_charts.py results/ --output charts/
"""

import json
import os
import sys
import glob
from collections import defaultdict


def load_all_results(results_dir: str) -> list[dict]:
    """Load all result JSON files."""
    pattern = os.path.join(results_dir, "**", "run_*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    
    results = []
    for filepath in files:
        with open(filepath) as f:
            result = json.load(f)
        # Add case name from path
        parts = os.path.relpath(filepath, results_dir).split(os.sep)
        if len(parts) >= 4:
            result["case_name"] = parts[0]
        results.append(result)
    
    return results


def generate_leakage_by_strategy_chart(results: list[dict], output_dir: str):
    """Generate bar chart: leakage rate by strategy across conditions."""
    
    conditions = [
        "naive_prompting", "structured_prompting", "self_monitoring",
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]
    condition_labels = {
        "naive_prompting": "Naive",
        "structured_prompting": "Structured",
        "self_monitoring": "Self-Mon",
        "isolated_architecture": "Isolated",
        "no_isolation_ablation": "No-Isol",
        "no_verifier_ablation": "No-Verif",
    }
    strategies = [
        "direct_questioning", "rephrasing", "emotional_appeal",
        "authority_claim", "gradual_escalation", "logical_inference",
    ]
    strategy_labels = {
        "direct_questioning": "Direct",
        "rephrasing": "Rephrase",
        "emotional_appeal": "Emotional",
        "authority_claim": "Authority",
        "gradual_escalation": "Gradual",
        "logical_inference": "Logical",
    }

    # Aggregate data
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        cond = r.get("condition", "")
        strat = r.get("strategy", "")
        leaks = r.get("summary", {}).get("leak_count", 0)
        data[cond][strat].append(leaks)

    # Build chart data as JSON for the HTML
    chart_data = []
    for strategy in strategies:
        row = {"strategy": strategy_labels[strategy]}
        for condition in conditions:
            values = data[condition].get(strategy, [0])
            row[condition_labels[condition]] = round(sum(values) / len(values), 1) if values else 0
        chart_data.append(row)

    # Also build condition summary
    condition_summary = {}
    for condition in conditions:
        all_leaks = []
        for strategy in strategies:
            all_leaks.extend(data[condition].get(strategy, [0]))
        condition_summary[condition_labels[condition]] = round(sum(all_leaks) / len(all_leaks), 1) if all_leaks else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Leakage Rate by Pressure Strategy</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
    body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        max-width: 1000px;
        margin: 40px auto;
        padding: 0 20px;
        background: #fff;
        color: #333;
    }}
    h1 {{
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .subtitle {{
        font-size: 13px;
        color: #666;
        margin-bottom: 24px;
    }}
    .chart-container {{
        position: relative;
        height: 400px;
        margin-bottom: 40px;
    }}
    .summary-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 20px;
    }}
    .summary-table th, .summary-table td {{
        padding: 8px 12px;
        text-align: right;
        border-bottom: 1px solid #eee;
    }}
    .summary-table th {{
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #333;
    }}
    .summary-table td:first-child {{
        text-align: left;
    }}
    .highlight {{
        background: #f0fdf4;
        font-weight: 600;
    }}
</style>
</head>
<body>
<h1>Leakage Rate by Pressure Strategy Across Conditions</h1>
<p class="subtitle">Average number of leaked withheld facts per 20-turn conversation. Lower is better.</p>

<div class="chart-container">
    <canvas id="leakChart"></canvas>
</div>

<h1>Summary by Condition</h1>
<p class="subtitle">Average leaks across all strategies and cases.</p>
<table class="summary-table">
    <tr>
        <th>Condition</th>
        <th>Avg Leaks</th>
        <th>Interpretation</th>
    </tr>
    {"".join(f'''<tr class="{'highlight' if 'Isolated' in label else ''}">
        <td>{label}</td>
        <td>{val}</td>
        <td>{
            'Near-zero leakage — architecture works' if val < 1 else
            'Moderate leakage — isolation broken' if val < 3 else
            'High leakage — baseline fails under pressure'
        }</td>
    </tr>''' for label, val in condition_summary.items())}
</table>

<script>
const data = {json.dumps(chart_data)};
const conditions = {json.dumps([condition_labels[c] for c in conditions])};

const colors = {{
    'Naive': '#ef4444',
    'Structured': '#f97316',
    'Self-Mon': '#eab308',
    'Isolated': '#22c55e',
    'No-Isol': '#3b82f6',
    'No-Verif': '#8b5cf6',
}};

const datasets = conditions.map(cond => ({{
    label: cond,
    data: data.map(d => d[cond]),
    backgroundColor: colors[cond] + '99',
    borderColor: colors[cond],
    borderWidth: 1,
}}));

new Chart(document.getElementById('leakChart'), {{
    type: 'bar',
    data: {{
        labels: data.map(d => d.strategy),
        datasets: datasets,
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{
                position: 'top',
                labels: {{ font: {{ size: 11 }} }}
            }},
        }},
        scales: {{
            y: {{
                beginAtZero: true,
                title: {{
                    display: true,
                    text: 'Average Leaks per Conversation',
                    font: {{ size: 12 }}
                }}
            }},
            x: {{
                title: {{
                    display: true,
                    text: 'Pressure Strategy',
                    font: {{ size: 12 }}
                }}
            }}
        }}
    }}
}});
</script>
</body>
</html>"""

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "leakage_by_strategy.html")
    with open(filepath, "w") as f:
        f.write(html)
    print(f"  Saved: {filepath}")


def generate_condition_comparison_chart(results: list[dict], output_dir: str):
    """Generate bar chart: overall leakage comparison across conditions, split by case."""
    
    conditions = [
        "naive_prompting", "structured_prompting", "self_monitoring",
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]
    condition_labels = {
        "naive_prompting": "Naive\nPrompting",
        "structured_prompting": "Structured\nPrompting",
        "self_monitoring": "Self-\nMonitoring",
        "isolated_architecture": "Isolated\nArchitecture",
        "no_isolation_ablation": "No-Isolation\nAblation",
        "no_verifier_ablation": "No-Verifier\nAblation",
    }
    
    cases = ["cardiology", "respiratory", "gi"]
    case_labels = {"cardiology": "Cardiology", "respiratory": "Respiratory", "gi": "GI"}

    # Aggregate
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        cond = r.get("condition", "")
        case = r.get("case_name", "")
        leaks = r.get("summary", {}).get("leak_count", 0)
        data[cond][case].append(leaks)

    chart_data = []
    for condition in conditions:
        row = {"condition": condition_labels[condition].replace("\n", " ")}
        for case in cases:
            values = data[condition].get(case, [0])
            row[case_labels[case]] = round(sum(values) / len(values), 1) if values else 0
        chart_data.append(row)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Leakage by Condition and Case</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
    body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        max-width: 900px;
        margin: 40px auto;
        padding: 0 20px;
        background: #fff;
        color: #333;
    }}
    h1 {{ font-size: 18px; font-weight: 600; margin-bottom: 4px; }}
    .subtitle {{ font-size: 13px; color: #666; margin-bottom: 24px; }}
    .chart-container {{ position: relative; height: 400px; }}
</style>
</head>
<body>
<h1>Average Leakage by Condition Across Clinical Cases</h1>
<p class="subtitle">Grouped by condition, colored by case specialty. Lower is better.</p>

<div class="chart-container">
    <canvas id="caseChart"></canvas>
</div>

<script>
const data = {json.dumps(chart_data)};
const cases = ['Cardiology', 'Respiratory', 'GI'];
const caseColors = {{ 'Cardiology': '#3b82f6', 'Respiratory': '#22c55e', 'GI': '#f97316' }};

const datasets = cases.map(c => ({{
    label: c,
    data: data.map(d => d[c]),
    backgroundColor: caseColors[c] + '99',
    borderColor: caseColors[c],
    borderWidth: 1,
}}));

new Chart(document.getElementById('caseChart'), {{
    type: 'bar',
    data: {{
        labels: data.map(d => d.condition),
        datasets: datasets,
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }}
        }},
        scales: {{
            y: {{
                beginAtZero: true,
                title: {{ display: true, text: 'Average Leaks per Conversation', font: {{ size: 12 }} }}
            }}
        }}
    }}
}});
</script>
</body>
</html>"""

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "leakage_by_condition_and_case.html")
    with open(filepath, "w") as f:
        f.write(html)
    print(f"  Saved: {filepath}")


def generate_disclosure_rate_chart(results: list[dict], output_dir: str):
    """Generate chart showing disclosure rates for architecture conditions."""
    
    arch_conditions = [
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]
    condition_labels = {
        "isolated_architecture": "Isolated",
        "no_isolation_ablation": "No-Isolation",
        "no_verifier_ablation": "No-Verifier",
    }
    strategies = [
        "direct_questioning", "rephrasing", "emotional_appeal",
        "authority_claim", "gradual_escalation", "logical_inference",
    ]
    strategy_labels = {
        "direct_questioning": "Direct",
        "rephrasing": "Rephrase",
        "emotional_appeal": "Emotional",
        "authority_claim": "Authority",
        "gradual_escalation": "Gradual",
        "logical_inference": "Logical",
    }

    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        cond = r.get("condition", "")
        strat = r.get("strategy", "")
        if cond not in arch_conditions:
            continue
        state = r.get("summary", {}).get("condition_state", {})
        unlocked = len(state.get("unlocked_fact_ids", []))
        # Get total withheld from case
        case_id = r.get("case_id", "")
        total_withheld = {"CARDIO-001": 8, "RESP-001": 7, "GI-001": 8}.get(case_id, 8)
        rate = unlocked / total_withheld if total_withheld else 0
        data[cond][strat].append(rate)

    chart_data = []
    for strategy in strategies:
        row = {"strategy": strategy_labels[strategy]}
        for condition in arch_conditions:
            values = data[condition].get(strategy, [0])
            row[condition_labels[condition]] = round(sum(values) / len(values) * 100) if values else 0
        chart_data.append(row)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Disclosure Rate</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
    body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        max-width: 900px;
        margin: 40px auto;
        padding: 0 20px;
        background: #fff;
        color: #333;
    }}
    h1 {{ font-size: 18px; font-weight: 600; margin-bottom: 4px; }}
    .subtitle {{ font-size: 13px; color: #666; margin-bottom: 24px; }}
    .chart-container {{ position: relative; height: 400px; }}
</style>
</head>
<body>
<h1>Disclosure Rate: Earned Facts Successfully Disclosed</h1>
<p class="subtitle">Percentage of withheld facts unlocked through proper student questioning. Higher is better — shows the architecture is not over-withholding.</p>

<div class="chart-container">
    <canvas id="disclosureChart"></canvas>
</div>

<script>
const data = {json.dumps(chart_data)};
const conditions = ['Isolated', 'No-Isolation', 'No-Verifier'];
const colors = {{ 'Isolated': '#22c55e', 'No-Isolation': '#3b82f6', 'No-Verifier': '#8b5cf6' }};

const datasets = conditions.map(c => ({{
    label: c,
    data: data.map(d => d[c]),
    backgroundColor: colors[c] + '99',
    borderColor: colors[c],
    borderWidth: 1,
}}));

new Chart(document.getElementById('disclosureChart'), {{
    type: 'bar',
    data: {{
        labels: data.map(d => d.strategy),
        datasets: datasets,
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }}
        }},
        scales: {{
            y: {{
                beginAtZero: true,
                max: 100,
                title: {{ display: true, text: 'Disclosure Rate (%)', font: {{ size: 12 }} }}
            }}
        }}
    }}
}});
</script>
</body>
</html>"""

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "disclosure_rate.html")
    with open(filepath, "w") as f:
        f.write(html)
    print(f"  Saved: {filepath}")


def print_text_summary(results: list[dict]):
    """Print a text summary table."""
    conditions = [
        "naive_prompting", "structured_prompting", "self_monitoring",
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]
    strategies = [
        "direct_questioning", "rephrasing", "emotional_appeal",
        "authority_claim", "gradual_escalation", "logical_inference",
    ]
    short = {
        "naive_prompting": "naive", "structured_prompting": "struct",
        "self_monitoring": "self_m", "isolated_architecture": "isolat",
        "no_isolation_ablation": "no_iso", "no_verifier_ablation": "no_ver",
    }

    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        cond = r.get("condition", "")
        strat = r.get("strategy", "")
        leaks = r.get("summary", {}).get("leak_count", 0)
        data[cond][strat].append(leaks)

    print("\nLeakage by Strategy × Condition (deterministic counts):\n")
    header = f"{'Strategy':<22}" + "".join(f"{short[c]:>8}" for c in conditions)
    print(header)
    print("-" * len(header))
    for strat in strategies:
        row = f"{strat:<22}"
        for cond in conditions:
            vals = data[cond].get(strat, [0])
            avg = sum(vals) / len(vals) if vals else 0
            row += f"{avg:>8.1f}"
        print(row)

    # Condition averages
    print()
    print("Condition averages:")
    for cond in conditions:
        all_vals = []
        for strat in strategies:
            all_vals.extend(data[cond].get(strat, [0]))
        avg = sum(all_vals) / len(all_vals) if all_vals else 0
        print(f"  {cond}: {avg:.1f} avg leaks")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate charts from experiment results")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--output", default="charts", help="Output directory for charts")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results\n")
    print_text_summary(results)

    print("\nGenerating charts...")
    generate_leakage_by_strategy_chart(results, args.output)
    generate_condition_comparison_chart(results, args.output)
    generate_disclosure_rate_chart(results, args.output)

    print(f"\nDone! Open charts in your browser:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith(".html"):
            print(f"  open {os.path.join(args.output, f)}")


if __name__ == "__main__":
    main()
