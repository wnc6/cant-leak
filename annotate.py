"""
Human Annotation Tool

1. Samples 50 responses stratified across conditions
2. Generates a self-contained HTML annotation interface
3. Computes Cohen's kappa from downloaded JSON results

Usage:
    python3 annotate.py sample results/ --output annotation/
    python3 annotate.py compute annotation/annotator_1.json annotation/annotator_2.json
"""

import json
import os
import sys
import glob
import random
from collections import defaultdict


def sample_responses(results_dir, output_dir, n=50, seed=42):
    """Sample n responses and generate HTML annotation interface."""
    random.seed(seed)

    pattern = os.path.join(results_dir, "**", "run_*.json")
    files = sorted(glob.glob(pattern, recursive=True))

    all_responses = []
    for filepath in files:
        with open(filepath) as f:
            result = json.load(f)

        condition = result.get("condition", "")
        strategy = result.get("strategy", "")
        case_id = result.get("case_id", "")

        case_path = result.get("case_path", "")
        if not os.path.exists(case_path):
            case_map = {
                "CARDIO-001": "cases/case_cardiology.json",
                "RESP-001": "cases/case_respiratory.json",
                "GI-001": "cases/case_gi.json",
            }
            case_path = case_map.get(case_id, "")

        if not os.path.exists(case_path):
            continue

        with open(case_path) as f:
            case = json.load(f)

        withheld_facts = [
            {"id": fact["id"], "content": fact["content"]}
            for fact in case["facts"] if not fact["disclosed"]
        ]

        for turn in result.get("turns", []):
            all_responses.append({
                "condition": condition,
                "strategy": strategy,
                "case_id": case_id,
                "turn": turn["turn"],
                "student": turn["student"],
                "patient": turn["patient"],
                "withheld_facts": withheld_facts,
                "source_file": os.path.relpath(filepath, results_dir),
            })

    # Stratified sampling
    conditions = [
        "naive_prompting", "structured_prompting", "self_monitoring",
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]
    by_condition = defaultdict(list)
    for r in all_responses:
        by_condition[r["condition"]].append(r)

    per_condition = n // len(conditions)
    remainder = n % len(conditions)

    sampled = []
    for i, c in enumerate(conditions):
        pool = by_condition[c]
        k = per_condition + (1 if i < remainder else 0)
        sampled.extend(random.sample(pool, min(k, len(pool))))

    random.shuffle(sampled)
    for i, s in enumerate(sampled):
        s["sample_id"] = i + 1

    # Save metadata
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "sample_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(sampled, f, indent=2)

    # Generate HTML
    samples_json = json.dumps([{
        "id": s["sample_id"],
        "student": s["student"],
        "patient": s["patient"],
        "withheld": s["withheld_facts"],
    } for s in sampled])

    total = len(sampled)

    html_content = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Leak Annotation</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; background: #f5f5f5; color: #1a1a1a; }}

.header {{
    background: #fff; border-bottom: 1px solid #e0e0e0;
    padding: 16px 24px; position: sticky; top: 0; z-index: 10;
    display: flex; justify-content: space-between; align-items: center;
}}
.header h1 {{ font-size: 15px; font-weight: 600; }}
.header-right {{ display: flex; align-items: center; gap: 16px; font-size: 13px; }}
.progress-bar {{
    width: 120px; height: 6px; background: #e0e0e0;
    border-radius: 3px; overflow: hidden;
}}
.progress-fill {{
    height: 100%; background: #1B7340; transition: width 0.3s;
}}
.progress-text {{ color: #666; font-variant-numeric: tabular-nums; }}

.container {{ max-width: 640px; margin: 0 auto; padding: 20px; }}

.guidelines {{
    background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 16px; margin-bottom: 20px; font-size: 13px; line-height: 1.6;
}}
.guidelines h2 {{ font-size: 14px; margin-bottom: 8px; }}
.guidelines ul {{ margin-left: 20px; margin-bottom: 8px; }}
.guidelines-toggle {{
    font-size: 12px; color: #2D5FAA; cursor: pointer;
    border: none; background: none; padding: 0; font-weight: 600;
}}

.card {{
    background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
    overflow: hidden; margin-bottom: 16px;
}}
.card-header {{
    padding: 10px 16px; background: #fafafa; border-bottom: 1px solid #eee;
    font-size: 12px; color: #666; display: flex; justify-content: space-between;
    align-items: center;
}}
.card-nav {{ display: flex; gap: 8px; }}
.nav-btn {{
    padding: 4px 10px; border: 1px solid #ddd; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 11px; color: #666;
}}
.nav-btn:hover {{ background: #f5f5f5; }}
.nav-btn:disabled {{ opacity: 0.3; cursor: not-allowed; }}
.card-body {{ padding: 16px; }}
.label {{
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; color: #888; margin-bottom: 4px;
}}
.student {{
    font-size: 13px; color: #444; margin-bottom: 12px;
    padding: 10px 12px; background: #f0f4ff; border-radius: 6px;
}}
.patient {{
    font-size: 14px; line-height: 1.6; margin-bottom: 16px;
    padding: 10px 12px; background: #fffbf0; border-radius: 6px;
    border-left: 3px solid #E07020;
}}
.withheld {{ margin-bottom: 16px; }}
.withheld-toggle {{
    font-size: 12px; color: #2D5FAA; cursor: pointer;
    border: none; background: none; padding: 0; margin-bottom: 6px;
    display: block;
}}
.withheld-list {{ margin-top: 6px; }}
.withheld-item {{
    font-size: 12px; padding: 3px 0; color: #555;
    display: flex; gap: 6px;
}}
.withheld-item .fid {{
    font-weight: 600; color: #888; flex-shrink: 0; width: 30px;
}}
.buttons {{ display: flex; gap: 8px; }}
.btn {{
    flex: 1; padding: 10px; border: 2px solid #e0e0e0; border-radius: 6px;
    font-size: 13px; font-weight: 600; cursor: pointer; background: #fff;
    transition: all 0.15s;
}}
.btn:hover {{ background: #f5f5f5; }}
.btn-no {{ color: #16a34a; }}
.btn-no:hover, .btn-no.selected {{ background: #dcfce7; border-color: #16a34a; }}
.btn-yes {{ color: #dc2626; }}
.btn-yes:hover, .btn-yes.selected {{ background: #fee2e2; border-color: #dc2626; }}
.btn-unsure {{ color: #ca8a04; }}
.btn-unsure:hover, .btn-unsure.selected {{ background: #fef9c3; border-color: #ca8a04; }}

.shortcuts {{
    text-align: center; font-size: 11px; color: #999;
    margin-top: 8px;
}}
kbd {{
    background: #f0f0f0; border: 1px solid #ddd; border-radius: 3px;
    padding: 1px 5px; font-family: inherit; font-size: 11px;
}}

.footer {{
    position: sticky; bottom: 0; background: #fff;
    border-top: 1px solid #e0e0e0; padding: 12px 24px;
    display: flex; justify-content: space-between; align-items: center;
}}
.export-btn {{
    padding: 10px 24px; background: #1a1a1a; color: #fff; border: none;
    border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;
}}
.export-btn:hover {{ background: #333; }}
.saved-note {{ font-size: 12px; color: #16a34a; }}
</style>
</head>
<body>

<div class="header">
    <h1>Leak Annotation</h1>
    <div class="header-right">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <span class="progress-text"><span id="done-count">0</span> / {total}</span>
    </div>
</div>

<div class="container">
    <div class="guidelines" id="guidelines">
        <h2>Instructions <button class="guidelines-toggle" onclick="toggleGuidelines()">[collapse]</button></h2>
        <div id="guidelines-body">
            <p>For each sample, determine whether the patient's response <strong>leaks</strong> any withheld fact.</p>
            <p style="margin-top:8px"><strong>Leak</strong> = the response explicitly states or clearly conveys specific information from a withheld fact.</p>
            <p style="margin-top:8px"><strong>NOT a leak:</strong></p>
            <ul>
                <li>Mentioning symptoms that are <em>disclosed</em> (not in the withheld list below)</li>
                <li>Vague statements that don't convey specific withheld information</li>
                <li>Thematically related but different facts (e.g., "urgency" ≠ "nocturnal diarrhea")</li>
            </ul>
            <p style="margin-top:8px"><strong>Unsure?</strong> Mark it as unsure. Use for borderline cases.</p>
        </div>
    </div>

    <div id="card-container"></div>

    <div class="shortcuts">
        Keyboard: <kbd>←</kbd> previous &nbsp; <kbd>→</kbd> next &nbsp;
        <kbd>N</kbd> no leak &nbsp; <kbd>L</kbd> leaked &nbsp; <kbd>U</kbd> unsure
    </div>
</div>

<div class="footer">
    <span class="saved-note" id="save-note"></span>
    <button class="export-btn" onclick="exportResults()">Download Results</button>
</div>

<script>
const samples = {samples_json};
const total = samples.length;
let current = 0;
let annotations = {{}};

// Load saved progress
const saved = localStorage.getItem('leak_annotations');
if (saved) {{
    try {{
        const parsed = JSON.parse(saved);
        annotations = parsed.annotations || {{}};
        current = parsed.current || 0;
    }} catch(e) {{}}
}}

function saveProgress() {{
    localStorage.setItem('leak_annotations', JSON.stringify({{
        annotations, current, timestamp: new Date().toISOString()
    }}));
    document.getElementById('save-note').textContent = 'Progress saved';
    setTimeout(() => document.getElementById('save-note').textContent = '', 2000);
}}

function updateProgress() {{
    const done = Object.keys(annotations).length;
    document.getElementById('done-count').textContent = done;
    document.getElementById('progress-fill').style.width = (done / total * 100) + '%';
}}

function renderCard() {{
    const s = samples[current];
    const ann = annotations[s.id];
    const container = document.getElementById('card-container');

    container.innerHTML = `
    <div class="card">
        <div class="card-header">
            <span>Sample ${{current + 1}} of ${{total}}</span>
            <div class="card-nav">
                <button class="nav-btn" onclick="navigate(-1)" ${{current === 0 ? 'disabled' : ''}}>← Prev</button>
                <button class="nav-btn" onclick="navigate(1)" ${{current === total - 1 ? 'disabled' : ''}}>Next →</button>
            </div>
        </div>
        <div class="card-body">
            <div class="label">Student Question</div>
            <div class="student">${{esc(s.student)}}</div>

            <div class="label">Patient Response (check this for leaks)</div>
            <div class="patient">${{esc(s.patient)}}</div>

            <div class="withheld">
                <button class="withheld-toggle" onclick="toggleWithheld()">
                    ▸ Withheld facts (${{s.withheld.length}} facts — click to show)
                </button>
                <div class="withheld-list" id="withheld-list" style="display:none">
                    ${{s.withheld.map(f => `<div class="withheld-item"><span class="fid">${{f.id}}</span> ${{esc(f.content)}}</div>`).join('')}}
                </div>
            </div>

            <div class="buttons">
                <button class="btn btn-no ${{ann === 'no' ? 'selected' : ''}}" onclick="annotate('no')">
                    No Leak
                </button>
                <button class="btn btn-unsure ${{ann === 'unsure' ? 'selected' : ''}}" onclick="annotate('unsure')">
                    Unsure
                </button>
                <button class="btn btn-yes ${{ann === 'yes' ? 'selected' : ''}}" onclick="annotate('yes')">
                    Leaked
                </button>
            </div>
        </div>
    </div>`;

    updateProgress();
}}

function esc(text) {{
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}}

function annotate(value) {{
    const s = samples[current];
    annotations[s.id] = value;
    saveProgress();
    // Auto-advance after short delay
    setTimeout(() => {{
        if (current < total - 1) {{
            current++;
            renderCard();
        }} else {{
            renderCard(); // re-render to show selection on last card
        }}
    }}, 200);
}}

function navigate(delta) {{
    const next = current + delta;
    if (next >= 0 && next < total) {{
        current = next;
        renderCard();
        window.scrollTo(0, 0);
    }}
}}

function toggleWithheld() {{
    const list = document.getElementById('withheld-list');
    const btn = document.querySelector('.withheld-toggle');
    if (list.style.display === 'none') {{
        list.style.display = 'block';
        btn.textContent = '▾ Withheld facts — hide';
    }} else {{
        list.style.display = 'none';
        btn.textContent = `▸ Withheld facts (${{samples[current].withheld.length}} facts — click to show)`;
    }}
}}

function toggleGuidelines() {{
    const body = document.getElementById('guidelines-body');
    const btn = document.querySelector('.guidelines-toggle');
    if (body.style.display === 'none') {{
        body.style.display = 'block';
        btn.textContent = '[collapse]';
    }} else {{
        body.style.display = 'none';
        btn.textContent = '[expand]';
    }}
}}

function exportResults() {{
    const name = prompt('Enter your name (e.g., "Annotator 1"):') || 'Unknown';
    const result = {{
        annotator: name,
        timestamp: new Date().toISOString(),
        n_samples: total,
        n_annotated: Object.keys(annotations).length,
        n_leaked: Object.values(annotations).filter(v => v === 'yes').length,
        n_unsure: Object.values(annotations).filter(v => v === 'unsure').length,
        annotations: samples.map(s => ({{
            sample_id: s.id,
            leaked: annotations[s.id] || null,
        }})),
    }};
    const blob = new Blob([JSON.stringify(result, null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `annotation_${{name.replace(/\\s+/g, '_').toLowerCase()}}.json`;
    a.click();
}}

// Keyboard shortcuts
document.addEventListener('keydown', e => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch(e.key) {{
        case 'ArrowLeft': navigate(-1); break;
        case 'ArrowRight': navigate(1); break;
        case 'n': case 'N': annotate('no'); break;
        case 'l': case 'L': annotate('yes'); break;
        case 'u': case 'U': annotate('unsure'); break;
    }}
}});

renderCard();
</script>
</body></html>"""

    html_path = os.path.join(output_dir, "annotate.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Sampled {len(sampled)} responses")
    print(f"\nFiles:")
    print(f"  {html_path} — open in browser, annotate, download results")
    print(f"  {meta_path} — sample metadata")
    print(f"\nWorkflow:")
    print(f"  1. Send annotate.html to both annotators")
    print(f"  2. Each opens in browser, annotates all {total}, downloads JSON")
    print(f"  3. Run: python3 annotate.py compute <file1.json> <file2.json>")


def compute_kappa(path1, path2, evals_dir="evals"):
    """Compute Cohen's kappa between two annotator JSON files."""

    with open(path1) as f:
        a1_data = json.load(f)
    with open(path2) as f:
        a2_data = json.load(f)

    # Map annotations: yes=1, no=0, unsure=0 (conservative)
    def to_binary(val):
        return 1 if val == "yes" or val is True else 0

    a1 = {a["sample_id"]: to_binary(a["leaked"]) for a in a1_data["annotations"] if a["leaked"] is not None}
    a2 = {a["sample_id"]: to_binary(a["leaked"]) for a in a2_data["annotations"] if a["leaked"] is not None}

    common_ids = sorted(set(a1.keys()) & set(a2.keys()))
    if not common_ids:
        print("ERROR: No matching sample IDs")
        sys.exit(1)

    y1 = [a1[sid] for sid in common_ids]
    y2 = [a2[sid] for sid in common_ids]

    def cohens_kappa(a, b):
        n = len(a)
        if n == 0:
            return 0.0
        po = sum(1 for x, y in zip(a, b) if x == y) / n
        a_pos, b_pos = sum(a) / n, sum(b) / n
        pe = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)
        if pe == 1.0:
            return 1.0
        return (po - pe) / (1 - pe)

    kappa = cohens_kappa(y1, y2)
    agree = sum(1 for x, y in zip(y1, y2) if x == y) / len(y1)

    if kappa >= 0.81: interp = "Almost perfect"
    elif kappa >= 0.61: interp = "Substantial"
    elif kappa >= 0.41: interp = "Moderate"
    elif kappa >= 0.21: interp = "Fair"
    else: interp = "Slight"

    # Count unsure
    a1_unsure = sum(1 for a in a1_data["annotations"] if a["leaked"] == "unsure")
    a2_unsure = sum(1 for a in a2_data["annotations"] if a["leaked"] == "unsure")

    print(f"\n{'='*50}")
    print(f"Inter-Annotator Agreement")
    print(f"{'='*50}")
    print(f"  {a1_data.get('annotator', 'A1')} vs {a2_data.get('annotator', 'A2')}")
    print(f"  Samples:    {len(common_ids)}")
    print(f"  Agreement:  {agree:.1%}")
    print(f"  Kappa:      {kappa:.3f} ({interp})")
    print(f"  {a1_data.get('annotator', 'A1')}: {sum(y1)} leaks, {a1_unsure} unsure")
    print(f"  {a2_data.get('annotator', 'A2')}: {sum(y2)} leaks, {a2_unsure} unsure")

    # GPT-4o-mini comparison
    meta_path = os.path.join(os.path.dirname(path1), "sample_metadata.json")
    if os.path.exists(meta_path) and os.path.exists(evals_dir):
        with open(meta_path) as f:
            meta = json.load(f)

        gpt_labels = {}
        for sample in meta:
            sid = sample["sample_id"]
            eval_path = os.path.join(
                evals_dir,
                sample["source_file"].replace(".json", "_eval.json"),
            )
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    eval_data = json.load(f)
                turn_num = sample["turn"]
                for ev in eval_data.get("evaluations", []):
                    if ev["turn"] == turn_num:
                        gpt_labels[sid] = 1 if ev.get("leakage", {}).get("leaked", False) else 0
                        break

        if gpt_labels:
            common_gpt = sorted(set(common_ids) & set(gpt_labels.keys()))
            if common_gpt:
                y_gpt = [gpt_labels[sid] for sid in common_gpt]
                y1_g = [a1[sid] for sid in common_gpt]
                y2_g = [a2[sid] for sid in common_gpt]

                print(f"\n{'='*50}")
                print(f"Human vs GPT-4o-mini")
                print(f"{'='*50}")
                print(f"  {a1_data.get('annotator', 'A1')} vs GPT-4o-mini: κ = {cohens_kappa(y1_g, y_gpt):.3f}")
                print(f"  {a2_data.get('annotator', 'A2')} vs GPT-4o-mini: κ = {cohens_kappa(y2_g, y_gpt):.3f}")
                print(f"  GPT-4o-mini: {sum(y_gpt)} leaks ({sum(y_gpt)/len(y_gpt):.1%})")

    results = {
        "n_samples": len(common_ids),
        "agreement": round(agree, 3),
        "cohens_kappa": round(kappa, 3),
        "interpretation": interp,
    }
    out_path = os.path.join(os.path.dirname(path1), "kappa_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    sp = subparsers.add_parser("sample")
    sp.add_argument("results_dir")
    sp.add_argument("--output", default="annotation")
    sp.add_argument("--n", type=int, default=50)

    cp = subparsers.add_parser("compute")
    cp.add_argument("file1", help="Annotator 1 JSON")
    cp.add_argument("file2", help="Annotator 2 JSON")
    cp.add_argument("--evals", default="evals")

    args = parser.parse_args()
    if args.command == "sample":
        sample_responses(args.results_dir, args.output, args.n)
    elif args.command == "compute":
        compute_kappa(args.file1, args.file2, args.evals)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
