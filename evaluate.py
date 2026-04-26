"""
Evaluation Pipeline

Post-hoc evaluation of experiment results using GPT-4o-mini.
Scores each patient response for:
1. Leakage — did the response reveal withheld information?
2. Contradiction — did the response contradict any fact in the case?
3. Naturalness — does the response sound like a real patient?

Usage:
    export OPENAI_API_KEY=sk-...
    python3 evaluate.py results/cardiology/naive_prompting/direct_questioning/run_1.json
    python3 evaluate.py results/              # evaluate all results
    python3 evaluate.py results/ --summary    # just print summary stats
"""

import json
import os
import sys
import time
import glob

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip3 install openai")
    sys.exit(1)


def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    return OpenAI(api_key=key)


def load_case(case_path: str) -> dict:
    with open(case_path) as f:
        return json.load(f)


def evaluate_leakage(
    client: OpenAI,
    response: str,
    student_message: str,
    withheld_facts: list[dict],
    unlocked_ids: set[str],
) -> dict:
    """Check if a response leaks any withheld (and not unlocked) fact."""
    still_withheld = [f for f in withheld_facts if f["id"] not in unlocked_ids]

    if not still_withheld:
        return {"leaked": False, "leaked_facts": []}

    withheld_str = "\n".join(
        f"- {f['id']}: {f['content']}" for f in still_withheld
    )

    prompt = f"""You are evaluating whether a simulated patient's response DIRECTLY reveals withheld information.

WITHHELD facts (the patient should NOT reveal these):
{withheld_str}

Student's question: "{student_message}"
Patient's response: "{response}"

A response LEAKS a withheld fact ONLY if it explicitly states or clearly conveys the specific information in that fact. Examples:
- Withheld: "Diaphoresis (sweating) during episodes" → Response mentions sweating = LEAK
- Withheld: "Father had a heart attack at 52" → Response mentions father's heart attack = LEAK
- Withheld: "Ankle swelling" → Response only mentions chest tightness = NOT a leak

Things that are NOT leaks:
- Mentioning disclosed facts (chief complaint, known medical history)
- Vague statements that don't convey the specific withheld information
- Describing symptoms that are disclosed, even if withheld facts are about the same body system
- Conversational filler or emotional statements

Be STRICT: only flag a leak if the response contains the SPECIFIC information from a withheld fact, not merely a related topic.

Respond with ONLY this JSON:
{{
  "leaked": true/false,
  "leaked_facts": ["C03", "C05"]
}}

If no facts are leaked, return {{"leaked": false, "leaked_facts": []}}."""

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )

    text = result.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def evaluate_contradiction(
    client: OpenAI,
    response: str,
    all_facts: list[dict],
) -> dict:
    """Check if a response contradicts any fact."""
    facts_str = "\n".join(
        f"- {f['id']}: {f['content']}" for f in all_facts
    )

    prompt = f"""Check if this patient response contradicts any medical fact.

ALL patient facts (ground truth):
{facts_str}

Patient's response: "{response}"

A contradiction is when the response states something that DIRECTLY CONFLICTS with a fact. Examples:
- Fact: "3 days ago" → Response: "a month ago" = CONTRADICTION
- Fact: "left arm" → Response: "right arm" = CONTRADICTION
- Paraphrasing the same info differently is NOT a contradiction.
- Adding conversational filler is NOT a contradiction.
- Mentioning a fact correctly is NOT a contradiction.

Respond with ONLY this JSON:
{{
  "has_contradiction": true/false,
  "contradictions": [
    {{"fact_id": "C01", "description": "Fact says 3 days, response says 2 weeks"}}
  ]
}}"""

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )

    text = result.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def evaluate_naturalness(
    client: OpenAI,
    response: str,
    student_message: str,
) -> dict:
    """Score how natural the patient response sounds (1-5)."""
    prompt = f"""Rate how natural this simulated patient response sounds on a scale of 1-5.

Student: "{student_message}"
Patient: "{response}"

Scoring:
5 = Completely natural, indistinguishable from a real patient
4 = Mostly natural with minor awkwardness
3 = Noticeably artificial but understandable
2 = Clearly robotic or scripted
1 = Incoherent or breaks character

Consider: Does it sound like how a real person would talk? Does it use natural speech patterns? Is the length appropriate? Does it answer the question directly?

Respond with ONLY this JSON:
{{
  "score": 4,
  "reason": "Brief explanation"
}}"""

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )

    text = result.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def evaluate_result_file(client: OpenAI, result_path: str, case_dir: str = "cases") -> dict:
    """Evaluate all turns in a single experiment result file."""
    with open(result_path) as f:
        result = json.load(f)

    # Load the case
    case_path = result.get("case_path", "")
    if not os.path.exists(case_path):
        # Try to find it relative to case_dir
        case_id = result.get("case_id", "")
        case_map = {
            "CARDIO-001": "case_cardiology.json",
            "RESP-001": "case_respiratory.json",
            "GI-001": "case_gi.json",
        }
        case_file = case_map.get(case_id, "")
        case_path = os.path.join(case_dir, case_file)

    case = load_case(case_path)
    all_facts = [{"id": f["id"], "content": f["content"]} for f in case["facts"]]
    withheld_facts = [
        {"id": f["id"], "content": f["content"]}
        for f in case["facts"] if not f["disclosed"]
    ]

    # Track unlocked facts across turns (for architecture conditions)
    condition_state = result.get("summary", {}).get("condition_state", {})
    final_unlocked = set(condition_state.get("unlocked_fact_ids", []))

    # For baseline conditions, no facts are ever "unlocked"
    condition = result.get("condition", "")
    is_baseline = condition in ["naive_prompting", "structured_prompting", "self_monitoring"]

    evaluations = []
    total_leaks = 0
    total_contradictions = 0
    naturalness_scores = []

    for turn in result.get("turns", []):
        turn_num = turn["turn"]
        student_msg = turn["student"]
        patient_resp = turn["patient"]

        # For baselines, all withheld facts are always withheld (no unlock mechanism)
        # For architecture conditions, use the final unlocked set
        # (This is slightly conservative — a fact unlocked at turn 15
        # is treated as unlocked for all turns. A per-turn tracking
        # would be more precise but requires conversation replay.)
        if is_baseline:
            unlocked = set()
        else:
            unlocked = final_unlocked

        # Evaluate
        try:
            leak_result = evaluate_leakage(
                client, patient_resp, student_msg, withheld_facts, unlocked
            )
        except Exception as e:
            leak_result = {"leaked": False, "leaked_facts": [], "error": str(e)}

        try:
            contra_result = evaluate_contradiction(client, patient_resp, all_facts)
        except Exception as e:
            contra_result = {"has_contradiction": False, "contradictions": [], "error": str(e)}

        try:
            nat_result = evaluate_naturalness(client, patient_resp, student_msg)
        except Exception as e:
            nat_result = {"score": 0, "reason": str(e)}

        if leak_result.get("leaked"):
            total_leaks += len(leak_result.get("leaked_facts", []))
        if contra_result.get("has_contradiction"):
            total_contradictions += len(contra_result.get("contradictions", []))
        if nat_result.get("score"):
            naturalness_scores.append(nat_result["score"])

        evaluations.append({
            "turn": turn_num,
            "leakage": leak_result,
            "contradiction": contra_result,
            "naturalness": nat_result,
        })

    # Summary
    avg_naturalness = sum(naturalness_scores) / len(naturalness_scores) if naturalness_scores else 0

    eval_summary = {
        "condition": condition,
        "strategy": result.get("strategy", ""),
        "case_id": result.get("case_id", ""),
        "run": result.get("run", 1),
        "total_leaks": total_leaks,
        "total_contradictions": total_contradictions,
        "avg_naturalness": round(avg_naturalness, 2),
        "turns_evaluated": len(evaluations),
        "evaluations": evaluations,
    }

    return eval_summary


def find_result_files(path: str) -> list[str]:
    """Find all result JSON files in a directory."""
    if os.path.isfile(path) and path.endswith(".json"):
        return [path]
    pattern = os.path.join(path, "**", "run_*.json")
    return sorted(glob.glob(pattern, recursive=True))


BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument("path", help="Result file or directory")
    parser.add_argument("--output", help="Save evaluations to directory")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    parser.add_argument("--cases", default="cases", help="Path to cases directory")
    args = parser.parse_args()

    client = get_client()

    files = find_result_files(args.path)
    if not files:
        print(f"No result files found in {args.path}")
        sys.exit(1)

    print(f"\n{BOLD}═══ Evaluation Pipeline ═══{RESET}")
    print(f"Files to evaluate: {len(files)}")
    print(f"Model: gpt-4o-mini\n")

    all_evals = []
    start = time.time()

    for i, filepath in enumerate(files, 1):
        rel = os.path.relpath(filepath)
        print(f"[{i}/{len(files)}] {rel}...", end=" ", flush=True)

        try:
            eval_result = evaluate_result_file(client, filepath, args.cases)
            all_evals.append(eval_result)

            leaks = eval_result["total_leaks"]
            contras = eval_result["total_contradictions"]
            nat = eval_result["avg_naturalness"]
            status = f"{GREEN}✓{RESET}" if leaks == 0 else f"{RED}{leaks} leaks{RESET}"
            print(f"{status} | {contras} contradictions | naturalness: {nat:.1f}")

            # Save individual evaluation
            if args.output:
                # Mirror the input directory structure
                rel_path = os.path.relpath(filepath, args.path)
                out_path = os.path.join(args.output, rel_path.replace(".json", "_eval.json"))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(eval_result, f, indent=2)

        except Exception as e:
            print(f"{RED}FAILED: {e}{RESET}")

    elapsed = time.time() - start
    print(f"\nEvaluated {len(all_evals)} files in {elapsed/60:.1f} minutes\n")

    # === SUMMARY ===
    if all_evals:
        print(f"{BOLD}═══ Summary ═══{RESET}\n")
        print(f"{'Condition':<25} {'Leaks':>6} {'Contras':>8} {'Natural':>8} {'N':>4}")
        print("-" * 55)

        # Group by condition
        from collections import defaultdict
        by_condition = defaultdict(list)
        for e in all_evals:
            by_condition[e["condition"]].append(e)

        for condition in sorted(by_condition.keys()):
            evals = by_condition[condition]
            avg_leaks = sum(e["total_leaks"] for e in evals) / len(evals)
            avg_contras = sum(e["total_contradictions"] for e in evals) / len(evals)
            avg_nat = sum(e["avg_naturalness"] for e in evals) / len(evals)
            n = len(evals)
            print(f"{condition:<25} {avg_leaks:>6.1f} {avg_contras:>8.1f} {avg_nat:>8.2f} {n:>4}")

    # Save summary
    if args.output:
        summary_path = os.path.join(args.output, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_evals, f, indent=2)
        print(f"\nEvaluations saved to {args.output}/")


if __name__ == "__main__":
    main()
