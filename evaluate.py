"""
Evaluation Pipeline

Post-hoc evaluation of experiment results using GPT-4o-mini.
Scores each patient response for:
1. Leakage — did the response reveal withheld information?
2. Contradiction — did the response contradict any fact in the case?
3. Naturalness — does the response sound like a real patient? (1-5, granular)
4. Failure attribution — planner error, generator error, or verifier miss?

Also tracks: per-turn leakage, first leak turn, regeneration rate,
over-withholding (earned facts not disclosed), leak distribution by turn.

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
from collections import defaultdict

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


def call_gpt4o_mini(client: OpenAI, prompt: str, max_tokens: int = 300) -> dict:
    """Call GPT-4o-mini and parse JSON response."""
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    text = result.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


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

    return call_gpt4o_mini(client, prompt, 200)


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

    return call_gpt4o_mini(client, prompt, 300)


def evaluate_naturalness(
    client: OpenAI,
    response: str,
    student_message: str,
) -> dict:
    """Score how natural the patient response sounds (1-5)."""
    prompt = f"""Rate how natural this simulated patient response sounds. Use the FULL 1-5 scale — most responses should NOT be a 4.

Student: "{student_message}"
Patient: "{response}"

Apply these criteria strictly:

5 = Perfectly natural. Sounds exactly like a real patient speaking spontaneously. Uses natural fillers, appropriate length, direct answer, emotionally authentic. Reserve this for truly exceptional responses.

4 = Good but slightly off. One of these issues: slightly too polished, slightly too long, uses a word a patient wouldn't, or misses an emotional beat. Most competent responses land here.

3 = Noticeable issues. Two or more of: too formal or clinical for a patient, unnaturally structured, gives information the patient wouldn't volunteer, robotic phrasing, or awkward hedging.

2 = Clearly artificial. Sounds like an AI, not a person. May include: listing symptoms mechanically, using medical terminology a patient wouldn't know, breaking character, or being incoherent.

1 = Broken. Incoherent, refuses to respond in character, or completely inappropriate.

Be discriminating. Consider:
- Would a real patient say it this way?
- Is the length appropriate? (Real patients give short answers to simple questions.)
- Does it sound rehearsed or spontaneous?
- Does it use words a patient of this background would use?

Respond with ONLY this JSON:
{{
  "score": 3,
  "reason": "Brief explanation of what's natural and what's not"
}}"""

    return call_gpt4o_mini(client, prompt, 200)


def evaluate_failure_attribution(
    client: OpenAI,
    response: str,
    student_message: str,
    leaked_facts: list[str],
    condition: str,
) -> dict:
    """
    When a leak or contradiction occurs, determine the cause:
    planner error, generator error, or verifier miss.
    """
    prompt = f"""A simulated patient system has three components:
1. PLANNER: decides what facts to authorize for disclosure
2. GENERATOR: produces the patient's response using only authorized facts
3. VERIFIER: checks the response for leaks and contradictions

The system uses condition: {condition}

Student asked: "{student_message}"
Patient responded: "{response}"
Leaked fact IDs: {leaked_facts}

What is the most likely cause of this failure?

- "planner_error": The planner incorrectly authorized a withheld fact
- "generator_error": The generator fabricated or hallucinated information not in its brief (most common for isolated architecture since the generator never sees withheld facts)
- "verifier_miss": The verifier failed to catch the problem

For baseline conditions (naive_prompting, structured_prompting, self_monitoring), always attribute to "generator_error" since they don't have a separate planner/verifier.

Respond with ONLY this JSON:
{{
  "attribution": "planner_error" | "generator_error" | "verifier_miss",
  "explanation": "Brief explanation"
}}"""

    return call_gpt4o_mini(client, prompt, 200)


def evaluate_line_to_line_consistency(
    client: OpenAI,
    current_response: str,
    previous_response: str,
    turn_num: int,
) -> dict:
    """
    Line-to-line consistency (adapted from Abdulhai et al. NeurIPS 2025).
    Checks if the current patient response contradicts what the patient
    said in the immediately preceding turn.
    """
    prompt = f"""Check if these two consecutive patient responses contradict each other.

Previous response (turn {turn_num - 1}): "{previous_response}"
Current response (turn {turn_num}): "{current_response}"

A contradiction is when the current response states something that directly conflicts with what the patient said before. Examples:
- Previous: "It started 3 days ago" → Current: "I've had it for weeks" = INCONSISTENT
- Previous: "I don't smoke" → Current: "I smoke half a pack a day" = INCONSISTENT
- Providing new information not mentioned before is NOT inconsistency.
- Elaborating on a previous statement is NOT inconsistency.

Respond with ONLY this JSON:
{{
  "consistent": true/false,
  "inconsistency": "Brief description if inconsistent, empty string if consistent"
}}"""

    return call_gpt4o_mini(client, prompt, 150)


def evaluate_prompt_to_line_consistency(
    client: OpenAI,
    response: str,
    withheld_facts: list[dict],
    unlocked_ids: set[str],
) -> dict:
    """
    Prompt-to-line consistency (adapted from Abdulhai et al. NeurIPS 2025).
    Checks if the patient response contradicts withheld facts —
    not by leaking them, but by saying something incompatible with them.

    Example: If the withheld fact is "father had MI at 52" and the patient
    says "nobody in my family has had heart problems," that's a
    withheld-fact inconsistency (even though it's not a leak).
    """
    still_withheld = [f for f in withheld_facts if f["id"] not in unlocked_ids]

    if not still_withheld:
        return {"consistent": True, "inconsistencies": []}

    withheld_str = "\n".join(
        f"- {f['id']}: {f['content']}" for f in still_withheld
    )

    prompt = f"""Check if this patient response CONTRADICTS any withheld fact.

This is NOT about leaking — it's about consistency. The patient doesn't know these facts, but their response should not DENY or CONTRADICT them.

WITHHELD facts:
{withheld_str}

Patient's response: "{response}"

Examples:
- Withheld: "Father had heart attack at 52" → Response: "No heart problems in my family" = INCONSISTENT (denies a true fact)
- Withheld: "Takes ibuprofen for pain" → Response: "I don't take any medications" = INCONSISTENT
- Withheld: "Ankle swelling" → Response: "My chest hurts" = CONSISTENT (doesn't address ankle at all)
- Response hedges with "I'm not sure" = CONSISTENT (appropriate uncertainty)

Respond with ONLY this JSON:
{{
  "consistent": true/false,
  "inconsistencies": [
    {{"fact_id": "C18", "description": "Patient denied family heart history"}}
  ]
}}"""

    return call_gpt4o_mini(client, prompt, 250)


def build_progressive_unlock_map(result: dict) -> dict[int, set[str]]:
    """
    Build a map of turn_number -> set of fact IDs unlocked BY that turn.

    We reconstruct this by scanning the conversation. For architecture
    conditions, we look at the brief's newly_unlocked at each turn.
    Since we don't store briefs in results, we approximate by checking
    which turns' student messages contain unlock keywords from the case.

    For a simpler approach: we linearly interpolate the final unlocked set
    across turns based on when relevant questions were asked.
    """
    # We don't have per-turn unlock data in the results, so we use
    # a conservative approach: assume no facts are unlocked until the
    # student asks about the relevant topic. We check student messages
    # against unlock_keywords from the case.
    return None  # Signal to use the fallback approach


def evaluate_result_file(client: OpenAI, result_path: str, case_dir: str = "cases") -> dict:
    """Evaluate all turns in a single experiment result file."""
    with open(result_path) as f:
        result = json.load(f)

    # Load the case
    case_path = result.get("case_path", "")
    if not os.path.exists(case_path):
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

    condition = result.get("condition", "")
    condition_state = result.get("summary", {}).get("condition_state", {})
    final_unlocked = set(condition_state.get("unlocked_fact_ids", []))
    is_baseline = condition in ["naive_prompting", "structured_prompting", "self_monitoring"]
    is_architecture = condition in ["isolated_architecture", "no_isolation_ablation", "no_verifier_ablation"]

    # Build progressive unlock tracking using keyword matching
    # For each turn, determine which facts have been unlocked SO FAR
    # by scanning all student messages up to that turn for unlock keywords
    withheld_with_keywords = [f for f in case["facts"] if not f["disclosed"]]
    cumulative_unlocked_per_turn = []
    cumulative_student_text = ""

    for turn in result.get("turns", []):
        cumulative_student_text += " " + turn["student"].lower()
        unlocked_so_far = set()
        if is_architecture:
            for fact in withheld_with_keywords:
                keywords = fact.get("unlock_keywords", [])
                if any(kw.lower() in cumulative_student_text for kw in keywords):
                    # Keyword appeared, fact may have been unlocked
                    if fact["id"] in final_unlocked:
                        unlocked_so_far.add(fact["id"])
        cumulative_unlocked_per_turn.append(unlocked_so_far)

    # Evaluate each turn
    evaluations = []
    total_leaks = 0
    total_contradictions = 0
    naturalness_scores = []
    per_turn_leaks = []
    failure_attributions = []
    line_to_line_inconsistencies = 0
    prompt_to_line_inconsistencies = 0
    previous_response = None

    for i, turn in enumerate(result.get("turns", [])):
        turn_num = turn["turn"]
        student_msg = turn["student"]
        patient_resp = turn["patient"]

        # Determine unlocked facts at this turn
        if is_baseline:
            unlocked = set()
        elif is_architecture and i < len(cumulative_unlocked_per_turn):
            unlocked = cumulative_unlocked_per_turn[i]
        else:
            unlocked = final_unlocked

        # Evaluate leakage
        try:
            leak_result = evaluate_leakage(
                client, patient_resp, student_msg, withheld_facts, unlocked
            )
        except Exception as e:
            leak_result = {"leaked": False, "leaked_facts": [], "error": str(e)}

        # Evaluate contradiction
        try:
            contra_result = evaluate_contradiction(client, patient_resp, all_facts)
        except Exception as e:
            contra_result = {"has_contradiction": False, "contradictions": [], "error": str(e)}

        # Evaluate naturalness
        try:
            nat_result = evaluate_naturalness(client, patient_resp, student_msg)
        except Exception as e:
            nat_result = {"score": 0, "reason": str(e)}

        # Track per-turn leakage
        turn_leaked_facts = leak_result.get("leaked_facts", [])
        if leak_result.get("leaked") and turn_leaked_facts:
            total_leaks += len(turn_leaked_facts)
            per_turn_leaks.append({
                "turn": turn_num,
                "leaked_facts": turn_leaked_facts,
                "student_message": student_msg,
                "patient_response": patient_resp,
            })

            # Failure attribution
            try:
                attr_result = evaluate_failure_attribution(
                    client, patient_resp, student_msg,
                    turn_leaked_facts, condition,
                )
                failure_attributions.append({
                    "turn": turn_num,
                    "type": "leak",
                    "leaked_facts": turn_leaked_facts,
                    "attribution": attr_result,
                })
            except Exception:
                pass

        # Track contradictions with failure attribution
        if contra_result.get("has_contradiction"):
            total_contradictions += len(contra_result.get("contradictions", []))
            try:
                attr_result = evaluate_failure_attribution(
                    client, patient_resp, student_msg,
                    [c.get("fact_id", "") for c in contra_result["contradictions"]],
                    condition,
                )
                failure_attributions.append({
                    "turn": turn_num,
                    "type": "contradiction",
                    "contradictions": contra_result["contradictions"],
                    "attribution": attr_result,
                })
            except Exception:
                pass

        if nat_result.get("score"):
            naturalness_scores.append(nat_result["score"])

        # Line-to-line consistency (turn N vs turn N-1)
        l2l_result = {"consistent": True, "inconsistency": ""}
        if previous_response is not None:
            try:
                l2l_result = evaluate_line_to_line_consistency(
                    client, patient_resp, previous_response, turn_num
                )
                if not l2l_result.get("consistent", True):
                    line_to_line_inconsistencies += 1
            except Exception as e:
                l2l_result = {"consistent": True, "inconsistency": "", "error": str(e)}

        # Prompt-to-line consistency (response vs withheld facts)
        p2l_result = {"consistent": True, "inconsistencies": []}
        try:
            p2l_result = evaluate_prompt_to_line_consistency(
                client, patient_resp, withheld_facts, unlocked
            )
            if not p2l_result.get("consistent", True):
                prompt_to_line_inconsistencies += len(p2l_result.get("inconsistencies", []))
        except Exception as e:
            p2l_result = {"consistent": True, "inconsistencies": [], "error": str(e)}

        previous_response = patient_resp

        evaluations.append({
            "turn": turn_num,
            "unlocked_at_this_turn": sorted(unlocked),
            "leakage": leak_result,
            "contradiction": contra_result,
            "naturalness": nat_result,
            "line_to_line_consistency": l2l_result,
            "prompt_to_line_consistency": p2l_result,
        })

    # Summary
    avg_naturalness = sum(naturalness_scores) / len(naturalness_scores) if naturalness_scores else 0
    first_leak_turn = per_turn_leaks[0]["turn"] if per_turn_leaks else None
    regeneration_rate = condition_state.get("total_retries", 0)

    # Over-withholding: for architecture conditions, count how many facts
    # were unlocked. Higher is better (student earned disclosure).
    # Compare against total withheld to get disclosure rate.
    total_withheld = len(withheld_facts)
    total_unlocked = len(final_unlocked) if is_architecture else 0
    disclosure_rate = total_unlocked / total_withheld if total_withheld > 0 else 0

    eval_summary = {
        "condition": condition,
        "strategy": result.get("strategy", ""),
        "case_id": result.get("case_id", ""),
        "run": result.get("run", 1),
        "total_leaks": total_leaks,
        "total_contradictions": total_contradictions,
        "avg_naturalness": round(avg_naturalness, 2),
        "line_to_line_inconsistencies": line_to_line_inconsistencies,
        "prompt_to_line_inconsistencies": prompt_to_line_inconsistencies,
        "turns_evaluated": len(evaluations),
        "first_leak_turn": first_leak_turn,
        "regeneration_rate": regeneration_rate,
        "disclosure_rate": round(disclosure_rate, 2),
        "total_unlocked": total_unlocked,
        "total_withheld": total_withheld,
        "per_turn_leaks": per_turn_leaks,
        "failure_attributions": failure_attributions,
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
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_summary(all_evals: list[dict]):
    """Print comprehensive summary statistics."""

    print(f"{BOLD}═══ Summary by Condition ═══{RESET}\n")
    print(f"{'Condition':<25} {'Leaks':>6} {'Contras':>8} {'L2L':>5} {'P2L':>5} {'Natural':>8} {'1st Leak':>9} {'Discl%':>7} {'N':>4}")
    print("-" * 87)

    by_condition = defaultdict(list)
    for e in all_evals:
        by_condition[e["condition"]].append(e)

    condition_order = [
        "naive_prompting", "structured_prompting", "self_monitoring",
        "isolated_architecture", "no_isolation_ablation", "no_verifier_ablation",
    ]

    for condition in condition_order:
        if condition not in by_condition:
            continue
        evals = by_condition[condition]
        avg_leaks = sum(e["total_leaks"] for e in evals) / len(evals)
        avg_contras = sum(e["total_contradictions"] for e in evals) / len(evals)
        avg_l2l = sum(e.get("line_to_line_inconsistencies", 0) for e in evals) / len(evals)
        avg_p2l = sum(e.get("prompt_to_line_inconsistencies", 0) for e in evals) / len(evals)
        avg_nat = sum(e["avg_naturalness"] for e in evals) / len(evals)
        first_leaks = [e["first_leak_turn"] for e in evals if e["first_leak_turn"] is not None]
        avg_first = sum(first_leaks) / len(first_leaks) if first_leaks else None
        avg_discl = sum(e["disclosure_rate"] for e in evals) / len(evals)
        n = len(evals)

        first_str = f"{avg_first:.1f}" if avg_first else "-"
        discl_str = f"{avg_discl:.0%}" if avg_discl > 0 else "-"
        print(f"{condition:<25} {avg_leaks:>6.1f} {avg_contras:>8.1f} {avg_l2l:>5.1f} {avg_p2l:>5.1f} {avg_nat:>8.2f} {first_str:>9} {discl_str:>7} {n:>4}")

    # Leakage by strategy
    print(f"\n{BOLD}═══ Leakage by Strategy (avg leaks per experiment) ═══{RESET}\n")
    by_strategy = defaultdict(lambda: defaultdict(list))
    for e in all_evals:
        by_strategy[e["strategy"]][e["condition"]].append(e["total_leaks"])

    strategies = ["direct_questioning", "rephrasing", "emotional_appeal",
                  "authority_claim", "gradual_escalation", "logical_inference"]

    short = {
        "naive_prompting": "naive",
        "structured_prompting": "struct",
        "self_monitoring": "self_m",
        "isolated_architecture": "isolat",
        "no_isolation_ablation": "no_iso",
        "no_verifier_ablation": "no_ver",
    }

    header = f"{'Strategy':<22}" + "".join(f"{short.get(c, c):>8}" for c in condition_order)
    print(header)
    print("-" * len(header))

    for strategy in strategies:
        row = f"{strategy:<22}"
        for condition in condition_order:
            leaks = by_strategy[strategy].get(condition, [])
            if leaks:
                avg = sum(leaks) / len(leaks)
                row += f"{avg:>8.1f}"
            else:
                row += f"{'-':>8}"
        print(row)

    # Failure attribution summary
    all_attributions = []
    for e in all_evals:
        all_attributions.extend(e.get("failure_attributions", []))

    if all_attributions:
        print(f"\n{BOLD}═══ Failure Attribution ═══{RESET}\n")
        attr_counts = defaultdict(int)
        for a in all_attributions:
            attr = a.get("attribution", {}).get("attribution", "unknown")
            attr_counts[attr] += 1
        total_attr = sum(attr_counts.values())
        for attr, count in sorted(attr_counts.items(), key=lambda x: -x[1]):
            pct = count / total_attr * 100
            print(f"  {attr}: {count} ({pct:.0f}%)")

    # Per-turn leak distribution
    print(f"\n{BOLD}═══ Leak Distribution by Turn ═══{RESET}\n")
    turn_leaks = defaultdict(int)
    for e in all_evals:
        for ptl in e.get("per_turn_leaks", []):
            turn_leaks[ptl["turn"]] += len(ptl["leaked_facts"])

    if turn_leaks:
        max_turn = max(turn_leaks.keys())
        for t in range(1, max_turn + 1):
            count = turn_leaks.get(t, 0)
            bar = "█" * count
            print(f"  Turn {t:2d}: {count:3d} {bar}")
    else:
        print("  No leaks detected across all experiments.")


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

            if args.output:
                rel_path = os.path.relpath(filepath, args.path)
                out_path = os.path.join(args.output, rel_path.replace(".json", "_eval.json"))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(eval_result, f, indent=2)

        except Exception as e:
            print(f"{RED}FAILED: {e}{RESET}")

    elapsed = time.time() - start
    print(f"\nEvaluated {len(all_evals)} files in {elapsed/60:.1f} minutes\n")

    if all_evals:
        print_summary(all_evals)

    if args.output:
        summary_path = os.path.join(args.output, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_evals, f, indent=2)
        print(f"\nEvaluations saved to {args.output}/")


if __name__ == "__main__":
    main()
