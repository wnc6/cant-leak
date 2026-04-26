"""
Fareez Distributional Comparison

Compares our generated patient responses against real OSCE patient responses
from Fareez et al. (2022) on three dimensions:
1. Naturalness scores (GPT-4o-mini, same prompt as evaluation pipeline)
2. Response length distribution
3. Hedging frequency

Usage:
    export OPENAI_API_KEY=sk-...
    python3 fareez_comparison.py --fareez "Clean Transcripts/" --evals evals/ --output fareez_analysis/
"""

import json
import os
import sys
import glob
import re
import random
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip3 install openai")
    sys.exit(1)


HEDGING_PHRASES = [
    "i'm not sure", "i don't know", "i'm not certain", "i can't remember",
    "maybe", "i think", "possibly", "not really sure", "hard to say",
    "i don't recall", "i'm not positive", "it's hard to tell",
    "i couldn't say", "i guess", "not that i know of", "i don't think so",
    "i'm not really", "honestly i'm not", "to be honest i'm not sure",
]


def parse_fareez_transcripts(fareez_dir, n_samples=50, seed=42):
    """Extract patient responses from Fareez transcripts."""
    random.seed(seed)
    
    all_patient_responses = []
    
    for filepath in sorted(glob.glob(os.path.join(fareez_dir, "*.txt"))):
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("P:"):
                response = line[2:].strip()
                if len(response) > 10:  # skip very short responses
                    # Get the preceding doctor question for context
                    doctor_msg = ""
                    for j in range(i - 1, -1, -1):
                        if lines[j].strip().startswith("D:"):
                            doctor_msg = lines[j].strip()[2:].strip()
                            break
                    all_patient_responses.append({
                        "student": doctor_msg,
                        "patient": response,
                        "source": os.path.basename(filepath),
                    })
    
    # Sample
    sampled = random.sample(all_patient_responses, min(n_samples, len(all_patient_responses)))
    return sampled


def load_our_responses(evals_dir, n_samples=50, seed=42):
    """Load patient responses from our evaluation results."""
    random.seed(seed)
    
    all_responses = []
    pattern = os.path.join(evals_dir, "**", "*_eval.json")
    
    for filepath in sorted(glob.glob(pattern, recursive=True)):
        if "summary" in filepath:
            continue
        with open(filepath) as f:
            eval_data = json.load(f)
        
        condition = eval_data.get("condition", "")
        for ev in eval_data.get("evaluations", []):
            # Get the response from the original turn
            nat_score = ev.get("naturalness", {}).get("score", 0)
            all_responses.append({
                "condition": condition,
                "naturalness": nat_score,
            })
    
    # Also load raw responses for length and hedging analysis
    raw_responses = []
    for filepath in sorted(glob.glob(os.path.join("results", "**", "run_*.json"), recursive=True)):
        with open(filepath) as f:
            result = json.load(f)
        for turn in result.get("turns", []):
            raw_responses.append({
                "student": turn["student"],
                "patient": turn["patient"],
                "condition": result.get("condition", ""),
            })
    
    sampled = random.sample(raw_responses, min(n_samples, len(raw_responses)))
    return sampled, all_responses


def score_naturalness(client, responses, label=""):
    """Score responses using the same naturalness prompt as evaluate.py."""
    scores = []
    
    for i, r in enumerate(responses):
        print(f"  [{label}] Scoring {i+1}/{len(responses)}...", end="\r")
        
        prompt = f"""Rate how natural this simulated patient response sounds. Use the FULL 1-5 scale — most responses should NOT be a 4.

Student: "{r['student']}"
Patient: "{r['patient']}"

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
{{"score": 3, "reason": "Brief explanation"}}"""

        try:
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
            parsed = json.loads(text.strip())
            scores.append(parsed.get("score", 0))
        except Exception as e:
            scores.append(0)
    
    print(f"  [{label}] Done. {len(scores)} scored.          ")
    return scores


def compute_hedging_rate(responses):
    """Count hedging phrases per response."""
    hedging_counts = []
    for r in responses:
        patient_lower = r["patient"].lower()
        count = sum(1 for phrase in HEDGING_PHRASES if phrase in patient_lower)
        hedging_counts.append(count)
    return hedging_counts


def compute_response_lengths(responses):
    """Word count per response."""
    return [len(r["patient"].split()) for r in responses]


def mann_whitney_u(x, y):
    """Simple Mann-Whitney U test (no scipy dependency)."""
    n1, n2 = len(x), len(y)
    combined = [(val, 'x') for val in x] + [(val, 'y') for val in y]
    combined.sort(key=lambda t: t[0])
    
    # Assign ranks (handle ties with average rank)
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            if combined[k] not in ranks:
                ranks[combined[k]] = []
            ranks[combined[k]].append(avg_rank)
        i = j
    
    # Sum ranks for x
    rank_sum_x = 0
    for val in x:
        # Find rank
        for k, (v, group) in enumerate(combined):
            if v == val and group == 'x':
                rank_sum_x += (k + 1)  # simplified: use position
                combined[k] = (v, 'used')
                break
    
    u1 = rank_sum_x - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    
    return min(u1, u2), max(u1, u2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fareez", default="Clean Transcripts/", help="Path to Fareez transcripts")
    parser.add_argument("--evals", default="evals/", help="Path to evaluation results")
    parser.add_argument("--output", default="fareez_analysis/", help="Output directory")
    parser.add_argument("--n", type=int, default=50, help="Number of samples per group")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Parse Fareez responses
    print("Loading Fareez transcripts...")
    fareez_responses = parse_fareez_transcripts(args.fareez, args.n)
    print(f"  Sampled {len(fareez_responses)} Fareez patient responses")

    # Load our responses
    print("Loading our responses...")
    our_responses, our_eval_data = load_our_responses(args.evals, args.n)
    print(f"  Sampled {len(our_responses)} of our patient responses")

    # === 1. Naturalness Comparison ===
    print("\n=== Naturalness Comparison ===")
    
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        client = OpenAI(api_key=key)
        
        fareez_scores = score_naturalness(client, fareez_responses, "Fareez")
        our_scores = score_naturalness(client, our_responses, "Ours")
        
        fareez_valid = [s for s in fareez_scores if s > 0]
        our_valid = [s for s in our_scores if s > 0]
        
        fareez_mean = sum(fareez_valid) / len(fareez_valid) if fareez_valid else 0
        our_mean = sum(our_valid) / len(our_valid) if our_valid else 0
        
        print(f"\n  Fareez mean naturalness: {fareez_mean:.2f} (n={len(fareez_valid)})")
        print(f"  Ours mean naturalness:   {our_mean:.2f} (n={len(our_valid)})")
        print(f"  Difference:              {our_mean - fareez_mean:+.2f}")
        
        # Distribution
        print(f"\n  Fareez distribution: ", end="")
        for score in range(1, 6):
            count = fareez_valid.count(score)
            pct = count / len(fareez_valid) * 100 if fareez_valid else 0
            print(f"{score}={pct:.0f}% ", end="")
        print()
        
        print(f"  Ours distribution:   ", end="")
        for score in range(1, 6):
            count = our_valid.count(score)
            pct = count / len(our_valid) * 100 if our_valid else 0
            print(f"{score}={pct:.0f}% ", end="")
        print()
    else:
        print("  Skipping (no OPENAI_API_KEY). Set it to score Fareez responses.")
        fareez_scores = []
        our_scores = []

    # === 2. Response Length Comparison ===
    print("\n=== Response Length Comparison ===")
    
    fareez_lengths = compute_response_lengths(fareez_responses)
    our_lengths = compute_response_lengths(our_responses)
    
    fareez_mean_len = sum(fareez_lengths) / len(fareez_lengths)
    our_mean_len = sum(our_lengths) / len(our_lengths)
    
    print(f"  Fareez mean length: {fareez_mean_len:.1f} words")
    print(f"  Ours mean length:   {our_mean_len:.1f} words")
    print(f"  Difference:         {our_mean_len - fareez_mean_len:+.1f} words")

    # === 3. Hedging Frequency Comparison ===
    print("\n=== Hedging Frequency Comparison ===")
    
    fareez_hedging = compute_hedging_rate(fareez_responses)
    our_hedging = compute_hedging_rate(our_responses)
    
    # Also compute per-condition hedging from all results
    print("\n  Per-condition hedging (all results):")
    condition_hedging = defaultdict(list)
    for filepath in sorted(glob.glob(os.path.join("results", "**", "run_*.json"), recursive=True)):
        with open(filepath) as f:
            result = json.load(f)
        condition = result.get("condition", "")
        for turn in result.get("turns", []):
            patient_lower = turn["patient"].lower()
            count = sum(1 for phrase in HEDGING_PHRASES if phrase in patient_lower)
            condition_hedging[condition].append(count)
    
    conditions = [
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
    
    fareez_hedge_rate = sum(1 for h in fareez_hedging if h > 0) / len(fareez_hedging) * 100
    print(f"  Fareez: {fareez_hedge_rate:.0f}% of responses contain hedging")
    
    for c in conditions:
        vals = condition_hedging.get(c, [])
        if vals:
            rate = sum(1 for h in vals if h > 0) / len(vals) * 100
            avg = sum(vals) / len(vals)
            print(f"  {condition_labels.get(c, c)}: {rate:.0f}% of responses contain hedging (avg {avg:.2f} per response)")

    # === Save Results ===
    results = {
        "n_fareez": len(fareez_responses),
        "n_ours": len(our_responses),
        "naturalness": {
            "fareez_mean": round(sum(fareez_scores) / len(fareez_scores), 2) if fareez_scores else None,
            "ours_mean": round(sum(our_scores) / len(our_scores), 2) if our_scores else None,
            "fareez_scores": fareez_scores,
            "our_scores": our_scores,
        },
        "response_length": {
            "fareez_mean": round(fareez_mean_len, 1),
            "ours_mean": round(our_mean_len, 1),
        },
        "hedging": {
            "fareez_rate": round(fareez_hedge_rate, 1),
            "per_condition": {
                condition_labels.get(c, c): {
                    "rate": round(sum(1 for h in condition_hedging.get(c, []) if h > 0) / len(condition_hedging.get(c, [1])) * 100, 1),
                    "avg_per_response": round(sum(condition_hedging.get(c, [0])) / len(condition_hedging.get(c, [1])), 2),
                }
                for c in conditions if condition_hedging.get(c)
            },
        },
    }
    
    results_path = os.path.join(args.output, "fareez_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
