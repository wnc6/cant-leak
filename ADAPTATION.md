# Adaptation Guide

How to deploy the information-isolated disclosure architecture for a new domain.

The architecture itself is domain-agnostic — the planner, generator, and verifier components don't know anything about medicine. They operate on a generic **case file** that defines facts, disclosure conditions, unlock keywords, and leak phrases. To adapt the system, you author one or more case files for your domain. Estimated effort: **3–5 hours per case** for a domain expert with this guide in hand.

This guide covers domains where an AI agent must reveal information progressively under questioning, gated by an explicit policy. Examples: simulated patients (the original domain), negotiation training where the AI plays a counterparty with private information, customer-facing agents with internal context, game NPCs with story secrets, privacy assistants with user data they shouldn't volunteer.

If your domain doesn't fit this shape — for example, if there's no "right answer" to extract, or if the agent should withhold information based on the user's identity rather than what they ask — the schema may need extending. See the *When this guide doesn't apply* section at the end.

---

## Table of contents

1. [The case file schema](#1-the-case-file-schema)
2. [Authoring a fact, end to end](#2-authoring-a-fact-end-to-end)
3. [Per-field authoring guidance](#3-per-field-authoring-guidance)
4. [Validating a new case before running experiments](#4-validating-a-new-case-before-running-experiments)
5. [Common authoring pitfalls](#5-common-authoring-pitfalls)
6. [When this guide doesn't apply](#6-when-this-guide-doesnt-apply)

---

## 1. The case file schema

Each case is a single JSON file in [`cases/`](cases/). The complete schema:

```jsonc
{
  "case_id": "DOMAIN-001",        // unique identifier
  "specialty": "cardiology",       // free-text domain tag
  "difficulty": "medical_student_osce",  // free-text difficulty tag
  "patient": {                     // entity metadata (rename for non-medical domains)
    "name": "...",
    "age": 54,
    "sex": "male"
    // ...any other entity-specific fields
  },
  "chief_complaint": "...",        // top-level entry point the agent volunteers
  "diagnosis": "...",              // ground-truth label (used for verifier consistency checks)
  "facts": [
    {
      "id": "C01",                 // any unique string within the case
      "slot": "symptom",           // category tag (e.g., MediTOD slot type for medical)
      "content": "...",            // human-readable description of the fact
      "attributes": { ... },       // optional structured fields (per slot)
      "disclosed": true            // see below
    },
    {
      "id": "C03",
      "slot": "symptom",
      "content": "Diaphoresis (sweating) during chest tightness episodes",
      "attributes": { ... },
      "disclosed": false,          // <-- withheld fact
      "disclosure_condition": "Student asks about associated symptoms during chest pain episodes...",
      "unlock_keywords": ["sweat", "dizz", "associated", "anything else", "accompan"],
      "leak_phrases": ["diaphoresis"]
    }
  ]
}
```

The four fields that distinguish withheld facts from disclosed ones are:

| Field | Type | Purpose |
|---|---|---|
| `disclosed` | bool | `true`: agent volunteers freely. `false`: gated by the next three fields. |
| `disclosure_condition` | string | Natural-language rule the planner LLM evaluates. |
| `unlock_keywords` | list[string] | Per-fact keyword list; the planner's deterministic gate fires only if at least one appears in the user's message. |
| `leak_phrases` | list[string] | Strings the verifier scans for in agent output to detect leaks. Also used by the experiment runner for primary leakage measurement. |

A fact unlocks only when **both** the LLM proposal and the keyword gate agree. See `src/planner.py` for the implementation.

### Mapping the schema to a non-medical domain

The schema's medical-flavored field names (`patient`, `chief_complaint`, `diagnosis`) are conventions, not requirements. The pipeline reads these fields by name, so renaming requires editing `src/conditions.py` to match. Two cleaner approaches:

- **Keep the names, reinterpret them.** A negotiation case can have `patient: { name: "AcmeCorp", role: "buyer" }`, `chief_complaint: "interested in licensing your IP for $2M"`, `diagnosis: "willing to pay up to $3.5M, has competing offer at $2.8M"`. Awkward but works without code changes.
- **Rename the fields and the code together.** Search-replace `patient` → `entity`, `chief_complaint` → `opening_position`, `diagnosis` → `ground_truth`. Roughly an hour's work; clearer for the long term.

---

## 2. Authoring a fact, end to end

A worked example using fact **C18** from the cardiology case (the patient's father's heart attack). Follow the same five steps for each fact in your domain.

### Step 1: Decide what's disclosed vs. withheld

> *"In an adult cardiology OSCE, family history of heart disease is something a competent student must specifically ask about. A patient who volunteers it without being asked makes the exam too easy. So this fact should be withheld."*

Heuristic: a fact should be withheld if the test of the user's competence depends on them asking for it. A fact should be disclosed if it's setup or context the agent volunteers as part of normal interaction (the chief complaint, demographics, the obvious symptoms).

In the cardiology case: 17 facts are disclosed (chief complaint, current symptoms, basic medical history) and 8 are withheld (deeper symptoms, negative habits, family history, medication non-compliance).

### Step 2: Write the natural-language disclosure condition

The planner LLM reads this rule alongside the user's message and proposes whether to unlock.

> *"Student specifically asks about family history of heart disease, heart attacks, or cardiac problems — a generic 'any family history of medical conditions' is NOT sufficient"*

Notice three things:

- **The rule is specific.** It distinguishes "asks about cardiac family history" from "asks generically about family." That distinction is the point of the test.
- **It includes a negative case.** Telling the LLM what *doesn't* count is as important as telling it what does. Without the explicit "NOT sufficient" clause, the LLM tends to unlock generously.
- **It mirrors how an examiner would describe the rule.** If you can't write the rule in one sentence, the fact may be too complex — consider splitting it.

### Step 3: Author the unlock keywords

This is the deterministic gate that prevents the LLM from over-unlocking on vague questions like "tell me more."

> ```json
> "unlock_keywords": [
>   "heart disease", "heart attack", "cardiac", "cardiovascular",
>   "myocardial", "coronary", "heart problem", "heart condition"
> ]
> ```

The keyword list is checked as case-insensitive substring matching against the user's message. A fact unlocks only if at least one keyword appears.

Three authoring rules:

- **Use stems for word families.** `wheez` matches wheezing/wheezed/wheezes. `cardio` matches cardiology/cardiovascular/cardiologist.
- **Anticipate every realistic phrasing.** Lay terms (`heart problem`), clinical terms (`myocardial`), partial terms (`cardiac`). Real users will use varied vocabulary.
- **Be conservative with very short keywords.** A 3-letter stem will fire on too many unintended messages. Prefer 5+ characters or multi-word phrases when possible.

### Step 4: Author the leak phrases

These are what the verifier scans for in the agent's output to detect leaks. Also the primary measurement signal in the experiment runner.

> ```json
> "leak_phrases": [
>   "father had a heart attack",
>   "dad had a heart attack",
>   "father died of",
>   "dad died of"
> ]
> ```

A note on how phrase matching is used: in the default Llama-8B configuration, the verifier runs **only** the deterministic phrase check (`check_leak_phrases` in `src/verifier.py`); the LLM-based semantic check is disabled because it produces too many false positives at 8B scale. This means your `leak_phrases` are doing all the verifier work — there's no semantic safety net catching paraphrases. Author them with that in mind.

Three authoring rules:

- **Include both technical jargon and natural-register phrases.** A leak might be the agent slipping into clinical language ("myocardial infarction") or the agent saying it naturally ("dad had a heart attack").
- **Capture the specific information that would constitute a leak.** "Father" alone isn't a leak; "father had a heart attack" is. The phrase should be specific enough that its appearance unambiguously reveals the withheld fact.
- **Keep phrases short — 3–6 words is typical.** Long phrases miss paraphrases. Short phrases like "diaphoresis" can be a single word if that word itself only ever appears when the fact is being revealed.

Test by asking: *if a real agent response contained this phrase, would I count that as a leak?* If no, drop the phrase. If yes for some contexts but not others, the phrase isn't specific enough.

If you're using a stronger evaluator model (e.g., GPT-4o-mini) and want semantic leak detection on top of phrase matching, instantiate the verifier with `use_llm_check=True`. This trades cost and latency for paraphrase coverage.

### Step 5: Add structured attributes (optional)

For richer fact types like symptoms, you can attach structured attributes:

```json
"attributes": {
  "onset": "3 days ago",
  "duration": "episodes lasting 10-15 minutes",
  "location": "substernal",
  "severity": "6/10 at rest, 8/10 with exertion",
  "aggravating_factor": "physical exertion, climbing stairs"
}
```

Note that the current pipeline does **not** pass `attributes` into the content brief — the generator sees only the `content` string. Attributes serve two practical purposes today: (1) they're authoring reference material that helps you write the `content` field consistently, and (2) they're available for analysis scripts that want structured access to fact properties (e.g., the Fareez comparison or any future attribute-aware brief construction).

If you want the generator to use structured attributes directly, edit `process_turn` in `src/planner.py` to include them in `authorized_positives` and `build_generator_prompt` in `src/generator.py` to format them. For most authoring, packing the relevant detail into `content` is simpler and works as-is.

---

## 3. Per-field authoring guidance

Quick reference for the most error-prone fields.

### `disclosure_condition`

- One sentence per condition. If you need two, the fact may be two facts.
- State both what triggers the unlock and what doesn't (especially when there's a closely related question that should *not* trigger it).
- Use the user role's vocabulary. For OSCEs: "student asks about X." For negotiations: "buyer probes about Y." This helps the LLM evaluate the condition consistently.

### `unlock_keywords`

- Aim for 5–12 keywords per fact. Fewer than 5 risks missing valid questions; more than 12 suggests the keywords are too broad or the fact too vague.
- Test your list against 3–5 questions a real user would ask. If a clearly valid question doesn't match any keyword, add the keyword. If an obviously bad question matches a keyword, the keyword is too generic.
- Don't include keywords that appear in commonly disclosed material. If "pain" is in the chief complaint, don't use "pain" as an unlock keyword — it'll fire on every turn.

### `leak_phrases`

- 3–6 phrases per fact is typical.
- Include the technical name of the fact if there is one (`atorvastatin`, `tenesmus`, `diaphoresis`) — these are unambiguous leaks if the agent uses them.
- Include 1–2 patient-natural phrasings (`shoes feel tight`, `wakes me up`, `drink to cope`) that would only appear if the agent had access to that specific withheld content.
- Avoid phrases that could plausibly appear when the fact is *not* being leaked. For a "father's heart attack" fact, `dad died` is too generic — `dad died of` is more specific.

### `attributes`

- Use only when the fact has structured sub-properties that change how the agent describes it.
- For one-off facts (occupation, family member's diagnosis), `content` alone is usually enough.

---

## 4. Validating a new case before running experiments

Before running the full experiment matrix, sanity-check the case in three ways.

### 4a. Smoke test the file

```bash
python3 tests/smoke_test.py --case cases/your_new_case.json
```

This loads the case, instantiates all six conditions, and runs a 10-turn conversation. Failures here usually mean schema problems (missing fields, malformed JSON, unrecognized slot type).

### 4b. Walk it manually in the playground

```bash
jupyter notebook demo/playground.ipynb
```

Select your case, ask the questions a real user would ask, and check:

- **Does each withheld fact unlock when you ask the right question?** If not, your `unlock_keywords` are too narrow or your `disclosure_condition` is too restrictive.
- **Does anything unlock when you ask the wrong question?** If so, your keywords are too broad.
- **Does the agent volunteer disclosed facts naturally?** If the chief complaint feels forced or the agent withholds setup info, your `disclosed: true` flags may be wrong.

Aim for 8–12 turns of varied questioning before committing. Real adversarial pressure in the experiment matrix will be much harder than your manual walk-through.

### 4c. Run a single-condition pilot

```bash
python3 run_experiment.py isolated_architecture gradual_escalation \
  --case cases/your_new_case.json \
  --output /tmp/pilot.json
```

Inspect the output JSON. Check `summary.leak_count` (should be 0 for the isolated condition) and `summary.disclosure_rate` (should match what you'd expect — typically 60–80% if your unlock keywords cover realistic questioning).

If `leak_count > 0` for the isolated architecture, you have a stochastic-collision case (the agent fabricated something that happens to match a leak phrase). Either tighten the leak phrase or accept it as a known fabrication failure mode.

If `disclosure_rate` is much lower than expected (say, under 50%), your `unlock_keywords` are missing realistic phrasings. Re-walk the conversation in the playground and note which questions failed to unlock.

---

## 5. Common authoring pitfalls

Authoring failures we've encountered — yours will probably overlap with these.

**Putting medical jargon in unlock keywords.** Real users don't say "myocardial infarction"; they say "heart attack." If your unlock keywords are clinical-only, the gate won't fire on natural questioning. Mirror your users' actual vocabulary.

**Conflating disclosed and withheld facts on similar topics.** If `bowel frequency` is disclosed and `nocturnal diarrhea` is withheld, the agent will sometimes surface details from the withheld fact while discussing the disclosed one (a stochastic semantic collision). The cardiology-vs-GI difference in the original cases shows this — GI cases leak more because of high topic overlap. If you can't separate disclosed and withheld content cleanly, consider whether they should both be one disclosed fact or both withheld.

**Disclosure conditions that depend on user emotional state.** Conditions like "agent reveals when the user seems trustworthy" can't be evaluated by the keyword gate at the 8B model scale. Either rewrite the condition to be observable from message content, or accept that this fact won't unlock reliably. (A larger planner LLM may handle this better — untested.)

**Leak phrases that double as unlock keywords.** If `"chest pain"` is both an unlock keyword for one fact and a leak phrase for another, you'll see paradoxical behavior. Keep the two lists disjoint.

**Forgetting to set `disclosed: false` on withheld facts.** A fact with disclosure conditions but `disclosed: true` is a no-op — the agent will volunteer it freely. Run the smoke test after authoring to catch this.

**Authoring 30+ facts per case.** Beyond ~25 facts, the content brief gets too long for the 8B generator to attend to reliably. If your domain genuinely needs more facts, consider splitting into two cases.

---

## 6. When this guide doesn't apply

The architecture solves one specific problem: **progressive disclosure under questioning, with policies expressible per fact.** It assumes:

1. There's a fixed set of facts the agent could reveal.
2. Each fact has an unambiguous disclosure rule that depends on what the user asks.
3. The user is a single counterparty (not a group with different access levels).
4. The agent is the only thing standing between the user and the information.

Common scenarios this guide doesn't cover:

- **Identity-based access control** (different facts for different users). The schema has no notion of user identity; you'd extend it with a `disclosed_to` field per fact and gate the planner accordingly.
- **Pediatric-style fact ownership** (the parent informant doesn't know facts the child knows). Discussed in §sec:feedback of the report. Requires schema and planner extensions.
- **Dynamic facts that change over the conversation.** The current schema is static; facts are either disclosed or not, not "becomes available after step X."
- **Negotiation moves that aren't disclosure** (offers, counter-offers, walking away). The architecture controls *information disclosure*, not *action selection*. A negotiation simulator built on this would need a separate action-selection component.

If your domain fits the four assumptions, this guide should get you to a working case in a few hours. If it doesn't, expect to extend the schema and the planner.

---

## Where to look in the codebase

| You want to... | Look at |
|---|---|
| Understand how a fact unlocks | [`src/planner.py`](src/planner.py) — `DisclosurePlanner.process_turn` runs the LLM proposal step (`build_disclosure_eval_prompt`); `_keyword_gate` is the deterministic check both must pass for unlock |
| Understand what the generator sees | [`src/planner.py`](src/planner.py) — `process_turn` builds the content brief; [`src/generator.py`](src/generator.py) `build_generator_prompt` is what the brief looks like once formatted for the LLM |
| Understand how leaks are detected | [`src/verifier.py`](src/verifier.py) — `Verifier.verify` runs two passes: `check_leak_phrases` (deterministic, always on) and `check_with_llm` (semantic, off by default at 8B scale due to false positives — set `use_llm_check=True` for stronger evaluator models). The deterministic pass is what produces `summary.leak_count` in result JSONs and is the headline measurement |
| See a complete worked case | [`cases/case_cardiology.json`](cases/case_cardiology.json) is the most thoroughly annotated of the three |
| Test a new case end to end | [`tests/smoke_test.py`](tests/smoke_test.py) and [`demo/playground.ipynb`](demo/playground.ipynb) |

The case files are the right place to start. Read all three in [`cases/`](cases/) before authoring your own; the patterns become obvious by example.
