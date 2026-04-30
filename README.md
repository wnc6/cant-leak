# You Can't Leak What You Don't Know
Information-isolated disclosure architecture for controlling what LLM dialogue agents reveal under pressure.

## Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) for local LLM inference
- 16+ GB disk, 12+ GB RAM for `llama3.1:8b-instruct-fp16`
- OpenAI API key (only for the supplementary GPT-4o-mini evaluation)

## Repository Layout

### Architecture and benchmark

The system itself, plus the cases it operates on. These are the inputs to everything else.

- [`src/`](src/) — the [`planner`](src/planner.py), [`generator`](src/generator.py), [`verifier`](src/verifier.py), [`student agent`](src/student_agent.py), and [`six experimental conditions`](src/conditions.py) (naive, structured, self-monitoring, isolated, two ablations)
- [`cases/`](cases/) — three OSCE clinical cases ([`cardiology`](cases/case_cardiology.json), [`respiratory`](cases/case_respiratory.json), [`GI`](cases/case_gi.json)) with disclosure conditions, unlock keywords, leak phrases, and CMAS symptom attributes
- [`tests/`](tests/) — unit tests for the [`planner`](tests/test_planner.py), [`end-to-end pipeline integration`](tests/test_pipeline.py), and [`a fast smoke test`](tests/smoke_test.py) (fixtures in [`tests/fixtures/`](tests/fixtures/))

### Main pipeline

The scripts that produce the report's primary numbers.

- [`run_experiment.py`](run_experiment.py) — run a single conversation (1 case × 1 condition × 1 strategy)
- [`run_all.py`](run_all.py) — run the full 324-experiment matrix (3 runs × 3 cases × 6 conditions × 6 strategies)
- [`evaluate.py`](evaluate.py) — supplementary GPT-4o-mini evaluation (contradictions, naturalness, failure attribution)
- [`summarize_runs.py`](summarize_runs.py) — aggregate `results/` into the report's primary leakage tables (Table 2, RQ1, RQ4)
- [`summarize_evals.py`](summarize_evals.py) — aggregate `evals/` into supplementary tables (Tables 4, 7, naturalness)
- [`generate_charts.py`](generate_charts.py) — produce the 6 charts used in the report
- [`annotate.py`](annotate.py) — sample 50 stratified responses and create the human annotation interface

### Pipeline outputs

Data the pipeline produces. Committed to the repo so analysis is reproducible without re-running expensive steps.

- [`results/`](results/) — 324 experiment outputs (one JSON per conversation), structured as `<case>/<condition>/<strategy>/run_N.json`. Source of all primary leakage measurements.
- [`evals/`](evals/) — 324 GPT-4o-mini supplementary evaluations, parallel structure to `results/`. Source of contradictions, naturalness, and failure attribution numbers.
- [`charts/`](charts/) — the 6 PDF charts in the report

### Side analyses

Self-contained bundles that aren't part of the main pipeline.

- [`annotation/`](annotation/) — the 50-sample human validation evidence
  - [`annotate.html`](annotation/annotate.html) — annotation interface used by both annotators
  - [`sample_metadata.json`](annotation/sample_metadata.json) — the 50 stratified samples shown to annotators
  - [`annotation_annotator_1.json`](annotation/annotation_annotator_1.json), [`annotation_annotator_2.json`](annotation/annotation_annotator_2.json) — independent labels
  - [`kappa_results.json`](annotation/kappa_results.json) — inter-annotator κ=0.919, human-vs-GPT κ=0.208
- [`fareez/`](fareez/) — distributional comparison vs. real OSCE transcripts (Table 5)
  - [`fareez_comparison.py`](fareez/fareez_comparison.py), [`fareez_comparison.json`](fareez/fareez_comparison.json) — script and its output
  - [`README.md`](fareez/README.md) — Fareez 2022 dataset download and citation
  - `clean_transcripts/` — gitignored; download txt files from **[figshare](https://doi.org/10.6084/m9.figshare.16550013)** and place them here
- [`demo/`](demo/) — playground notebook and stitched figure
  - [`playground.ipynb`](demo/playground.ipynb) — interactive notebook (open in classic Jupyter, not VSCode or Colab)
  - [`stitch_playground.py`](demo/stitch_playground.py) — combines playground cell outputs into a single page
  - [`demo_figure.html`](demo/demo_figure.html) — composed by `stitch_playground.py`

### Course deliverables

- [`milestone_docs/`](milestone_docs/)
  - [`proposal.pdf`](milestone_docs/proposal.pdf) (also available on [Google Drive](https://docs.google.com/document/d/1tOcCJONUz_ntJvuhBHgy4a_0rhxdzq4RKYVVpoKXy80/edit?usp=sharing))
  - [`progress_report.pdf`](milestone_docs/progress_report.pdf) (also available on [Google Drive](https://docs.google.com/document/d/1X4RFFXo_oR_dbB5HivVlmhwtDHNzk51dQ9Ex1sdO_4A/edit?usp=sharing))
  - final report
  - [`demo_slides.pptx`](milestone_docs/demo_slides.pptx) (also available on [Google Drive](https://docs.google.com/presentation/d/1S6ZL4I1kWZGnstxX_ksZV60AL7DYXUaYPsCd0kc14bg/edit?usp=drive_link))
  - [`leak_inspection_report.md`](milestone_docs/leak_inspection_report.md) (manual evidence behind the GPT-4o-mini false-positive analysis)
- [`assets/`](assets/) — static images
  - [architecture diagram](assets/architecture.png)
  - [human annotator interface screenshot](assets/human_annotation_tool.png)
  - [demo figure screenshot](assets/demo_figure_screenshot.png)
  - [kernel selection screenshot](assets/kernel_selection.png)

### Project files

- [`README.md`](README.md) — this file
- [`requirements.txt`](requirements.txt) — Python dependencies
- [`LICENSE`](LICENSE) — MIT
- [`.gitignore`](.gitignore)


## Build the System
**1. Clone the repo and install Python dependencies:**
```bash
git clone https://github.com/wnc6/cant-leak.git
cd cant-leak

# Use a virtual environment so dependencies don't conflict with system Python
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
**2. Install and start Ollama** (keep running)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start local ollama server
ollama serve
```
**3. Pull the model** (in a new terminal)
```bash
# ~16 GB download - takes few minutes
ollama pull llama3.1:8b-instruct-fp16

# Smoke test → should respond
ollama run llama3.1:8b-instruct-fp16 "hello"
```
**4. Set the OpenAI API key** (for the supplementary GPT-4o-mini evaluation)
```bash
export OPENAI_API_KEY=sk-...
```
**5. Verify the build**
Run the smoke test to confirm Ollama is reachable, all six conditions instantiate, and each produces a non-empty response on a 10-turn cardiology conversation:

```bash
python3 tests/smoke_test.py
```
If this passes, the system is working end-to-end and you can proceed to [Deploy and Run the System](#deploy-and-run-the-system).

[`tests/test_planner.py`](tests/test_planner.py) and [`tests/test_pipeline.py`](tests/test_pipeline.py) provide finer-grained unit and integration tests; both run with `python3 tests/<name>.py`.

## Deploy and Run the System

### Run the Playground
**1. Register the venv as a Jupyter kernel**
```bash
pip install ipykernel
python3 -m ipykernel install --user --name cant-leak --display-name "cant-leak"
```
**2. Start Jupyter Notebook** (make sure Ollama is running in another terminal)
> VSCode or Colab would ***NOT*** work, due to styled outputs
```bash
jupyter notebook demo/playground.ipynb
```
**3. Select the `cant-leak` kernel**
![Kernel selection](assets/kernel_selection.png)

### Create the Demo Figure

<details>
<summary>Click to preview Demo Figure</summary>

![Preview](assets/demo_figure_screenshot.png)

</details>

After running the playground and saving your desired outputs:
```bash
# Combine playground cell outputs
python3 demo/stitch_playground.py demo/playground.ipynb -o demo/demo_figure.html
open demo/demo_figure.html
```

### Run a Single Experiment
Run one conversation (1 case × 1 condition × 1 strategy):
```bash
python3 run_experiment.py isolated_architecture gradual_escalation \
  --case cases/case_cardiology.json \
  --output /tmp/test.json
```
To see all available conditions and strategies:
```bash
python3 run_experiment.py --list
```

### Run All 324 Experiments
The committed [`results/`](results/) contains all 324 outputs, so you can skip this entirely and proceed to [Reproduce the Evaluation](#reproduce-the-evaluation). 

To regenerate from scratch (6 conditions × 6 strategies × 3 cases × 3 runs):
```bash
python3 run_all.py --runs 3
```
To preview what would run without executing:
```bash
python3 run_all.py --runs 3 --dry-run
```
The script writes incrementally; a Ctrl+C interruption preserves completed conversations and `--resume` skips them on a re-run.

## External Software Built On

| Tool | Used for | Notes |
|---|---|---|
| [Ollama](https://ollama.ai) | Local model server | Avoids API costs and rate limits across 324 experiments. |
| [Llama 3.1 8B Instruct (FP16)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | All six experimental conditions (planner, generator, verifier, baselines) | 8B chosen deliberately — if isolation works at this scale, it works at larger scale (§2.1 of report). |
| GPT-4o-mini via [OpenAI API](https://platform.openai.com/docs/api-reference) | Supplementary post-hoc evaluation only | Never used inline during experiments. Treated as supplementary because human validation showed ~27% recall on leak detection. |
| Python deps: [requests](https://pypi.org/project/requests/), [openai](https://pypi.org/project/openai/), [matplotlib](https://matplotlib.org/), [jupyter](https://jupyter.org/), [ipykernel](https://github.com/ipython/ipykernel) | HTTP client, OpenAI SDK, plotting, notebook, kernel registration | Versions in [`requirements.txt`](requirements.txt). |

No other models, embeddings, or external services are used in the experimental pipeline. External datasets and schemas (Fareez 2022, MediTOD CMAS) are documented in [Datasets and Benchmarks](#datasets-and-benchmarks).

## Reproduce the Evaluation

The report's claims rest on five reproducible artifacts. Each can be re-derived from this repository.

### 1. Primary leakage measure (Table 2, RQ1, RQ4)

**Requires:** [`results/`](results/) (committed). No API calls, no LLM evaluator.

The report's headline numbers come from deterministic phrase matching, computed by the runner during each conversation and stored in `summary.leak_count`.

```bash
python3 summarize_runs.py results/
```

<details>
<summary>Click to view expected output</summary>

```
Loaded 324 result files

=================================================================
LEAKAGE BY CONDITION (mean ± SD across all runs)
=================================================================
Condition                    N   Mean     SD  Min  Max
-----------------------------------------------------------------
Naive Prompting             54   4.13   2.07    0    8
Structured Prompting        54   5.17   2.42    1   11
Self-Monitoring             54   4.22   2.30    0    9
Isolated Architecture       54   0.00   0.00    0    0
No-Isolation Ablation       54   0.81   1.10    0    3
No-Verifier Ablation        54   0.04   0.19    0    1

=====================================================================================
LEAKAGE BY CONDITION × STRATEGY (mean ± SD)
=====================================================================================
Condition                     Direct   Rephrase    Emotion  Authority   Escalate      Logic
-------------------------------------------------------------------------------------
Naive Prompting            3.4±2.2   3.4±1.2   3.4±2.2   5.1±2.0   5.0±1.9   4.3±2.4 
Structured Prompting       4.0±1.9   3.3±1.5   5.6±2.3   5.6±2.1   6.7±2.5   5.9±2.8 
Self-Monitoring            2.4±1.7   3.4±1.0   3.3±2.7   5.6±1.7   5.0±2.1   5.6±2.6 
Isolated Architecture      0.0±0.0   0.0±0.0   0.0±0.0   0.0±0.0   0.0±0.0   0.0±0.0 
No-Isolation Ablation      1.1±1.5   0.4±0.5   1.0±1.0   1.0±1.2   1.3±1.3   0.0±0.0 
No-Verifier Ablation       0.0±0.0   0.0±0.0   0.1±0.3   0.0±0.0   0.1±0.3   0.0±0.0 

=================================================================
LEAKAGE BY CONDITION × CASE (mean ± SD)
=================================================================
Condition                     Cardio       Resp         GI
-----------------------------------------------------------------
Naive Prompting            3.2±1.6   2.9±1.5   6.2±1.3 
Structured Prompting       4.6±2.1   3.9±1.5   6.9±2.5 
Self-Monitoring            3.9±2.1   3.0±1.9   5.8±2.1 
Isolated Architecture      0.0±0.0   0.0±0.0   0.0±0.0 
No-Isolation Ablation      0.5±0.5   0.2±0.4   1.7±1.4 
No-Verifier Ablation       0.0±0.0   0.1±0.2   0.1±0.2 

=================================================================
MANN-WHITNEY U: Each condition vs Isolated Architecture
=================================================================
Note: the isolated condition has zero variance by construction,
so these tests are degenerate. The magnitude of separation is
the more meaningful comparison (see report Table 2 caption).
p-values shown for reference only.

Naive Prompting           4.13±2.07 vs 0.00±0.00  U=27  p=0.000000  ***
Structured Prompting      5.17±2.42 vs 0.00±0.00  U=0  p=0.000000  ***
Self-Monitoring           4.22±2.30 vs 0.00±0.00  U=27  p=0.000000  ***
No-Isolation Ablation     0.81±1.10 vs 0.00±0.00  U=783  p=0.000034  ***
No-Verifier Ablation      0.04±0.19 vs 0.00±0.00  U=1404  p=0.740040  ns

=================================================================
STRUCTURED vs NAIVE (suggestive only — see report §2.2)
=================================================================
The report flags this as a suggestive direction, not an
established finding. Caveats: borderline p-value that won't
survive multiple-comparisons correction; structured bundles
three changes from naive; direction inconsistent across cases.

Naive:      4.13 ± 2.07 (n=54)
Structured: 5.17 ± 2.42 (n=54)
Mann-Whitney U: 1123, p = 0.0396 (uncorrected)

=================================================================
DISCLOSURE RATE (architecture conditions only)
=================================================================
Isolated Architecture     73.2% ± 17.4%
No-Isolation Ablation     73.1% ± 18.1%
No-Verifier Ablation      72.5% ± 18.8%
```

</details>


### 2. Supplementary GPT-4o-mini evaluation (Tables 4, 7, naturalness)

**Requires:** [`evals/`](evals/) (committed) for aggregation, or [`results/`](results/) + OpenAI API key for regeneration.

GPT-4o-mini scores contradictions (P2L, L2L), naturalness, and failure attribution. Treated as supplementary because human annotation showed ~27% recall on leak detection (§3.6 of the report).

```bash
# Aggregate the committed evaluations
python3 summarize_evals.py evals/

# Or regenerate from scratch
python3 evaluate.py results/ --output evals/
```
<details>
<summary>Click to view expected output</summary>

```
Loaded 324 eval files

======================================================================
TABLE 4: Consistency Metrics (mean ± SD across all 3 runs)
======================================================================
Condition                       Contrad.            L2L            P2L
----------------------------------------------------------------------
Naive Prompting             1.9 ± 2.2      2.6 ± 1.6      9.2 ± 5.9  
Structured Prompting        1.5 ± 2.1      2.1 ± 1.7     11.6 ± 10.4 
Self-Monitoring             1.6 ± 2.4      1.7 ± 1.3      5.5 ± 5.4  
Isolated Architecture       0.6 ± 1.6      1.6 ± 1.1      3.7 ± 4.8  
No-Isolation Abl.           0.5 ± 1.1      1.3 ± 1.2      6.4 ± 6.0  
No-Verifier Abl.            1.1 ± 2.5      1.6 ± 1.3      3.7 ± 4.6  

N = 54 per condition

======================================================================
NATURALNESS (mean ± SD across all 3 runs)
======================================================================
Condition                     Mean       SD     N
----------------------------------------------------------------------
Naive Prompting               3.43     0.14    54
Structured Prompting          3.30     0.16    54
Self-Monitoring               3.37     0.14    54
Isolated Architecture         3.38     0.15    54
No-Isolation Abl.             3.25     0.13    54
No-Verifier Abl.              3.36     0.13    54

======================================================================
TABLE 7: Failure Attribution (totals across all 3 runs)
======================================================================
Generator errors:    760 (48%)
Planner errors:      792 (50%)
Verifier misses:      24 (2%)
Total:              1576
```

</details>

### 3. Human validation (§3.6, Table 6)

**Requires:** [`annotation/`](annotation/) and [`evals/`](evals/) (both committed).

Two annotators independently labeled the same 50 stratified responses from run 1; both label sets were then compared against GPT-4o-mini.

```bash
python3 annotate.py compute \
    annotation/annotation_annotator_1.json \
    annotation/annotation_annotator_2.json
```

<details>
<summary>Click to view expected output</summary>

```
==================================================
Inter-Annotator Agreement
==================================================
  human1 vs human2
  Samples:    50
  Agreement:  96.0%
  Kappa:      0.919 (Almost perfect)
  human1: 22 leaks, 2 unsure
  human2: 22 leaks, 5 unsure

==================================================
Human vs GPT-4o-mini
==================================================
  human1 vs GPT-4o-mini: κ = 0.208
  human2 vs GPT-4o-mini: κ = 0.208
  GPT-4o-mini: 6 leaks (12.0%)

  Saved: annotation/kappa_results.json
```

</details>

To re-run the annotation interface on a fresh sample (additionally requires [`results/`](results/) and two human annotators):

```bash
python3 annotate.py sample
```
This outputs [`annotation/annotate.html`](annotation/annotate.html) and [`annotation/sample_metadata.json`](annotation/sample_metadata.json). Send the HTML to two annotators; each opens it in a browser, labels all 50 samples, and downloads a JSON file. Save the downloads as [`annotation/annotation_annotator_1.json`](annotation/annotation_annotator_1.json) and [`annotation/annotation_annotator_2.json`](annotation/annotation_annotator_2.json), then re-run the `compute` command above.

<details>
<summary>Click to preview annotate.html</summary>

![Preview](assets/human_annotation_tool.png)

</details>

### 4. Report charts (Figures 2–7)

**Requires:** [`results/`](results/) (committed).

```bash
python3 generate_charts.py results/ --format pdf
```
Outputs are saved to [`charts/`](charts/). Figure 1 ([`assets/architecture.png`](assets/architecture.png)) was drawn manually and is not produced by this pipeline.

### 5. Distributional comparison vs. real OSCE transcripts (Table 5)

**Requires:** [`results/`](results/), [`evals/`](evals/), and the Fareez 2022 dataset (external). An OpenAI API key is optional — without it, length and hedging numbers still compute but naturalness is skipped.

**Setup:** Download the Fareez transcripts from [figshare](https://doi.org/10.6084/m9.figshare.16550013) and place .txt files in `fareez/clean_transcripts/`. See [`fareez/README.md`](fareez/README.md) for citation details.

Compares 50 sampled isolated-architecture responses against 50 sampled real OSCE responses on naturalness, response length, and hedging frequency.

```bash
python3 fareez/fareez_comparison.py
```
<details>
<summary>Click to view expected output</summary>

```
Loading Fareez transcripts...
  Sampled 50 Fareez patient responses
Loading our responses...
  Sampled 50 of our patient responses

=== Naturalness Comparison ===
  Skipping (no OPENAI_API_KEY). Set it to score Fareez responses.

=== Response Length Comparison ===
  Fareez mean length: 14.3 words
  Ours mean length:   46.9 words
  Difference:         +32.6 words

=== Hedging Frequency Comparison ===

  Per-condition hedging (all results):
  Fareez: 20% of responses contain hedging
  Naive Prompting: 23% of responses contain hedging (avg 0.27 per response)
  Structured Prompting: 15% of responses contain hedging (avg 0.20 per response)
  Self-Monitoring: 9% of responses contain hedging (avg 0.10 per response)
  Isolated Architecture: 17% of responses contain hedging (avg 0.20 per response)
  No-Isolation Ablation: 11% of responses contain hedging (avg 0.12 per response)
  No-Verifier Ablation: 19% of responses contain hedging (avg 0.22 per response)

Results saved to /Users/wenni/Documents/GitHub/cant-leak/fareez/fareez_comparison.json
```

</details>

## Datasets and Benchmarks

This project contributes one OSCE benchmark with structured disclosure annotations ([`cases/`](cases/)) and three datasets supporting the evaluation ([`results/`](results/), [`evals/`](evals/), [`annotation/`](annotation/)). Two external resources informed case authoring: [Fareez et al. 2022](https://doi.org/10.6084/m9.figshare.16550013) for structural realism, and [MediTOD](https://aclanthology.org/2024.emnlp-main.936/) for diagnostic-slot coverage.

### Bundled with this repository

- **[`cases/`](cases/)** — three clinical cases authored as a benchmark for disclosure-control evaluation: [cardiology](cases/case_cardiology.json), [respiratory](cases/case_respiratory.json), [GI](cases/case_gi.json).
  - *Per fact:* natural-language disclosure condition, deterministic unlock keywords, leak phrases for verifier matching, CMAS-style symptom attributes.
  - *Validated against:* [Fareez et al. 2022](https://doi.org/10.6084/m9.figshare.16550013) (structural realism), and [MediTOD](https://aclanthology.org/2024.emnlp-main.936/) (diagnostic-slot coverage).
  - *License:* MIT.
- **[`results/`](results/)** — 324 conversation transcripts produced by running the full experiment matrix. Source for all primary leakage measurements (Table 2, Figures 2–7).
- **[`evals/`](evals/)** — 324 GPT-4o-mini supplementary evaluations of those transcripts. Source for contradiction, naturalness, and failure-attribution numbers (Tables 4, 7).
- **[`annotation/`](annotation/)** — 50 stratified responses with independent labels from two human annotators, plus GPT-4o-mini predictions and the Cohen's κ computation. Source for the human validation evidence (§3.6).

### External resources

- **Fareez et al. 2022** — 272 simulated patient–physician OSCE transcripts (predominantly respiratory) recorded by medical residents and senior medical students, with manually corrected text transcripts.
  - *Used as:* (a) a structural reference for case authoring — ensuring authored cases match real OSCE structure (chief complaint, HPI, family/social history, ROS) rather than being LLM-fabricated; (b) a naturalness comparison benchmark (Table 5).
  - *Access:* Download from [figshare](https://doi.org/10.6084/m9.figshare.16550013); cite the original authors.

- **MediTOD ([EMNLP 2024](https://aclanthology.org/2024.emnlp-main.936/))** — the Clinical Medical Agent Schema (CMAS), a structured representation of medical dialogue across 11 diagnostic slots: chief complaint · history of present illness · past medical history · medications · allergies · family history · social history · review of systems · physical exam · assessment · plan.
  - *Used to:* validate that each authored case provides at least one informative fact per slot — ensuring authored cases aren't missing material a real clinician would expect.

## Comparison Systems

The isolated architecture is evaluated against five alternative systems implemented in this repository [`src/conditions.py`](src/conditions.py). Three are prompting-based baselines representing the standard approaches in the LLM-simulated-patient literature; two are ablations of the proposed architecture isolating individual component contributions.

| Condition | Mechanism |
|---|---|
| Naive prompting | Full case in system prompt + "do not reveal these facts" |
| Structured prompting | Per-fact disclosure rules, few-shot hedging examples, chain-of-thought |
| Self-monitoring | Generate response, self-check for leaks, regenerate if needed |
| **Isolated architecture (this work)** | Planner → content brief → generator + verifier |
| No-isolation ablation | Same pipeline as isolated, but generator sees full case |
| No-verifier ablation | Same as isolated, but without the verifier |

All five conditions use the same Llama 3.1 8B Instruct model, the same student agent strategies, and the same clinical cases as the isolated architecture, ensuring fair comparison. Only the disclosure-control mechanism differs. Run any of them via [`run_experiment.py`](run_experiment.py) (see [Run a Single Experiment](#run-a-single-experiment)).

Architectural antecedents discussed in the report's Related Work but not directly compared as runnable systems: [EvoPatient (ACL 2025)](https://aclanthology.org/2025.acl-long.846/), [AIPatient (2024)](https://arxiv.org/abs/2409.18924), [Abdulhai et al. (NeurIPS 2025)](https://arxiv.org/abs/2511.00222).


## AI Tools Used in Development

- **Claude (claude.ai)** — drafting, debugging, and refactoring code and docstrings; all code reviewed and all reported numbers independently verified.
- **Perplexity** — literature search; cited sources verified independently against original publications.
- **GPT-4o-mini** — pipeline evaluator [`evaluate.py`](evaluate.py), not a development tool; treated as supplementary because human validation showed ~27% recall on leak detection.

Clinical case content, leak phrases, and unlock keywords were authored manually.

## License

MIT — see [`LICENSE`](LICENSE). The cases, code, and results are reusable for academic and commercial work with attribution.

