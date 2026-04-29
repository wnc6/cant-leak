# You Can't Leak What You Don't Know
Information-isolated disclosure architecture for controlling what LLM dialogue agents reveal under pressure.

## Repository Layout

### Architecture and benchmark

The system itself, plus the cases it operates on. These are the inputs to everything else.

- [`src/`](src/) — the [`planner`](src/planner.py), [`generator`](src/generator.py), [`verifier`](src/verifier.py), [`student agent`](src/student_agent.py), and [`six experimental conditions`](src/conditions.py) (naive, structured, self-monitoring, isolated, two ablations)
- [`cases/`](cases/) — three OSCE clinical cases ([`cardiology`](cases/case_cardiology.py), [`respiratory`](cases/case_respiratory.py), [`GI`](cases/case_gi.py)) with disclosure conditions, unlock keywords, leak phrases, and CMAS symptom attributes
- [`tests/`](tests/) — unit tests for the [`planner`](tests/test_planner.py), [`end-to-end pipeline integration`](tests/test_pipeline.py), and [`a fast smoke test`](tests/smoke_test.py) (fixtures in [`tests/fixtures/`](tests/fixtures/))

### Main pipeline

The scripts that produce the report's primary numbers. Run in this order to reproduce.

- [`run_experiment.py`](run_experiment.py) — run a single conversation (1 case × 1 condition × 1 strategy)
- [`run_all.py`](run_all.py) — run the full 324-experiment matrix (3 runs × 3 cases × 6 conditions × 6 strategies)
- [`evaluate.py`](evaluate.py) — supplementary GPT-4o-mini evaluation (contradictions, naturalness, failure attribution)
- [`summarize_runs.py`](summarize_runs.py) — aggregate `results/` into the report's primary leakage tables (Table 2, RQ1, RQ4)
- [`summarize_evals.py`](summarize_evals.py) — aggregate `evals/` into supplementary tables (Tables 4, 7, naturalness)
- [`generate_charts.py`](generate_charts.py) — produce the 6 charts used in the report
- [`annotate.py`](annotate.py) — sample 50 stratified responses and create the human annotation interface

### Pipeline outputs

Data the pipeline produces. Committed to the repo so analysis is reproducible without re-running expensive steps.

- [`results/`](results/) — 324 experiment outputs (one JSON per conversation), structured as `<case>/<condition>/<strategy>/run_N.json`
- [`evals/`](evals/) — 324 GPT-4o-mini supplementary evaluations, parallel structure to `results/`
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
  - `clean_transcripts/` — gitignored; stores txt files downloaded from **[HERE](https://doi.org/10.6084/m9.figshare.16550013)**
- [`demo/`](demo/)
  - [`playground.ipynb`](demo/playground.ipynb) — interactive notebook (open in classic Jupyter, not VSCode or Colab)
  - [`stitch_playground.py`](demo/stitch_playground.py) — combines playground cell outputs into a single page
  - [`demo_figure.html`](demo/demo_figure.html) — composed by `stitch_playground.py`

### Course deliverables

- [`milestone_docs/`](milestone_docs/) — [`proposal`](milestone_docs/proposal.pdf), [`progress report`](milestone_docs/progress_report.pdf), final report, demo slides, and [`a leak inspection report`](milestone_docs/leak_inspection_report.md) (manual evidence behind the GPT-4o-mini false-positive analysis)
- [`assets/`](assets/) — static images ([`architecture diagram`](assets/architecture.png), [`human annotator tool screenshot`](assets/human_annotation_tool.png))

### Project files

- [`README.md`](README.md) — this file
- [`requirements.txt`](requirements.txt) — Python dependencies
- [`LICENSE`](LICENSE) — MIT
- [`.gitignore`](.gitignore)

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) for local LLM inference
- 16+ GB disk, 12+ GB RAM for `llama3.1:8b-instruct-fp16`
- OpenAI API key (only for the supplementary GPT-4o-mini evaluation)

### Setup
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
brew install ollama
ollama serve
```
**3. Pull the model** (in a new terminal)
```bash
# ~16 GB download - takes few minutes
ollama pull llama3.1:8b-instruct-fp16

# smoke test - should respond
ollama run llama3.1:8b-instruct-fp16 "hello"
```

### Run the playground
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

### Combine playground cell outputs into [demo_figure.html](demo/demo_figure.html)

> ***Save changes*** to disk first
```bash
python3 demo/stitch_playground.py demo/playground.ipynb -o demo/demo_figure.html
open demo_figure.html
```