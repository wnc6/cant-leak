# Fareez Distributional Comparison

This folder contains the script and results for the distributional
comparison against real OSCE patient transcripts (RQ3 in the report).

## Files

- `fareez_comparison.py` — script that samples 50 isolated-architecture
  responses and 50 real OSCE responses, then computes naturalness,
  response length, and hedging frequency for each
- `fareez_comparison.json` — output of the script (numbers in Table 5)
- `clean_transcripts/` — sample of cleaned Fareez transcripts (NOT
  committed to this repo; see Setup below)

## Setup

The Fareez transcripts are not redistributed in this repository. To
reproduce the comparison:

1. Download the Fareez 2022 OSCE transcripts from
   https://figshare.com/articles/dataset/MedDG/19514055
2. Move the dataset's "clean_transcripts" folder (containing .txt files) into fareez/
3. Run `python3 fareez/fareez_comparison.py` from the repo root
