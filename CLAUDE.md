# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep-Spell-9 v2 is a neural auto-completion and matching system for geographic queries (PyTorch 2, Python 3.10+). Three models work in a pipeline: Tagger (BiLSTM token classification), Completer (class-conditioned char LM with beam search) and TokenEncoder (contrastive, backs the embedding-space corrector). A SymSpell corrector and SQLite FTS5 lookup complete the stack.

## Key Commands

```bash
# Setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch && pip install -e .[dev]

# Quality gates (both must pass; CI runs exactly these)
ruff check src tests
pytest                    # 52 tests, ~20 s, smoke-trains tiny models on CPU

# The ds9 CLI is the single entry point
ds9 data import-legacy-tsv|import-geonames|build-lookup|build-symspell|build-embedding-space|info
ds9 train tagger|completer|encoder   # see docs/v2-training-runbook.md
ds9 eval tagger|completer|corrector
ds9 serve configs/service-v2.example.json    # FastAPI + browser UI on :8091
ds9 demo <tagger.json> <completer.json>      # terminal REPL
```

## Architecture

- `src/deepspell/` — the v2 package
  - `gazetteer/` — single SQLite data artifact + importers (legacy TSV, GeoNames)
  - `sampling/` — phrase grammar + typo corruption (v1-JSON-compatible, seeded)
  - `models/` — Tagger / Completer / TokenEncoder; persisted as model card `.json` + `.pt`
  - `decode/` — Python beam search; beams never cross a token-class boundary
  - `correct/` — symspellpy backend (default) and embedding-space backend
  - `lookup/` — parameterized FTS5 queries; supports self-built indexes and licensed NDS databases
  - `service/` — FastAPI app (`/healthz`, `/api/complete`, `/api/lookup`) + static UI
  - `train/` — training loops with validation; **models are saved only at the end of a run** (wrap long trainings in `caffeinate -is` on macOS)
- `models-v2/` — trained NA models (committed; eval numbers in the model cards and commit messages)
- `modules/`, `models/`, root `*.py` scripts, `serve.bash` — **legacy v1 (TF 1.9), kept for reference only**; do not extend; the runnable v1 demo is the published docker image `ghcr.io/klebert-engineering/ds9:2023.1`

## Data

The full NA training corpus (`deepspell_data_north_america_nozip_v2.tsv`, 9.49 M rows) and licensed NDS databases are not in git — they live in Klebert Engineering internal storage (OneDrive, `deep-spell-corpora`) and belong in `corpora/` (gitignored). Only `corpora/deepspell_minimal.tsv` and the grammar JSONs are checked in; the test suite needs nothing else.

## Development Notes

1. Retraining, hyperparameters and expected metrics: `docs/v2-training-runbook.md`. Design rationale: `docs/modernization-proposal.md`.
2. Tests must stay CPU-only (`device="cpu"` in fixtures) — GitHub's macos-14 runners advertise MPS but cannot allocate on it.
3. Run pytest bare (not only `python -m pytest`) before pushing; CI uses `pytest -x` and depends on `pythonpath = ["."]` in pyproject.
4. No AI mentions in commit messages or PR descriptions.
5. Gazetteer/lookup/symspell/embedding artifacts are reproducible — keep them out of git; model cards + weights are committed.
