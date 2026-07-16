# DS9 v2 — Training Guide

How to (re)train the v2 models and rebuild the serving artifacts from
scratch. All wall-clock numbers below were measured on an Apple M3 Ultra
(macOS, torch 2.13, MPS) training the full North-America corpus; the same
sequence works on CUDA and plain CPU (`--device auto` picks the best
available backend).

The resulting NA models are committed under `models-v2/` — you only need
this guide to retrain (new corpus, new hyperparameters, new region such as
EU, or after importer changes).

## 0. Setup

```bash
git clone git@github.com:Klebert-Engineering/deep-spell-9.git && cd deep-spell-9
python3.12 -m venv .venv && source .venv/bin/activate   # or: uv venv .venv --python 3.12
pip install torch                 # CUDA/MPS build as appropriate for the box
pip install -e .[dev]
pytest                            # ~20 s, must be green (smoke-trains on CPU)
```

## 1. Data

The full training corpus and the licensed NDS databases are **not in git**:

* `deepspell_data_north_america_nozip_v2.tsv` — 359 MB, 9,492,749 rows
* `RoadFTS5_USA_unzipped.nds` — 1.8 GB, optional, for NDS lookup testing

Both live in Klebert Engineering internal storage (OneDrive,
`deep-spell-corpora` folder); place them in `corpora/` (gitignored).

Build the gazetteer (~40 s on M3 Ultra, ~5 min on an older CPU box):

```bash
ds9 data import-legacy-tsv corpora/deepspell_data_north_america_nozip_v2.tsv corpora/na-v2.sqlite
ds9 data info corpora/na-v2.sqlite     # expect 9,492,749 tokens, ROAD-dominated
```

Optional GeoNames overlay (open data, CC-BY; adds worldwide CITY/STATE/COUNTRY):

```bash
ds9 data fetch-geonames corpora/geonames
ds9 data import-geonames corpora/geonames corpora/geonames.sqlite
```

## 2. Train

Hyperparameters mirror the v1 model capacity. The three trainings are
independent processes and can run **concurrently** on one GPU — on the M3
Ultra this costs no per-model throughput and finishes everything in ~3 h:

```bash
ds9 train tagger    corpora/na-v2.sqlite corpora/grammar-address-na.json models-v2/tagger-na \
    --steps 20000 --batch-size 512 --hidden 128 --layers 2 --device auto     # ~1.9 h

ds9 train completer corpora/na-v2.sqlite corpora/grammar-address-na.json models-v2/completer-na \
    --steps 40000 --batch-size 512 --hidden 256 --layers 2 --device auto     # ~2.9 h

ds9 train encoder   corpora/na-v2.sqlite models-v2/encoder-na \
    --steps 20000 --batch-size 512 --hidden 128 --device auto                # ~1.3 h
```

Notes:

* **The loop saves only at the end of a run.** On macOS wrap long trainings
  in `caffeinate -is ...` so the machine cannot sleep mid-run; on Linux make
  sure the box stays up.
* Batch sampling streams from SQLite in the main process (~13k samples/s on
  the full gazetteer), so the GPU, not sampling, is the bottleneck at these
  batch sizes. Bump `--batch-size` until the GPU is saturated.
* Val metrics are logged every 250 steps and stored in the model card JSON
  under `training`.

## 3. Evaluate

```bash
ds9 eval tagger    models-v2/tagger-na.json    corpora/na-v2.sqlite corpora/grammar-address-na.json
ds9 eval completer models-v2/completer-na.json corpora/na-v2.sqlite corpora/grammar-address-na.json
```

Targets (≈ v1 shipped quality) and the results of the 2026-07-16 run:

| Metric | Target | Achieved |
|---|---|---|
| tagger `unit_accuracy` (truncated phrases) | ≳ 0.95 | 0.958 |
| completer `exact_top6` | ≳ 0.5 | 0.549 |
| encoder `val_retrieval_top1` | — | 1.0 |

If a run falls short, extend `--steps` before touching hyperparameters.

## 4. Build serving artifacts

All reproducible from the gazetteer + encoder; they stay out of git:

```bash
ds9 data build-lookup   corpora/na-v2.sqlite corpora/na-v2-lookup.sqlite     # FTS5 index, ~3 min, 1.0 GB
ds9 data build-symspell corpora/na-v2.sqlite corpora/na-v2-symspell.txt      # 3.0 M terms, ~10 s
# optional neural corrector space (~5 min):
ds9 data build-embedding-space models-v2/encoder-na.json corpora/na-v2.sqlite corpora/na-v2-space
ds9 eval corrector corpora/na-v2.sqlite --dictionary corpora/na-v2-symspell.txt
#   2026-07-16 run: recall@1 0.95, recall@3 0.97
```

## 5. Serve

```bash
ds9 serve configs/service-v2.example.json        # http://localhost:8091
```

`configs/service-v2-nds.example.json` runs the same service against a
licensed NDS database (stock sqlite3 reads `RoadFTS5_*` fine) with the
embedding corrector — verified against `RoadFTS5_USA_unzipped.nds`,
including alt-name resolution; injection probes return empty.

## 6. Wrap up

* Commit the model cards + weights (a few MB per model) with the eval
  numbers in the commit message; gazetteer/lookup/symspell/embedding
  artifacts are reproducible and stay out of git.
* Optional side-by-side with the legacy v1 demo (#52):
  `docker run --rm -p 8092:8091 ghcr.io/klebert-engineering/ds9:2023.1 //ds9/serve.bash`
