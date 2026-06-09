# DS9 v2 — Training Runbook (GPU box)

Everything below was prepared and CPU-validated on the WSL2 machine; this is
the exact sequence to produce real North-America models on a machine with GPU
support (CUDA or Apple-Silicon MPS — `--device auto` picks it up).

## 0. Setup

```bash
git clone git@github.com:Klebert-Engineering/deep-spell-9.git && cd deep-spell-9
git checkout ds9-v2
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch                  # CUDA/MPS build as appropriate for the box
uv pip install -e .[dev]
pytest                                # ~30 s, must be green
```

Copy the data that is not in git from the old machine into `corpora/`:

* `deepspell_data_north_america_nozip_v2.tsv` (377 MB — the training corpus)
* `RoadFTS5_USA_unzipped.nds` / `RoadFTS5_California_unzipped.nds` (optional,
  for NDS lookup parity testing)

## 1. Build the gazetteer (~5 min, CPU)

```bash
ds9 data import-legacy-tsv corpora/deepspell_data_north_america_nozip_v2.tsv corpora/na-v2.sqlite
ds9 data info corpora/na-v2.sqlite     # expect ~9.49 M tokens, ROAD-dominated
```

Optional GeoNames overlay (open data, CC-BY; adds worldwide CITY/STATE/COUNTRY):

```bash
ds9 data fetch-geonames corpora/geonames
ds9 data import-geonames corpora/geonames corpora/geonames.sqlite
```

## 2. Train

Suggested starting points (hyperparameters mirror the v1 capacity; bump
`--batch-size` until the GPU is saturated):

```bash
ds9 train tagger    corpora/na-v2.sqlite corpora/grammar-address-na.json models-v2/tagger-na \
    --steps 20000 --batch-size 512 --hidden 128 --layers 2 --device auto

ds9 train completer corpora/na-v2.sqlite corpora/grammar-address-na.json models-v2/completer-na \
    --steps 40000 --batch-size 512 --hidden 256 --layers 2 --device auto

ds9 train encoder   corpora/na-v2.sqlite models-v2/encoder-na \
    --steps 20000 --batch-size 512 --hidden 128 --device auto
```

Notes:

* Batch sampling streams from SQLite in the main process. On the WSL2 box it
  sustains ~13k phrase samples/s on the full 9.5 M-token gazetteer, so the
  GPU, not sampling, is the bottleneck at these batch sizes.
* Tagger/completer val metrics are logged every 250 steps and stored in the
  model card JSON under `training`.

## 3. Evaluate

```bash
ds9 eval tagger    models-v2/tagger-na.json    corpora/na-v2.sqlite corpora/grammar-address-na.json
ds9 eval completer models-v2/completer-na.json corpora/na-v2.sqlite corpora/grammar-address-na.json
```

Reference targets (v1 shipped quality, measured informally): tagger unit
accuracy ≳ 0.95 on truncated phrases; completer exact_top6 ≳ 0.5. If short,
extend steps before touching hyperparameters.

## 4. Build serving artifacts

```bash
ds9 data build-lookup   corpora/na-v2.sqlite corpora/na-v2-lookup.sqlite     # FTS5 index (~4 min)
ds9 data build-symspell corpora/na-v2.sqlite corpora/na-v2-symspell.txt      # default corrector (~20 s)
# optional neural corrector space:
ds9 data build-embedding-space models-v2/encoder-na.json corpora/na-v2.sqlite corpora/na-v2-space
ds9 eval corrector corpora/na-v2.sqlite --dictionary corpora/na-v2-symspell.txt
```

## 5. Serve

```bash
ds9 serve configs/service-v2.example.json        # http://localhost:8091
```

`configs/service-v2-nds.example.json` shows the same service running against
a licensed NDS database (verified to work with stock sqlite3) and the
embedding corrector.

## 6. Wrap up

* Commit the model cards + weights (small: a few MB per model) or publish
  them as a release artifact; gazetteer/lookup/symspell artifacts are
  reproducible and stay out of git.
* Compare side-by-side with the legacy demo:
  `docker run --rm -p 8092:8091 ghcr.io/klebert-engineering/ds9:2023.1 //ds9/serve.bash`
