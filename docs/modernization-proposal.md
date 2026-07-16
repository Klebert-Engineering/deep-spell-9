# Deep-Spell-9 — Codebase Analysis & Modernization Proposal

*Status: draft for discussion — 2026-06-09*

This document is the result of a full read of the current codebase (every module under
`modules/`, all entry scripts, configs, the web UI, and the data assets on disk). Part 1
describes how the system actually works today and what state it is in. Part 2 proposes a
modernized redesign that keeps the use case identical: **interactive neural classification,
auto-completion, spell-correction and database lookup of geographic queries.**

---

## Part 1 — Analysis of the current system

### 1.1 What the system does, end to end

A user types a free-form geographic query ("los angeles calif…"). On every keystroke the
browser UI calls `GET /extrapolate?s=<query>`, which runs a four-stage pipeline
(`modules/deepspell_service/views.py:65`):

1. **Discriminate** — a bidirectional LSTM tagger (`DSLstmDiscriminator`) assigns each
   character a probability distribution over token classes
   (`CITY`, `STATE`, `COUNTRY`, `ROAD`, `EOL`). A Viterbi-lite post-process
   (`extract_best_class_sequence`) picks the best class per whitespace-delimited token
   unit by cumulative log-prob.
2. **Extrapolate** — a class-conditioned character-level LSTM language model
   (`DSLstmExtrapolator`) continues the query. Input characters are "2-hot" encoded
   (one-hot charset ⊕ one-hot class from step 1). A custom beam search (beam=6) is
   implemented *inside the TF graph* as a `tf.while_loop`; beams terminate when the
   predicted class switches, so a completion never crosses a token-class boundary.
3. **Correct** — each tokenized class segment is spell-checked. The neural path encodes
   the token with a variational LSTM autoencoder (`DSVariationalLstmAutoEncoder`,
   8-dim latent) and queries a pre-built `scipy.spatial.cKDTree` of all corpus token
   embeddings (k·4 nearest neighbours, re-ranked by Damerau-Levenshtein). The baseline
   path is a SymSpell delete-variant DAWG.
4. **Lookup** — the browser separately calls `GET /lookup?CITY=..&ROAD=..`, which builds
   an FTS5 `MATCH` query against an NDS SQLite database, scoping each token to its
   class-specific column (`criterionA..D`), grouped by the most specific criterion.

The three models are trained on *synthetic phrases*: a probabilistic grammar
(`corpora/grammar-address-na.json`) samples a token plus its hierarchical
ancestors/descendants (ROAD→CITY→STATE→COUNTRY) into randomized-order phrases
("Los Angeles Main St", "California Los Angeles", …), optionally corrupted with random
edits (insert/delete/substitute/transpose, count drawn from a normal distribution).
The discriminator additionally trains on random prefix truncations; the autoencoder
trains to denoise corrupted single tokens.

### 1.2 Component map

| Component | Location | Role |
|---|---|---|
| `DSFeatureSet` | `deepspell/featureset.py` | charset + class "2-hot" numpy encoding, padding, truncation, corruption hooks |
| `DSCorpus` / `DSToken` | `deepspell/corpus.py`, `grammar.py` | loads TSV/NDS into an in-RAM parent/child token graph; batch generator |
| `DSGrammar` | `deepspell/grammar.py` | JSON-defined random-sequence rules + corruption model |
| Models (inference) | `deepspell/models/` | TF1 graphs: tagger, LM+in-graph beam search, VAE encoder |
| Optimizers (training) | `deepspell_optimization/models/` | mixin subclasses adding loss/`RMSProp`/training loop |
| `DSTokenLookupSpace` | `deepspell/token_lookup_space.py` | kd-tree kNN over token embeddings, DL re-rank, pickle persistence |
| `DSFtsDatabaseConnection` | `deepspell/ftsdb.py` | FTS5 MATCH query builder against NDS SQLite |
| Baselines | `deepspell/baseline/` | SymSpell DAWG corrector, trie-based completer |
| Service | `deepspell_service/` | Flask app, global model singletons, jQuery UI (`templates/index.html`) |
| Entry scripts | repo root | `train-*.py` (hardcoded config), `eval-*.py` (argparse), `predict.py`/`match.py` (REPLs), `encode.py` |

Model artifacts are a JSON "model card" (hyperparameters + featureset) plus a TF1
checkpoint triplet; the JSON path is the universal reference.

### 1.3 Data assets actually available

| Asset | Size | Tracked in git | Usable? |
|---|---|---|---|
| `corpora/deepspell_data_north_america_nozip_v2.tsv` | 377 MB, 9.49 M rows | no (gitignored, on disk) | **yes — the primary training corpus survives.** 9.45 M ROAD, 44 K CITY, 128 STATE, 9 COUNTRY, with parent links and abbreviations |
| `corpora/_deepspell…v2.tokens` / `.kdtree` | 45 MB / 257 MB | no | corrector lookup space (pickled scipy cKDTree — fragile across versions) |
| `corpora/RoadFTS5_USA_unzipped.nds`, `…California…` | 1.8 GB / 777 MB | no (licensed NDS) | **yes — verified readable with stock Python `sqlite3`** (FTS5, `unicode61` tokenizer, `☱` separates alternate names). No proprietary `nds_sqlite3` needed for lookup |
| `models/*` (12 trained models) | 85 MB | yes | TF1.9 checkpoints; weights extractable, but runtime is dead |
| `corpora/grammar-*.json`, `deepspell_minimal.tsv` | tiny | yes | grammar definitions + test fixture |

What is *lost*: the original NDS "training_data" source views/EU data and whatever
produced the TSV. What is *not lost*: everything needed to retrain North America models
and to serve lookups.

TSV format (tab-separated): `class, id, name, ignored("*"), parent_class, parent_id[, abbreviation]`.

### 1.4 Why it cannot be merely "upgraded"

- **TensorFlow 1.9**: `tf.contrib`, manual `Graph`/`Session`, in-graph beam search via
  `tf.while_loop`, `tf.nn.dynamic_rnn` — all removed in TF2. There are no TF1.9 wheels
  for Python ≥3.7, none for macOS arm64, none for linux arm64. A "lib upgrade" is a
  rewrite of every model file regardless of framework choice.
- **Python 3.6 / `python:3.6-buster`**: EOL since 2021.
- **`DAWG` C-extension** (SymSpell baseline): unmaintained, fails to build on modern
  toolchains; `scipy==1.1.0` pin equally dead.
- **uWSGI**: 4 forked workers ⇒ 4 full model copies in RAM; painful to build on macOS.

### 1.5 Defects found during the review

| # | Where | Issue |
|---|---|---|
| 1 | `deepspell/ftsdb.py:46-84` + `views.py:114` | **SQL injection**: `/lookup` request args (keys *and* values) are string-formatted into the FTS5 MATCH statement. Unknown keys silently map to `criterionH` via `defaultdict`. |
| 2 | `deepspell_service/views.py:67` | Input is unconditionally `.lower()`-ed; the `lowercase` config flag is ignored, breaking full-case models. |
| 3 | `deepspell_service/views.py:91` | `completion[0][1][0]` raises `IndexError` when the top beam is empty. |
| 4 | `train-extrapolator.py:10` | Imports `deepspell_optimization.grammar` — module does not exist; script cannot run. |
| 5 | `encode.py:42` | Calls `encode_corpus(corpus, output_dir, batch_size=…)` but the signature is `encode_corpus(path, batch_size)` → `TypeError`; also nothing is ever written (writing lives in `DSTokenLookupSpace`). |
| 6 | `deepspell/baseline/fts5.py:57` | Padding count is `len(completions) - min_num_completions` (negative ⇒ no-op); operands reversed. |
| 7 | `deepspell_optimization/models/optimizer.py` | `train_test_split` is accepted and silently unused — **there is no validation split, early stopping, or test metric anywhere in training**. |
| 8 | `token_lookup_space.py:58-62` | `pickle` persistence of a 257 MB scipy object — unsafe to load, breaks across scipy versions. |
| 9 | saved model JSONs | Persisted charset lacks the BOL char `^` (falls back to index 0 = `'a'`); featureset compat is enforced only by warnings + `assert`. |
| 10 | `deepspell_optimization/models/encoder.py:54-85` | `global batch…` in nested functions creates module-level globals at call time — works by accident. |
| 11 | `ftsdb.py:82`, `views.py:119` | Debug `print()` of every SQL statement / request in the serving path. |
| 12 | repo-wide | Zero tests, no linting, no CI, no types, hardcoded training configs, `README.md` contains a "DO NOT PUSH" section with local paths. |

### 1.6 What is worth keeping (conceptually)

- The **four-stage pipeline contract** (classify → complete → correct → lookup) and the
  keystroke-level interactive UX (tab-to-complete).
- **Grammar-driven synthetic phrase sampling** from a hierarchical gazetteer — this is a
  genuinely good idea and is exactly how you train on gazetteer data without real query
  logs.
- **Class-conditioned completion** (completions never cross token-class boundaries).
- **JSON model cards** describing artifacts.
- The **FTS5 lookup contract** (class→column scoping, specificity ordering) — and the
  verified fact that stock SQLite serves it.
- `corpora/deepspell_minimal.tsv` as a perfect unit-test fixture.

---

## Part 2 — Modernized design ("DS9 v2")

### 2.1 Goals & constraints

- Same functionality and demo UX; same artifacts replaceable piece by piece.
- Runs natively on Linux x86_64/arm64, macOS (Apple Silicon + Intel), WSL2 — CPU-first,
  GPU (CUDA/MPS) optional for training.
- Training data: the surviving 9.5 M-row NA TSV **plus** reproducible open-data corpora.
- No proprietary runtime dependencies; licensed NDS databases remain usable via adapter.

### 2.2 Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language/runtime | Python ≥3.11, `pyproject.toml`, `src/` layout, `uv` for env | modern, fast installs everywhere |
| ML framework | **PyTorch 2.x** | first-class CPU/CUDA/MPS wheels on all three platforms; beam search in plain Python |
| Inference (optional) | ONNX export + `onnxruntime` | slim serving image without torch; phase-2 optimization, not a requirement |
| Service | **FastAPI + uvicorn** | async, pydantic validation, OpenAPI for free; replaces Flask+uWSGI |
| Lookup | stdlib `sqlite3` FTS5 | verified to work on the NDS files and on self-built indexes |
| ANN for corrector | `hnswlib` (file-mmap index) or brute-force numpy for ≤5 M tokens | replaces pickled cKDTree |
| Fuzzy baseline | `symspellpy` + `rapidfuzz` | replaces dead `DAWG` C-extension |
| CLI | `typer` — single `ds9` entry point | replaces 9 root-level scripts with hardcoded config |
| Config | `pydantic-settings` (JSON/env) | typed `service.json` successor, env-overridable |
| Quality | `ruff`, `pytest`, `mypy` (gradual), GitHub Actions matrix (ubuntu, macos-arm64), `docker buildx` multi-arch | the repo currently has none of these |

### 2.3 Model architecture

Recommendation: **two small PyTorch models + one optional**, replacing the three TF
graphs. Faithful to the original behavior, but with embeddings instead of 2-hot numpy
matrices and Python-side decoding instead of in-graph `tf.while_loop`.

```
                       chars (ids)            chars ⊕ class-plan (ids)
                          │                          │
                ┌─────────▼─────────┐      ┌─────────▼─────────┐
                │  Tagger           │      │  Completer        │
                │  char-emb 64      │      │  char-emb ⊕       │
                │  BiLSTM 2×256     │      │  class-emb 64     │
                │  (or 4-layer      │      │  causal LSTM      │
                │  bidir transformer│      │  2×384 (or causal │
                │  encoder)         │      │  transformer +    │
                │  → class logits   │      │  KV cache)        │
                │    per char       │      │  → next-char +    │
                └───────────────────┘      │    next-class     │
                                           │    heads          │
                                           └───────────────────┘
   Python beam search (beam k, stop on class switch / EOL,
   length-normalized log-probs, deduplication) — ~40 lines, testable.

                ┌────────────────────────────────────────────┐
                │  Corrector (two interchangeable backends)  │
                │  a) symspellpy + rapidfuzz re-rank (default│
                │     — no training, ms-fast, deterministic) │
                │  b) char encoder (mean-pooled BiLSTM or    │
                │     mini-transformer, trained contrastively│
                │     on (corrupted, clean) pairs) + hnswlib │
                └────────────────────────────────────────────┘
```

Notes:

- Parameter budget ~2-6 M per model → <25 MB fp32 each, int8-quantizable; keystroke
  latency well under 50 ms on CPU for ≤64-char prefixes.
- The tagger stays bidirectional (the original's bw-pass is why it beats a causal
  tagger); the completer stays causal and class-conditioned, preserving the
  "complete only the current token class" semantics.
- Backend (b) of the corrector preserves the original VAE idea (errors beyond edit
  distance 2, phonetic-ish neighborhoods) with a simpler, stronger training objective
  (contrastive/triplet on corruption pairs instead of a VAE), and replaces
  pickle+cKDTree with an mmap-able `hnswlib` index + `.npz` vectors + a plain-text
  token list.
- A single shared-backbone multi-task model is possible later (tagging head +
  LM head on one encoder), but separate small models are easier to train, evaluate and
  swap — and mirror the existing artifact layout.
- Old TF checkpoints: portable in principle (`BasicLSTMCell` kernels → torch LSTM with
  i,c,f,o → i,f,g,o gate reorder; `tf-cat-model.py` already dumps tensors), but since
  the full training TSV survives, **retraining is the cleaner path**; weight porting is
  a fallback only if training compute is unavailable.

### 2.4 Data layer — one gazetteer artifact instead of four formats

Replace {TSV, in-RAM token graph, `.tokens`/`.kdtree`, NDS-or-nothing lookup} with a
single canonical **SQLite gazetteer** that powers both training and serving:

```sql
-- gazetteer.sqlite
CREATE TABLE token (
  id INTEGER PRIMARY KEY,
  class TEXT NOT NULL,            -- COUNTRY | STATE | CITY | ROAD (extensible: ZIP, POI)
  name TEXT NOT NULL,             -- display form
  name_norm TEXT NOT NULL,        -- folded/normalized form used by models
  abbrev TEXT,                    -- "CA", "USA", …
  parent_id INTEGER REFERENCES token(id),
  source TEXT,                    -- geonames | osm | legacy_tsv | nds
  freq REAL DEFAULT 1.0
);
CREATE VIRTUAL TABLE lookup USING fts5(      -- serving index, same contract as NDS
  doc_id UNINDEXED, morton UNINDEXED,
  road, city, state, country,
  tokenize = "unicode61 remove_diacritics 2"
);
```

- **Training**: a `torch IterableDataset` streams `token` rows, walks parent/child links
  in SQL, applies the (ported) grammar sampler + corruption on the fly — no 9.5 M
  Python objects in RAM, seeds everywhere for reproducibility.
- **Serving**: the `lookup` FTS5 table answers `/lookup` with **parameterized,
  whitelist-validated** queries (fixes the injection). An `NdsLookupAdapter` keeps the
  existing licensed `.nds` files working unchanged (column mapping config exactly as
  today's `fts_db` block).
- **Normalization**: Unicode NFKD + casefold with an explicit, versioned charset in the
  model card (keep `unidecode` behavior as a compat option so the legacy TSV reproduces
  byte-identical inputs).

Importers (each a `ds9 data import-…` subcommand):

1. **`import-legacy-tsv`** — the surviving 377 MB NA corpus. Day-one parity.
2. **`import-geonames`** — countries (`countryInfo.txt`), states/provinces
   (`admin1CodesASCII.txt`), cities (`cities500.zip`, ~200 K places; `allCountries` for
   full depth), plus `alternateNames` for abbreviations/multilingual forms.
   License CC-BY 4.0 — unproblematic. Covers COUNTRY/STATE/CITY worldwide, **no roads**.
3. **`import-osm`** — road names from OpenStreetMap extracts (Geofabrik per-region
   `.osm.pbf`): ways with `highway=*` and `name=*`, de-duplicated per city; city/state
   assignment via `boundary=administrative` polygons (admin_level 8/4) using a two-pass
   `pyosmium` + STRtree spatial join (or DuckDB-spatial). License **ODbL 1.0** —
   share-alike applies to the derived *gazetteer database*; attribution required.
   This replaces the road coverage of the lost NDS training views.
4. **`import-openaddresses`** *(optional)* — street/city/region triples from
   OpenAddresses CSVs as a second road source (mostly permissive/CC-BY per source).
5. **`import-nds`** *(optional)* — ingest licensed NDS FTS tables where available
   (works with stock sqlite3, as verified).

With (2)+(3) the project becomes fully reproducible from open data for any region —
including the EU coverage the original team lost.

### 2.5 Service & UI

- **FastAPI** app, models loaded once in lifespan; endpoints:
  - `GET /api/complete?q=…` → `{classes, completions, corrections, timings}` (pydantic
    schema; legacy `/extrapolate` alias kept for the old UI contract),
  - `GET /api/lookup?road=…&city=…&n=10` → parameterized FTS5,
  - `GET /` → static single-file UI (vanilla JS, no jQuery/CDN deps; keep the
    category-heatmap, completion list, correction list, lookup table and tab-to-complete
    exactly as today).
- One uvicorn worker by default (models are CPU-bound; `torch.inference_mode()` +
  `torch.set_num_threads`), `--workers N` opt-in.
- Structured logging (`logging`/`structlog`), `/healthz`, request timing in the payload
  as today.

### 2.6 Training & evaluation harness

- `ds9 train tagger|completer|encoder --config configs/….yaml` — declarative configs
  replace the hardcoded `train-*.py`; checkpoints + model cards under `models/`,
  TensorBoard or W&B optional.
- Real validation: held-out token split (the thing `train_test_split` never did),
  early stopping, seeded sampling.
- `ds9 eval` reimplements the two eval scripts' metrics as a regression suite:
  - tagger: per-char and per-token class accuracy vs. truncation length,
  - completer: completion accuracy@k / saved-keystrokes vs. the trie baseline
    (port of `baseline/fts5.py`, bug fixed),
  - corrector: recall@k vs. corruption level, vs. symspell baseline.
- `ds9 demo` replaces `predict.py`/`match.py` REPLs; `ds9 encode` replaces `encode.py`.

### 2.7 Proposed repository layout

```
deep-spell-9/
├── pyproject.toml              # one package: deepspell (lib) + ds9 (CLI)
├── src/deepspell/
│   ├── charset.py              # normalization + vocab (versioned)
│   ├── gazetteer/              # sqlite schema, importers (geonames, osm, legacy, nds)
│   ├── sampling/               # grammar rules, phrase sampler, corruption (seeded)
│   ├── models/                 # torch: tagger.py, completer.py, encoder.py, cards.py
│   ├── decode/                 # beam search, class-sequence extraction, tokenize
│   ├── correct/                # symspell backend, embedding backend (hnswlib)
│   ├── lookup/                 # fts5 store + query builder, nds adapter
│   ├── train/                  # loops, configs, eval metrics
│   ├── service/                # fastapi app, schemas, static UI
│   └── cli.py                  # typer: data/train/eval/serve/demo/encode/export
├── tests/                      # fixtures incl. deepspell_minimal.tsv
├── configs/                    # training + service configs
├── docker/Dockerfile           # multi-arch (linux/amd64, linux/arm64), python:3.12-slim
└── docs/
```

### 2.8 Migration roadmap

| Phase | Deliverable | Notes |
|---|---|---|
| **0. Hygiene** | pyproject + ruff + pytest + CI; port pure-Python parts (grammar sampler, tokenization, charset) with golden tests against the legacy implementation; fix the FTS5 injection in a minimal patch if the old service must keep running | small |
| **1. Data** | gazetteer schema + `import-legacy-tsv` + FTS5 lookup store + NDS adapter; `/lookup` served from the new layer | unblocks everything else; verified feasible |
| **2. Models** | tagger + completer in PyTorch trained on the legacy TSV; eval harness shows parity with the shipped TF models (use the old service as reference oracle while it still runs in Docker) | the core effort |
| **3. Corrector** | symspellpy backend (default) + contrastive encoder backend + hnswlib space; `ds9 encode` | independent of phase 2 |
| **4. Service/UI** | FastAPI + static UI + docker buildx multi-arch images; retire Flask/uWSGI/TF image | demo parity on macOS/arm64 |
| **5. Open data** | `import-geonames` + `import-osm`; retrain NA from open data and compare against legacy-TSV models; document ODbL obligations | makes the project self-sufficient |

Each phase leaves the system runnable; the legacy Docker image remains the reference
implementation until phase 4 completes.

### 2.9 Risks & open questions

- **OSM road→city assignment** is the only technically fiddly importer (boundary
  polygons, enclaves, unnamed ways). Mitigation: start with OpenAddresses or
  per-region Geofabrik extracts; accuracy needs only to match the training-noise level.
- **ODbL share-alike** applies to the derived gazetteer DB if distributed; model weights
  are a legal gray zone (community consensus: produced works). Document attribution
  either way. GeoNames (CC-BY) is the conservative core; OSM adds roads.
- **Quality parity**: synthetic-grammar training reproduces the old behavior, but exact
  metric parity needs the eval harness early (phase 2 gate).
- **Latency on low-end CPUs**: if LSTM beam search is too slow per keystroke, fall back
  to int8 ONNX or shrink the completer; the trie baseline is a functional floor.
- Keep or drop the **VAE-style neural corrector**? Default symspell covers the demo;
  the neural backend is where research value lies. Proposal keeps both behind one
  interface.
