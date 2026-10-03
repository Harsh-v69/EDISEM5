# Ayurvedic knowledge graph for herb–drug interaction research

A research project built around **one shared knowledge graph** of Ayurvedic herbs → their compounds → protein targets/enzymes → commonly co-prescribed drugs. On top of it: a herb–drug interaction (HDI) benchmark, a formulation re-weighting study, and a cited question-answering system over two classical texts.

> **Everything here is research software. Outputs are research risk scores and hypotheses, never medical advice or a diagnosis.** The scores come from predicted enzyme inhibition and have known false alarms (for example ginger with warfarin).

## Run it (short version)

**On the computer where the data is already built** (open a terminal in the project folder):

```bash
.venv\Scripts\python -m streamlit run ayurveda_kg/demo/app.py
```

Your browser opens the demo at http://localhost:8501. It is set to answer on this computer only (see `.streamlit/config.toml`), because it shows IMPPAT-derived data that must not be shared. Pick a herb and a drug on the first tab; the other tabs re-weight a formulation and search the classical texts. (For written answers on the third tab, start Ollama with the `qwen3:8b` model and tick the checkbox; without it you still get the cited passages.)

Check that everything works:

```bash
.venv\Scripts\python -m pytest -q
```

**On a new computer:** install Python 3.12, then `python -m venv .venv` and `.venv\Scripts\python -m pip install -r requirements.txt`. The tests run right away (the ones that need data skip themselves). **The demo needs the data, which is not in this repository** (IMPPAT's licence forbids publishing it), so rebuild it first with the steps under "Rebuild the results" below; the slow part is two polite web crawls of about an hour each.

## Details

New to the project? Read [`progress.md`](progress.md) (plain-language status and log) or [`context.md`](context.md) (full briefing for reports and slides).

## What exists

| Phase | What | Where |
|---|---|---|
| 0–2 | Reproducible data ingestion and the validated knowledge graph (20 herbs, 35 drugs) | [`docs/schema.md`](docs/schema.md), [`docs/kg_report.md`](docs/kg_report.md) |
| 3 | Mechanistic silver labels, leakage control, leak-free splits, a 12-pair literature gold set | [`docs/labels_report.md`](docs/labels_report.md) |
| 4 | Baselines and a graph neural network, with honest results | [`docs/phase4_results.md`](docs/phase4_results.md) |
| 5 | Re-weighting real formulations to lower predicted risk | [`docs/phase5_results.md`](docs/phase5_results.md) |
| 6 | GraphRAG over the Charaka and Sushruta translations | [`docs/phase6_results.md`](docs/phase6_results.md) |
| 7 | Jivha/Nadi: **not built** (needs a clinical partner and ethics approval); protocol and tested agreement code only | [`docs/jivha_nadi_protocol.md`](docs/jivha_nadi_protocol.md) |
| 8 | Local demo app and two paper drafts | [`ayurveda_kg/demo`](ayurveda_kg/demo), [`docs/paper`](docs/paper) |

## Headline findings (details and caveats in the reports)

- **Label leakage is a large effect.** Removing the label-defining enzyme edges from the training graph changes cold-split AUROC by up to about 0.11.
- **Graph structure added little** over compound chemistry and target annotations: a Random Forest beats the GNN on every cold split, and the GNN matches its no-message-passing ablation.
- **Drug-side generalisation is weak.** A pooled cold-drug AUROC near 0.98 is almost entirely compound-side skill; ranking unseen drugs for a compound is near 0.7 (Random Forest) and near chance (GNN).
- **Formulation re-weighting has modest effects** (median ~4% predicted-risk reduction) and inherits the labels' false alarms.
- **GraphRAG:** alias expansion plus graph facts raise recall on English-name questions about 10× over plain retrieval, but absolute recall is still low; a small local model refuses, over-reads or blends passages in ways automatic metrics miss. **No expert-verified question set exists yet.**

## Quickstart

Python 3.12. For GPU training install PyTorch from the CUDA wheel index first (see the header of [`requirements.txt`](requirements.txt)).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest -q                            # data-dependent tests skip cleanly on a fresh clone
```

### Rebuild the results from the raw sources

The raw data and every IMPPAT-derived table are **not in this repository** (licences, below), so a fresh clone must regenerate them. Steps run in order and are resumable; the crawls are deliberately slow (1 request/second).

```bash
python -m ayurveda_kg.ingest.fetch_all             # DGIdb, DDInter, ChEMBL, TDC labels and the two public-domain texts (manifest-tracked, idempotent)
python -m ayurveda_kg.ingest.crawl_imppat          # ~1,700 compound pages, about an hour
python -m ayurveda_kg.ingest.crawl_formulations    # ~1,600 formulation pages, about an hour
python -m ayurveda_kg.build                         # entity resolution, the knowledge graph, its validator and report
python -m ayurveda_kg.phase3 && python -m ayurveda_kg.phase4 && python -m ayurveda_kg.phase4_report
python -m ayurveda_kg.phase5                        # composition study (~15 min)
python -m ayurveda_kg.rag.build_index               # needs the public-domain texts and the embedding model
python -m ayurveda_kg.rag.run_extract               # needs a local model server (Ollama, qwen3:8b)
python -m ayurveda_kg.phase6
python -m ayurveda_kg.paper                         # renders docs/paper/*.md from the stored results
```

Every data file that is downloaded is recorded in `data/manifest.json` (source URL, licence, checksum) and logged in section 9 of `progress.md`.

### Run the demo (local only)

```bash
.venv/Scripts/python -m streamlit run ayurveda_kg/demo/app.py
```

Three tabs: risk lookup with a graph-path explanation and the published study where one exists; re-weight a real formulation for a patient's drugs; ask the classical texts (retrieval only, or with the local model).

## Data sources and licences

| Source | Used for | Licence note |
|---|---|---|
| IMPPAT 2.0 (IMSc Chennai) | herbs, compounds, targets, predicted CYP inhibition, formulations | **CC BY-NC-ND 4.0**: derived tables must not be redistributed; the demo must not be hosted publicly. Permission should be requested from the authors before any release of derived data |
| ChEMBL, DGIdb, DDInter 2.0 | drug identity, enzymes, targets, drug–drug interactions | open data; cite, and verify the exact terms before any release |
| TDC / Carbon-Mangels and Hutter | CYP2C9/2D6/3A4 substrate and non-substrate labels | open; cite both |
| Kaviratna *Charaka-Samhita*, Bhishagratna *Sushruta Samhita* | GraphRAG corpus | public domain (US); archive.org scans |
| DrugBank | not used | needs an academic licence |

`.gitignore` keeps `data/raw/`, `data/processed/` and the IMPPAT test fixtures out of version control; the tests that need them skip when they are absent.

## Repository map

`ayurveda_kg/` code (`ingest/`, `rag/`, `demo/`, `vision_nadi/`, plus the phase modules) · `config/` the fixed scope and the entity lexicon · `data/gold`, `data/curated` hand-curated cited data · `data/manifest.json` · `docs/` design spec, plans, schema and generated reports, `docs/paper` the draft templates · `tests/` automated checks (every module is test-first).

## Known limits

Only 20 herbs, 35 drugs and five CYP enzymes; labels are partly in-silico predictions; the gold set has 12 pairs and awaits expert review; GraphRAG awaits an expert-verified question set; the AyurParam comparison and Jivha/Nadi are not done. See the backlog in `progress.md` section 10.
