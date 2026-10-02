# Phase 1-2 Implementation Plan: Raw data + unified Knowledge Graph

> **For agentic workers:** Use superpowers:executing-plans (inline). Steps use checkbox syntax. Each task ends with passing tests + a `progress.md` entry.

**Goal:** Download and validate raw herb/compound/target/drug data for a fixed scope, resolve entities, and build a validated heterogeneous KG on disk.

**Architecture:** Parsers are pure functions over HTML/TSV text (tested on small fixtures cut from real pages). Crawlers only download and cache raw files (polite, 1 req/s, index of sha256). Resolution and KG building read raw data and write parquet under `data/processed/`.

**Tech Stack:** Python 3.12, requests, beautifulsoup4 + lxml, pandas, pyarrow, networkx, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-ayurveda-kg-hdi-design.md`

## Global Constraints

Same as the Phase 0 plan (research risk score wording, local only, manifest for every raw file, no accounts/credentials, validate then update `progress.md`). Plus:
- IMPPAT is **CC BY-NC-ND 4.0**: raw/derived IMPPAT tables are never committed or redistributed; the paper releases code + IDs only, and IMSc permission is requested for any derived release.
- Crawl rate: at most 1 request/second, descriptive User-Agent, cached (never refetch).
- Scope is fixed here and not expanded later: `config/scope.yaml`.

## Findings that shaped this plan (2026-10-01 live investigation)

- IMPPAT's advertised batch-download files return HTTP 404 (retry before finalising). Per-page access works: plant page -> compounds; `phytochemical-detailedpage/<ID>` -> SMILES, InChIKey, CID, ChEMBL ID, synonyms; `humantargets/<ID>` -> target genes (Ensembl, Entrez, symbol, source) + SwissADME predictions (CYP1A2/2C19/2C9/2D6/3A4 inhibitor, P-gp substrate).
- IMPPAT names *Tinospora cordifolia* as *Tinospora sinensis*; herb synonym resolution is a real requirement, not an edge case.
- DrugBank is HTTP 403 (licence); use open drug sources.

## Tasks

### Task 1.1: Scope config
Files: `config/scope.yaml`, `ayurveda_kg/scope.py`, `tests/test_scope.py`.
Interface: `load_scope(path="config/scope.yaml") -> dict` with `herbs: list[{imppat_name, aliases, common}]`, `drugs: list[{name, cls}]`. Tests: loads; no duplicate herb or drug names (case-insensitive); every alias unique across herbs; every herb has non-empty `imppat_name`.

### Task 1.2: IMPPAT parsers
Files: `ayurveda_kg/ingest/imppat.py`, `tests/fixtures/imppat_*.html`, `tests/test_imppat.py`.
Interfaces (pure functions on HTML text):
- `parse_plant_page(html) -> list[dict]`: keys `plant, part, phy_id, phy_name, reference`.
- `parse_detail_page(html) -> dict`: keys `phy_id, name, synonyms: list[str], cid, chembl_id, smiles, inchi, inchikey`.
- `parse_targets_page(html) -> dict`: keys `targets: list[{gene, ensembl, entrez, source}]`, `adme: {"CYP3A4 inhibitor": "Yes"/"No", ..., "P-glycoprotein substrate": ...}`.
Tests use fixtures cut from real pages (Vitamin E `IMPPAT3_PHYID000017`; Curcuma longa rows). Assert exact known values: InChIKey `GVJHHUAWPYXKBD-IEOSBIPESA-N`, ChEMBL `CHEMBL47`, CID `14985`, CYP3A4 target present with entrez `1576`, `P-glycoprotein substrate == "Yes"`.

### Task 1.3: IMPPAT crawler + run
Files: `ayurveda_kg/ingest/imppat.py` (add `crawl_plants`, `crawl_compounds`).
Interfaces:
- `crawl_plants(herb_names, raw_dir, delay=1.0) -> dict[str, Path]` (plant pages).
- `crawl_compounds(phy_ids, raw_dir, delay=1.0, log_every=100)` downloads detail + humantargets page per ID into `raw_dir/compounds/<id>.detail.html` / `.targets.html`; writes `raw_dir/crawl_index.tsv` (url, path, sha256, fetched_at); registers the index in `data/manifest.json`. Resumable (cache).
Validation: index row count == 2 x unique IDs; no zero-byte files; 3 random pages re-parse with no exceptions. Run in the background (~1 h).

### Task 1.4: Drug-side raw data
Investigate then fetch (formats unknown until checked): DDInter 2.0 downloads, DGIdb TSV, ChEMBL (mechanisms + activities against CYP/transporter targets), a public CYP substrate list. Each fetched through `fetch()` into the manifest. Validation: file non-empty, expected columns present, scoped drugs found (report coverage %).

### Task 1.5: Phase 1 gate
Report `docs/phase1_data_report.md`: counts (herbs, compounds, targets, drugs), coverage of scoped drugs, known gaps. `progress.md` updated.

### Task 2.1: Entity resolution
Files: `ayurveda_kg/resolve.py`, `tests/test_resolve.py`.
Interfaces:
- `norm_name(s) -> str` (lowercase, strip, collapse whitespace/punctuation).
- `resolve_herbs(scope, plant_rows) -> DataFrame[herb_id, name, aliases]`: alias/synonym merged (Tinospora cordifolia -> sinensis).
- `resolve_compounds(details) -> DataFrame[compound_id(InChIKey), phy_ids, name, smiles, cid, chembl_id]`: merge by InChIKey; report merge count.
- `resolve_drugs(scope, sources) -> DataFrame[drug_id, name, cls, chembl_id]`.
Tests: synonym merge, InChIKey merge, unmatched scoped drug reported not dropped silently.

### Task 2.2: KG builder
Files: `ayurveda_kg/kg.py`, `tests/test_kg.py`.
Interfaces: `build_kg(nodes: dict[str, DataFrame], edges: dict[str, DataFrame]) -> networkx.MultiDiGraph`; `save_kg(g, dir)` parquet nodes/edges; `load_kg(dir)`.
Node types: Herb, Compound, Target, Drug, Disease (later), Formulation (later). Edge types: `contains` (Herb->Compound), `modulates` (Compound->Target), `predicted_cyp_inhibitor` (Compound->Target), `targets` (Drug->Target), `substrate_of` (Drug->Target).

### Task 2.3: KG validation + schema doc
Files: `ayurveda_kg/validate.py`, `tests/test_validate.py`, `docs/schema.md`.
Checks (each a function returning a list of problems): no dangling edges; every edge's endpoint types match the schema; no duplicate node ids; every scoped herb present; herbs with zero compounds flagged; drug coverage report. Gate: problems list empty on the real KG (or each exception documented).

### Task 2.4: Phase 2 gate
Real KG built from real data, validation passes, `docs/schema.md` written, `progress.md` updated.
