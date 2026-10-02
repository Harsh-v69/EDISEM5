# Project Progress (read this first)

_Last updated: 2026-10-02 (Phases 0-3 complete). For a full project briefing (slides/reports) see `context.md`._ This file is written so someone with zero background can understand the project and where it stands._

## 1. What is this project?

In India, many people take **Ayurvedic herbal medicines together with modern prescription drugs**. Herbs contain active chemicals that can change how a drug is absorbed, broken down or cleared, sometimes making the drug useless or dangerous. Nobody checks this the way pharmacists check drug-drug clashes. This project builds a research system to start filling that gap.

The system is built on **one shared Ayurvedic Knowledge Graph (KG)**: a network linking herbs → the chemicals inside them → the proteins those chemicals act on → the drugs that act on the same proteins, plus diseases, doshas and classical formulations. On top of it we build:

1. **Herb-drug interaction (HDI) predictor**: a graph neural network (GNN) that scores how risky a herb + drug pair is.
2. **Safe-composition suggester**: suggests which ingredients and proportions in a formulation would avoid predicted interactions with a patient's drugs.
3. **Ayurveda GraphRAG**: a question-answering assistant over classical texts plus the KG that always shows citations (so it cannot just make things up).
4. **Jivha (tongue/nail) and Nadi (pulse) input** _(gated)_: photos/pulse signals are turned into a dosha (body-constitution) estimate that personalises the answers. This needs real clinical data and ethics approval, so it only goes ahead if a clinical partner is secured.

**Important:** everything produced is a **research risk score / decision support. It is not medical advice or diagnosis.**

Goal: publish research papers (venue undecided; built to a bioinformatics-journal standard). Planned split: Paper 1 = KG + HDI-GNN + safe-composition; Paper 2 = GraphRAG; Paper 3 (optional) = Jivha/Nadi.

## 2. Glossary

| Term | Meaning |
|---|---|
| Herb / Compound / Target / Drug | A plant / a chemical inside it / a protein it acts on / a prescription medicine |
| CYP | Cytochrome P450 liver enzymes that break down most drugs. Herbs that affect them cause many interactions |
| KG | Knowledge graph: entities (nodes) and relationships (edges) |
| GNN | Graph neural network: a model that learns from graph structure |
| Link prediction | Predicting whether an edge (here: herb-drug interaction) exists |
| RAG / GraphRAG | Retrieval-augmented generation: an LLM answers using retrieved passages (and, for GraphRAG, graph facts) |
| Silver / gold labels | Silver = interaction labels derived mechanistically (used for training). Gold = hand-curated from published case reports (used only to evaluate) |
| Jivha / Nadi | Tongue (and nail) examination / pulse examination in Ayurveda |
| Dosha / prakriti | The three body-energy types (Vata, Pitta, Kapha) / a person's constitution |
| IMPPAT | Public database of Indian medicinal plants and their chemicals (IMSc Chennai) |

## 3. Where things are

| Path | What |
|---|---|
| `docs/superpowers/specs/2026-10-01-ayurveda-kg-hdi-design.md` | The approved design (architecture, data, validation, risks) |
| `docs/superpowers/plans/` | Step-by-step implementation plans |
| `docs/data_access_audit.md` | Which data sources are reachable (probed live) |
| `ayurveda_kg/` | The Python package |
| `tests/` | Automated checks (run after every step) |
| `data/raw/`, `data/manifest.json` | Downloaded raw data + checksummed record of each file's source and licence |

How to run the checks: `.\.venv\Scripts\python -m pytest -q`

## 4. Status by phase

| Phase | What | Status |
|---|---|---|
| 0 | Scaffold, dataset manifest, polite downloader, data-access audit | **Done** |
| 1 | Download raw data (IMPPAT, ChEMBL, DGIdb, DDInter, TDC), fix herb/drug scope | **Done**: 1,696 compounds crawled (3,392 pages, 0 missing), all drug-side data in `data/manifest.json` |
| 2 | Resolve entities, build the unified KG | **Done**: 0 validation problems, 5 plausibility checks pass (see `docs/kg_report.md`) |
| 3 | HDI labels (silver/gold), leak-safe splits | **Done**: 59,360 labelled pairs, masking proven on real data, cold-herb folds leak-free, 12-row cited gold set (see `docs/labels_report.md`) |
| 4 | Baselines, GNN, evaluation; **Jivha/Nadi go/no-go** | Not started |
| 5 | Safe-composition optimiser | Not started |
| 6 | GraphRAG + evaluation | Not started |
| 7 | Jivha/Nadi pipeline (only if go) | Not started |
| 8 | Demo + paper drafts | Not started |

Test suite: 90 automated tests, all passing (`.\.venv\Scripts\python -m pytest -q`).

## 5. Log (newest first)

- **2026-10-02, Phase 3 complete (gate passed).** Real-data gate: masking the label-source edges leaves 0 of 10,235 positives re-derivable; all 5 cold-herb folds are leak-free with every herb held out once (108-303 shared compounds dropped per fold, reported in `docs/labels_report.md`). Final silver labels: 10,235 positive / 33,071 negative / 16,054 unlabelled. Reproduce: `python -m ayurveda_kg.phase3`.
  - **Two things Phase 4 must respect:** (1) the silver labels give false alarms against clinical gold (ginger-warfarin at the 97th percentile although a human trial found no interaction), so results must be reported against the gold set and the limitation stated; (2) the folds are very uneven (test positives range from 412 to 3,373 per fold because the ginger and licorice folds differ a lot), so report per-fold numbers, not just a mean.
- **2026-10-02, Phase 3 (in progress).** Built and tested: silver labels at compound-drug level (rule in `docs/superpowers/plans/2026-10-02-phase3-labels-splits.md`), leakage masking (a test proves labels cannot be re-derived from the masked graph), cold-compound / cold-herb / cold-drug splits, herb-level aggregation, a 12-row **literature gold set** (each row read from the primary PubMed record or full-text passage via Europe PMC, marked pending expert review) and a cited substrate supplement for rivaroxaban, apixaban (FDA labels) and theophylline (PMID 7619675). Current silver labels: 59,360 pairs = 10,235 positive / 33,071 negative / 16,054 unlabelled.
  - **Honest finding:** silver labels agree with the literature for piperine (phenytoin, carbamazepine, propranolol, theophylline) and guggul (diltiazem, propranolol), and correctly miss the pharmacodynamic licorice-diuretic pairs, but give a **false alarm for ginger-warfarin** (human trial found no interaction; silver scores it at the 97th percentile) and a milder one for garlic-warfarin. Predicted CYP inhibition is a mechanistic hypothesis, not a clinical interaction. Details in `context.md` section 7.3.
  - Repo published to GitHub. IMPPAT-derived data and page excerpts are deliberately **not** committed (CC BY-NC-ND); fixture-based tests skip when those local files are absent.
- **2026-10-01, Phases 1-2 complete (gate passed).** IMPPAT crawl finished (one page hit a transient connection drop and was re-fetched; this exposed that an empty cached file would never be re-downloaded, which is now fixed and tested). Final KG: 20 herbs, 1,696 compounds, 35 drugs, 1,667 target genes; edges: 2,721 herb-compound, 11,947 compound-target, 1,436 predicted CYP inhibitions, 482 predicted P-gp substrates, 1,209 drug-target, 50 drug-enzyme substrate + 31 verified non-substrate, 377 drug-drug. 0 validation problems; 52 tests pass. Evidence checks: per-herb compound counts equal IMPPAT's own page counts; piperine (black pepper) links to ABCB1/P-gp and CYP1A2/2C9/2C19; licorice has the most predicted CYP3A4 inhibitors, consistent with its known reputation.
  - **Findings that change Phase 3 (important):** (a) 1,120 of 1,696 compounds (66%) have no compound-target edge, so mechanism-based scoring is sparse; consider adding ChEMBL/PubChem bioactivity for compounds that have IDs. (b) Only 20 herbs x 35 drugs = 700 pairs, too few to split by herb with statistical power; plan to predict at **compound-drug level** (~59,000 pairs) and aggregate to herb, keeping herb-wise splitting as the honest test. (c) Substrate gaps remain for rivaroxaban, apixaban, theophylline (fill from cited literature).
- **2026-10-01, Phase 1-2 build (partial data).**
  - *Data found:* IMPPAT's advertised bulk files are dead links (HTTP 404), so we crawl its per-page data instead, only for our 20 scoped herbs (1,696 unique compounds; 2 pages each; 1 request/second with caching). Per compound this gives identifiers (InChIKey, ChEMBL, PubChem), target genes, and SwissADME CYP-inhibition predictions. Drug side: DGIdb (drug-gene), DDInter (drug-drug), ChEMBL metabolism (drug-enzyme), TDC Carbon-Mangels (CYP2C9/2D6/3A4 substrate and verified non-substrate labels).
  - *Code:* page parsers, crawler, ChEMBL resolver, entity resolution, KG builder, validator, report generator. Every piece was written test-first.
  - *Bugs found and fixed by checking real data, not just tests:* (1) drug matching returned a duplicate ChEMBL entry instead of the parent molecule (theophylline); (2) enzyme families like `ESTERASES` became fake gene nodes; (3) 11 drugs had zero DGIdb targets because of name variants (glyburide vs glibenclamide, salts), fixed with a per-drug alias table that refuses ambiguous aliases; (4) ChEMBL alone gave substrate info for only ~14 of 35 drugs, so TDC was added.
  - *Partial-data KG (191 of 1,696 compounds crawled):* 20 herbs, 35 drugs, ~1,300 target genes; 0 validation problems.
  - *Still open:* crawl completion; final KG build; plausibility spot-checks (e.g. piperine should link to CYP3A4/P-gp).
- **2026-10-01, Phase 0 complete.** Created package, venv, `manifest.py` (records source/licence/checksum of each raw file), `fetch.py` (rate-limited cached downloader), `audit.py` (probes sources). 9 tests pass. Live audit: IMPPAT, PubChem, ChEMBL, DGIdb, DDInter, HuggingFace reachable; DrugBank returned 403 (needs an academic licence application, stays optional). Smoke test: fetched a real PubChem file, manifest verified clean.

## 6. Key decisions so far

- One system, several papers (see §1). Open-data-first: DrugBank is an optional plug-in.
- Labels: silver (training) and gold (evaluation only); splits by herb; masked-edge leakage test. This avoids the model "cheating" by rediscovering the rule used to make the labels.
- Local open LLMs only; no API keys. Hardware: RTX 4060 (8 GB), ~15 GB RAM.
- NetworkX + parquet instead of Neo4j.

## 7. Known risks / open items

- IMPPAT's bulk-download links are broken (404). We crawl per-page, scoped to our 20 herbs. Its licence is CC BY-NC-ND 4.0 (no derivatives): we must not redistribute IMPPAT-derived tables. Plan: release code + identifiers only, and ask IMSc (contact on their site) for permission/bulk files before publication. _(Sending that email is for the project owner.)_
- Substrate data gaps: rivaroxaban, apixaban, theophylline have no CYP substrate entry in our open sources; fill from cited literature in Phase 3.
- Many compound-target and CYP-inhibition edges are in-silico predictions (SwissADME, STITCH-style); the graph keeps the source on every edge so the paper can separate predicted from experimental evidence.
- Need a **domain advisor** (Ayurveda expert) for verifying the GraphRAG question set and KG spot-checks.
- Need a **clinical partner + ethics approval** for Jivha/Nadi (go/no-go at end of Phase 4).
- Project folder is not a git repo yet (versioning is recommended).

## 8. Next

1. Phase 4: baselines (Random Forest, matrix factorisation) then GNN link predictor on the **masked** graph, evaluated on cold-herb / cold-compound / cold-drug splits with per-fold results, plus gold-set recall for the pharmacokinetic subset.
2. Decide the **Jivha/Nadi go/no-go** at the end of Phase 4 (needs a clinical partner + ethics approval; otherwise future work).
3. Human review still needed: the 12 gold pairs should be checked by a pharmacist or Ayurveda/pharmacology expert before any paper claims.
4. To rebuild everything from cached raw files (no network): `python -m ayurveda_kg.build` then `python -m ayurveda_kg.phase3`.
