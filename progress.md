# Project Progress (read this first)

_Last updated: 2026-10-02 (Phases 0-4 complete; Phase 5 code done, final run waiting on a data crawl; Jivha/Nadi go/no-go pending your decision). For a full project briefing (slides/reports) see `context.md`. This file is written so someone with zero background can understand the project and where it stands._

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
| 4 | Baselines, GNN, evaluation; **Jivha/Nadi go/no-go** | **Done** (`docs/phase4_results.md`): RF beats GNN on every cold split; graph structure adds nothing over node features; drug-side generalisation weak. **Go/no-go decision still needs you** |
| 5 | Safe-composition optimiser | **Code and tests done** (LP optimiser, out-of-fold risk, scenario/validation analysis, 142 tests pass); **final run pending** the crawl of IMPPAT's ~1,576 formulation pages (about an hour at the polite rate). A smoke run on the first 109 formulations behaved correctly (coverage floor respected, no herb cut below half its baseline) but its numbers are not final |
| 6 | GraphRAG + evaluation | **Code and tests done** (corpus, index, lexicon, grounded extraction, hybrid retrieval, generation checks, evaluation harness; 195 tests pass). **Real evaluation run pending** the LLM triple extraction (476 passages, running) and then the answer evaluation. Plan: `docs/superpowers/plans/2026-10-02-phase6-graphrag.md` |
| 7 | Jivha/Nadi pipeline (only if go) | Not started |
| 8 | Demo + paper drafts | Not started |

Test suite: 195 automated tests, all passing (`.\.venv\Scripts\python -m pytest -q`).

## 5. Log (newest first)

- **2026-10-02, Phase 6 code built (real evaluation run pending).** Downloaded the approved texts (Charaka, Kaviratna; Sushruta vols 1-3, Bhishagratna; ~8 MB) and the MiniLM embedding model; 6,493 passages indexed in 57 s. New modules under `ayurveda_kg/rag/`: `corpus` (OCR cleaning + chunking), `index` (brute-force cosine; FAISS not needed at this size), `lexicon` (alias-aware entity linker; `config/lexicon.yaml` is a hand-written v1 that needs an advisor), `extract` (grounded LLM triple extraction + co-occurrence graph), `textgraph`, `kgfacts`, `retrieve` (dense + alias expansion + text graph + KG facts, fused by reciprocal rank), `generate` (citation-constrained prompt, disclaimer appended by code, citation-validity and lexical-support checks), `evaluate` (synthetic retrieval questions, KG-grounded multi-hop questions with programmatic gold, expert template).
  - **What real data showed:** (1) a *vocabulary gap*: a question about "turmeric" misses passages that say "Haridra", which alias expansion is designed to close; (2) the linker finds 16 of the 20 scoped herbs in the texts (fenugreek, guggul, kalmegh, sarpagandha not found under our spellings) and 476 passages mention a herb together with a condition.
  - **Extraction bug found by looking at results, not just tests:** the first LLM extraction run accepted **0 of 100** triples. The grounding filter was right to reject them, but the setup was wrong: the model extracted triples about herbs outside our 20-herb scope, and used verbs like "allays" and "destroys" that our allow-list rejected. Fix: a prompt focused on the scoped herbs actually present in each passage, and a transparent verb map (the raw verb is kept on every triple). A 10-passage probe then accepted 17 of 35 triples. One over-link remains (an object phrase containing the word "wind" links to the vata dosha); it is listed as a limitation for expert review.
  - **Crawler bug:** the formulation crawl crashed when a failure message contained a Sanskrit name that the Windows console cannot print; fixed with a tested safe-logging helper and resumed.
- **2026-10-02, Phase 5 code built (final run pending).** New: `compose.py` (linear-programming optimiser: re-weights a formulation's in-scope herbs to minimise predicted interaction risk for a patient's drug, with per-herb share bounds and a coverage floor per therapeutic use; a randomised property test of 400 problems checks every constraint holds and the result is never worse than baseline), `risk.py` (out-of-fold compound/herb-drug risk, so no score comes from a model that saw that compound), `phase5.py` (formulation parsing, scenarios, constraint sweeps, equal-parts-assumption sensitivity, literature-linked case studies, computed report), and `ingest/crawl_formulations.py`.
  - **Data fact that shapes the design:** IMPPAT formulation pages list ingredients and plant parts but **no proportions**, so the baseline is an assumed equal-parts split (and the report tests how much that assumption matters). Only the 20 scoped herbs can be re-weighted; other ingredients stay fixed and unscored.
  - Smoke run on 109 of 1,576 formulations: 60 had two or more scoped herbs; median risk reduction 4% (small, because scoped herbs are only ~20% of the ingredients); 57% of suggestions also improved under the independent silver risk (Spearman 0.64). Final numbers wait for the full crawl.
- **2026-10-02, Phase 4 complete.** Models (prior, matrix factorisation, Random Forest, hetero-GraphSAGE GNN, and a no-message-passing MLP ablation) trained on the **masked** graph and evaluated on 5-fold cold-compound, cold-herb, cold-drug and (inflated reference) random-pair splits; GNN/MLP use 3 seeds. Everything is in `docs/phase4_results.md` (reproduce: `python -m ayurveda_kg.phase4` then `python -m ayurveda_kg.phase4_report`; the run is resumable).
  - **Headline numbers (mean AUROC over folds):** cold-compound RF 0.945 (GNN 0.903, prior 0.714); cold-herb RF 0.885 +/- 0.052 (GNN 0.837, prior 0.704; folds range 0.795-0.930); cold-drug pooled RF 0.985 (GNN 0.969).
  - **What the grouped metrics revealed (checked after the pooled score looked too good):** cold-drug pooled AUROC is almost entirely *compound-side* skill (within-drug 0.989). *Drug-side* skill on unseen drugs (ranking drugs for a compound) is only 0.72 for RF, 0.60 for MLP and 0.53 (chance) for the GNN. So the models know which compounds look like CYP inhibitors but barely generalise to new drugs.
  - **Graph structure adds nothing:** GNN minus MLP (same network, 0 graph layers) = -0.003 (cold-compound), -0.005 (cold-herb), +0.014 (cold-drug, pooled; drug-side is worse). The GNN is not under-trained: five configurations gave validation AUROC 0.895-0.900 and test 0.91-0.92 on two folds (differences within noise), so we kept defaults rather than tuning.
  - **Leakage ablation:** the same RF with the unmasked CYP features reaches 0.998 / 0.999 / 0.997 AUROC (cold-compound / herb / drug) vs 0.945 / 0.885 / 0.985 masked, i.e. +0.053 / +0.114 / +0.012 inflation. This is the evidence that masking matters.
  - **Gold set:** both models inherit the silver false alarms: ginger-warfarin (a human-trial negative) gets the top percentile (1.00) from both models, above the median PK positive. Honest reading: the models reproduce the mechanistic silver rule; they do not predict clinical interactions better than silver does.
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

1. **Decision for you (Jivha/Nadi go/no-go):** do you have, or can you secure within about two weeks, a clinical partner (BAMS college/clinic) and ethics approval for tongue/nail/pulse data? If yes, Phase 7 gets a real data-collection pipeline; if not, Jivha/Nadi stays a designed-but-stubbed module and appears in the paper as future work. (Phases 5 and 6 do not depend on it.)
2. Before Phase 5, strengthen the weakest link: **drug-side generalisation** and **experimental (not predicted) CYP evidence**. Ideas: add drug chemistry fingerprints (ChEMBL SMILES), add ChEMBL experimental CYP-inhibition bioactivity for compounds that have ChEMBL IDs, and report results with and without SwissADME-derived labels.
3. Phase 5: safe-composition optimiser (uses the compound-level risk model from Phase 4; limitations above apply). Phase 6: GraphRAG.
4. Human review still needed: the 12 gold pairs (pharmacist/Ayurveda expert).
5. Rebuild everything from cached raw files (no network): `python -m ayurveda_kg.build`, `python -m ayurveda_kg.phase3`, `python -m ayurveda_kg.phase4`, `python -m ayurveda_kg.phase4_report`.

## 9. Downloads log (what was fetched, from where, why)

Everything downloaded is recorded here and, for data files, in `data/manifest.json` (source, licence, checksum). Raw data is never committed.

| Date | What | Source | Size | Purpose / licence note |
|---|---|---|---|---|
| 2026-10-01 | IMPPAT plant, compound and human-target pages (1,696 compounds) | cb.imsc.res.in/imppat | ~3,400 pages | Phase 1-2 graph. CC BY-NC-ND: not redistributed |
| 2026-10-01 | ChEMBL molecule searches + metabolism table; DGIdb TSVs; DDInter CSVs | EBI, dgidb.org, ddinter2.scbdd.com | ~50 MB | Drug side of the graph. Open licences |
| 2026-10-01 | TDC CYP2C9/2D6/3A4 substrate files | Harvard Dataverse | ~140 KB | Substrate and verified non-substrate labels |
| 2026-10-02 | PyTorch 2.11 (CUDA 12.8), PyG, RDKit, scikit-learn | PyTorch/PyPI | ~3 GB | Phase 4 models |
| 2026-10-02 | IMPPAT formulation pages (~1,576) and therapeutic-use pages (20 herbs) | cb.imsc.res.in/imppat | ~220 MB | Phase 5 composition optimiser. CC BY-NC-ND: not redistributed |
| 2026-10-02 | **Approved, Phase 6:** Charaka Samhita (Kaviratna) and Sushruta Samhita (Bhishagratna) English translations, plain text | archive.org / HathiTrust (public domain, US) | ~5-15 MB | GraphRAG corpus; edition and URLs recorded in `data/manifest.json` |
| 2026-10-02 | **Approved, Phase 6:** sentence-embedding model all-MiniLM-L6-v2 | Hugging Face | ~90 MB | Passage retrieval |
| not downloaded | AyurParam GGUF (community conversion, ~1.8 GB) | Hugging Face (arunmcops/AyurParam-GGUF) | ~1.8 GB | **Deferred by the project owner**; needed for the AyurParam-vs-general-LLM comparison. Existing `qwen3:8b` (Ollama, 5.2 GB, already installed) is used as the generator meanwhile |

## 10. Future improvements (backlog, to do for betterment)

**Data and labels**
- Ask IMSc for permission/bulk files for IMPPAT before any release of derived data. Apply for the DrugBank academic licence and add it as a plug-in.
- Add **experimental** CYP-inhibition bioactivity (ChEMBL) as an independent label source, so labels are not only SwissADME predictions; report results with and without the predicted labels.
- Extend the scope beyond 20 herbs and 35 drugs; add UGT and transporter (P-gp) label logic, not only the five CYPs.
- Have a pharmacist / Ayurveda expert review the 12 gold pairs; grow the gold set with more cited human studies (and more true negatives).

**Modelling (Phase 4)**
- Add drug chemistry fingerprints (weak drug-side generalisation is the main gap); scaffold split; probability calibration and uncertainty; compound abundance data if any source exists.
- The graph adds nothing over node features right now: test richer graph signals (e.g. pathway, disease, formulation context) before claiming a graph benefit.

**Composition (Phase 5)**
- Real proportions: mine classical texts or pharmacopoeia dosage tables instead of equal parts; handle pharmacodynamic interactions (invisible to the CYP model); multi-drug patients; efficacy beyond binary therapeutic-use labels; validate against a pharmacist.

**GraphRAG (Phase 6)**
- Add AyurParam (GGUF) for the AyurParam-vs-general-LLM comparison; run official AyurParam weights if hardware allows; add more texts (e.g. Ashtanga Hridaya, Bhavaprakasha) once licence-clean editions are found; Hindi evaluation slice.
- Replace the lexical/LLM-judge faithfulness proxies with an NLI model and, above all, an **expert-verified 50-100 question set**.

**Engineering / publication**
- Add CI (run tests on push); fix line-ending noise (CRLF) with a `.gitattributes`; consider data versioning; containerise for reproducibility; document GPU non-determinism in the GNN runs.
- Re-verify the LASI and other brief-sourced statistics near submission; choose venue; internal and mentor review.
