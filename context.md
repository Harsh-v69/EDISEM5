# Project Context: Ayurvedic Knowledge Graph for Herb-Drug Interaction Prediction

_Purpose of this file: a single, self-contained briefing that anyone (or any tool) can use to understand the project, then write a report, build slides, draft a paper section or answer questions, without reading the code. For live status see `progress.md`; for design detail see `docs/`._
_State described: 2026-10-03 (Phases 0-6 and 8 complete; Phase 7 not built). Numbers below are measured from the real build unless marked "planned"._

---

## 1. One-paragraph summary

Millions of people in India take Ayurvedic herbal medicines together with modern prescription drugs, but nobody checks the combination the way pharmacists check drug-drug interactions. This project builds a research system on **one shared Ayurvedic knowledge graph** (herbs -> chemical compounds -> protein targets/enzymes -> drugs) that (1) predicts herb-drug interaction (HDI) risk with a graph neural network, (2) suggests safer formulation compositions and proportions, (3) answers Ayurveda questions with cited, graph-grounded retrieval (GraphRAG), and (4) optionally accepts tongue/nail (Jivha) and pulse (Nadi) observations to estimate a dosha profile. All outputs are **research risk scores, not medical advice**. The goal is publication in a recognised bioinformatics / digital-health venue.

## 2. Problem and motivation

- **Real, documented co-use.** A 2025 study of the Longitudinal Ageing Study in India (LASI) found that among adults 45+, 6.68% used AYUSH therapies alongside conventional treatment in the past year, and 7.78% of people treated for diabetes combined AYUSH with conventional drugs; hypertension, acid reflux and arthritis were the most common conditions. _(Figure taken from the project brief; re-verify before submission.)_
- **Herbs are pharmacologically active.** Compounds in herbs can inhibit or induce drug-metabolising enzymes (notably the cytochrome P450 family, "CYP") or transporters (P-glycoprotein), changing how much of a drug reaches the bloodstream. Example outside India: St John's wort lowers the effect of blood thinners and contraceptives.
- **Structural gap.** A doctor prescribing a blood-pressure drug rarely sees the patient's herbal supplement; an Ayurvedic practitioner rarely sees the full prescription list. Drug-drug interaction checkers exist; an equivalent for Ayurvedic herbs does not.
- **Research gap.** AI for herb-drug interaction exists for Traditional Chinese Medicine and generally, but is thin for Ayurveda. TCM has a GraphRAG precedent (OpenTCM, 2025) and a tongue-image dataset (TCM-Tongue, 2025); Ayurveda has no equivalent of either. Sustainable Development Goals addressed: 3 (health), with 4 and 10 for the knowledge-access parts.

## 3. Goals and intended contributions

| # | Contribution | Status |
|---|---|---|
| C1 | A scoped, validated, source-attributed **Ayurvedic knowledge graph** linking herbs, compounds, targets/enzymes and drugs | **Built** (Phase 2) |
| C2 | An **HDI predictor** (GNN link prediction) evaluated against baselines with leakage-safe splits and a literature gold set | Labels, masking, splits, gold set (Phase 3) and baselines + GNN evaluation (Phase 4) **done**; see section 7.4 for the honest results |
| C3 | A **safe-composition suggester**: choose compounds/ratios that minimise predicted interaction risk while keeping therapeutic coverage | **Done** (Phase 5); see section 7.5 |
| C4 | **Ayurveda GraphRAG**: cited question answering over classical texts plus the same graph, compared to plain RAG | **Done** (Phase 6) except expert verification; see section 7.6 |
| C5 | **Jivha (tongue/nail) and Nadi (pulse)** input producing a dosha estimate | Gated on a clinical partner + ethics approval (Phase 7) |
| C6 | An honest **methodology contribution**: how to build and evaluate HDI labels without circularity (label-source edge masking, herb-wise splits, verified gold set) | Designed and partly implemented |

Everything is "decision support / research score". Claims of clinical validity are explicitly avoided.

## 4. System architecture

```
Raw sources (IMPPAT, ChEMBL, DGIdb, DDInter, TDC CYP labels, curated literature)
        |  ingest/ (polite cached download, checksummed manifest)
        v
Entity resolution (herb aliases, compounds by InChIKey, drugs by ChEMBL parent + aliases)
        v
Unified heterogeneous Knowledge Graph  (parquet tables -> NetworkX / PyG)
   Herb --contains--> Compound --modulates / predicted_cyp_inhibitor--> Target(gene)
   Drug --targets / substrate_of / non_substrate_of--> Target(gene);  Drug --ddi-- Drug
        |
        +--> HDI labels (silver, mechanistic) + gold (literature)  --> leak-safe splits
        |        --> baselines (Random Forest, matrix factorisation) --> GNN link predictor
        +--> Safe-composition optimiser (risk minimisation under coverage constraint)
        +--> GraphRAG (vector index of classical texts + graph traversal + local LLM, cited answers)
        +--> Jivha/Nadi module (image/signal -> dosha estimate -> patient node)   [gated]
        v
Demo app: risk lookup with graph-path explanation, cited chat, composition suggestions
```

Design principles: one graph reused by all modules; every edge keeps its **source**; raw data immutable and checksummed; every step test-first and validated before moving on; local open models only (no API keys); NetworkX + parquet rather than Neo4j (scale does not need it).

## 5. Data sources

| Source | What it provides | Access / licence | Role |
|---|---|---|---|
| IMPPAT 2.0 (IMSc Chennai) | 4,010 Indian medicinal plants, ~18k phytochemicals, formulations; per-compound InChIKey, SMILES, ChEMBL/PubChem IDs, target genes, SwissADME CYP-inhibition and P-gp predictions | Web access (bulk-download links are broken, HTTP 404); **CC BY-NC-ND 4.0: derived tables must not be redistributed** | herb-compound-target backbone |
| ChEMBL | Drug identity; whole metabolism (drug-enzyme) table, 2,147 rows | Open, CC BY-SA | drug identity, substrate edges |
| DGIdb | Drug-gene interactions | Open TSV | drug-target edges |
| DDInter 2.0 | Drug-drug interactions with severity | Open | drug-drug edges (auxiliary) |
| TDC / Carbon-Mangels & Hutter 2011 | CYP2C9/2D6/3A4 substrate **and verified non-substrate** labels (~665 drugs each) | Open, cite TDC | substrate edges + real negatives |
| Curated literature | 12-row gold HDI set; 3-row substrate supplement (FDA labels, PMID) | Hand-curated with citations | evaluation + gap filling |
| DrugBank | drug-target, enzymes | Needs academic licence (HTTP 403 without) | **not used**; optional plug-in |
| Classical texts, AyurParam | Charaka/Sushruta translations; open Ayurveda LLM | Public-domain/licensed only | Phase 6 |

Scope is **fixed on purpose** (no expansion later): 20 herbs and 35 drugs.
- Herbs: Curcuma longa (turmeric), Withania somnifera (ashwagandha), Glycyrrhiza glabra (licorice), Zingiber officinale (ginger), Allium sativum (garlic), Piper nigrum (black pepper), Trigonella foenum-graecum (fenugreek), Momordica charantia (bitter gourd), Tinospora sinensis (= T. cordifolia, guduchi), Bacopa monnieri, Ocimum tenuiflorum (tulsi), Commiphora wightii (guggul), Boswellia serrata, Gymnema sylvestre, Azadirachta indica (neem), Andrographis paniculata, Centella asiatica, Phyllanthus emblica (amla), Terminalia chebula, Rauvolfia serpentina.
- Drugs (by class): anticoagulants/antiplatelets (warfarin, clopidogrel, aspirin, rivaroxaban, apixaban); antidiabetics (metformin, glibenclamide, glimepiride, gliclazide, pioglitazone, sitagliptin); antihypertensives and diuretics (amlodipine, nifedipine, diltiazem, verapamil, atenolol, propranolol, metoprolol, losartan, enalapril, hydrochlorothiazide, furosemide); statins (atorvastatin, simvastatin, rosuvastatin); levothyroxine; immunosuppressants (cyclosporine, tacrolimus); digoxin; omeprazole; theophylline; anticonvulsants (carbamazepine, phenytoin); alprazolam; diclofenac.

## 6. Methods

### 6.1 Knowledge graph construction (done)
- Herb names resolved with aliases (e.g. IMPPAT files *Tinospora cordifolia* under *Tinospora sinensis*). Compounds keyed by full InChIKey (fallback IMPPAT ID so nothing is dropped). Drugs keyed to the ChEMBL **parent** molecule; per-drug alias lists (from ChEMBL synonyms) catch spellings such as glyburide/glibenclamide; ambiguous aliases are discarded.
- A validator checks: no duplicate node IDs, no dangling edges, edge endpoint types match the schema, every scoped herb/drug present, herbs with no compounds flagged. Plausibility tests check known pharmacology (below).
- Data hygiene lessons (see section 8): enzyme "families" such as ESTERASES are not gene nodes; ChEMBL alone covered only ~14 of 35 drugs' enzymes, so TDC labels and a cited supplement were added.

### 6.2 HDI labels (Phase 3, done)
- **Unit of prediction: (compound, drug)**, about 59,000 pairs. Herb-drug has only 20 x 35 = 700 pairs, too few to train or split; herb-level risk is aggregated afterwards.
- **Silver label** (mechanistic, used for training), over five CYP enzymes (CYP1A2, 2C9, 2C19, 2D6, 3A4):
  - **1** = compound predicted to inhibit enzyme X **and** the drug is a substrate of X;
  - **0** = for every enzyme the compound is predicted to inhibit, the drug is a verified non-substrate (or the compound inhibits none);
  - **-1 (unlabelled)** = undetermined; never guessed. Compounds with no SwissADME predictions are always unlabelled. If sources conflict, "substrate" wins and the conflict is reported (currently zero conflicts).
- **Gold set** (evaluation only, never training): 12 herb-drug pairs from published studies, each with PMID/DOI, mechanism type (PK / PD / none / unclear), evidence level, confidence and a verification note. Evidence read directly from PubMed abstracts / full-text passages via Europe PMC; rows are marked **pending expert review**.
- **Leakage control (key methodological point):** the silver label is computed from CYP edges that are also in the graph, so a GNN could simply rediscover the rule. Before training, all label-source edges (`predicted_cyp_inhibitor`, `substrate_of`, `non_substrate_of`, plus any `modulates`/`targets` edge into the five CYPs) are removed, and a test proves labels cannot be re-derived from the masked graph.
- **Splits:** cold-compound K-fold; cold-herb hold-out (test compounds belong only to held-out herbs; compounds shared with training herbs are dropped and counted); cold-drug folds; a random-pair split is provided only as a labelled "inflated" reference. Note: cold-compound still lets close chemical analogues straddle folds; a scaffold split is a Phase 4 option.

### 6.3 Models (Phase 4, done)
Baselines first (Random Forest on engineered features; matrix factorisation) so there is something to beat; then GraphSAGE and/or Graph Attention Network link predictors in PyTorch Geometric on the masked graph. Metrics: AUROC, AUPRC, precision at top-k, recall on the gold set (reported separately for PK pairs, because a CYP-based model cannot predict pharmacodynamic interactions), K-fold with multiple seeds (mean +/- std). Compound chemistry (fingerprints) may be added as node features.

### 6.4 Safe-composition suggester (Phase 5, done)
Constrained optimisation: given a formulation (set of herbs/compounds with proportions) and a patient's drug list, choose compounds and ratios that minimise aggregate predicted interaction risk subject to keeping therapeutic coverage (targets / dosha action of the original formulation) above a threshold and proportions summing to 1. Start with linear programming or greedy search. **Proportions are a risk proxy, not pharmacokinetics**, because no public dose-response data exists; outputs are hypothesis-generating. Validated against IMPPAT's real formulations (about 1,133).

### 6.4b GraphRAG (Phase 6, done)
Chunk classical texts (2-3 translated texts, public-domain/licensed only) into a vector index; extract entities/relations with a local LLM into the same graph (each edge keeps its source passage); hybrid retrieval merges vector hits with multi-hop graph traversal and re-ranks; a local model (AyurParam, benchmarked against a general open LLM) answers with inline citations. Evaluated on 50-100 expert-verified questions for faithfulness, hallucination rate, retrieval precision/recall, against a plain-RAG baseline. **Needs a domain advisor.**

### 6.5 Jivha and Nadi (Phase 7, gated)
Interface fixed first: image or signal in, dosha/prakriti estimate with uncertainty out, becoming a Patient node that biases graph queries. Pipeline developed on public proxy data only; no Ayurvedic accuracy claim from proxies. Real data collection requires a clinical partner (BAMS college/clinic) and ethics approval; validation is Cohen's kappa against independent practitioner labels, framed as decision support. Go/no-go checkpoint at the end of Phase 4; otherwise it appears as future work.

## 7. Results so far (measured)

### 7.1 Knowledge graph (final Phase 2 build, 0 validation problems)
| Item | Count |
|---|---|
| Herbs / Compounds / Drugs / Target genes | 20 / 1,696 / 35 / 1,667 |
| herb -> compound (`contains`) | 2,721 |
| compound -> target (`modulates`) | 11,947 |
| predicted CYP inhibition (SwissADME via IMPPAT) | 1,436 |
| predicted P-gp substrate | 482 |
| drug -> target (DGIdb) | 1,209 |
| drug substrate_of enzyme / verified non_substrate_of | 53 / 31 |
| drug-drug interactions (DDInter, scoped pairs) | 377 |

Compounds per herb range from 35 (Bacopa) to 343 (ginger); counts match IMPPAT's own plant pages exactly. 1,651 of 1,696 compounds have SwissADME predictions. **1,120 of 1,696 compounds (66%) have no compound-target edge**, so mechanism coverage is sparse.

Plausibility checks that pass on the real graph: black pepper contains piperine and turmeric contains curcumin; piperine links to ABCB1/P-gp and to CYP1A2/2C9/2C19; licorice has the most predicted CYP3A4 inhibitors (78), consistent with its reputation; warfarin is a CYP2C9 substrate and simvastatin a CYP3A4 substrate.

### 7.2 Silver labels (Phase 3, final)
59,360 compound-drug pairs: **10,235 positive**, **33,071 negative**, **16,054 unlabelled**, no source conflicts; 785 compounds have at least one positive. Positives concentrate on drugs metabolised by many CYPs (warfarin, omeprazole, carbamazepine, diltiazem); drugs with no CYP route (metformin, aspirin, atenolol, furosemide, digoxin) have none, as expected.

### 7.3 Silver labels vs. the literature gold set (important honest finding)
Herb-level silver score = fraction of the herb's labelled compounds that are positive for the drug; "percentile" is rank among all 700 herb-drug pairs.

| Gold pair (literature) | Gold | Silver score | Percentile | Reading |
|---|---|---|---|---|
| Piper nigrum + phenytoin / carbamazepine | interaction (human PK) | 0.48 / 0.49 | 94th / 95th | agrees |
| Piper nigrum + propranolol / theophylline | interaction (human PK) | 0.35 / 0.33 | 79th / 78th | agrees |
| Commiphora wightii + diltiazem / propranolol | interaction (human PK) | 0.57 / 0.43 | 99th / 86th | agrees |
| Trigonella + warfarin (case report, confounded) | interaction | 0.41 | 85th | weak agreement |
| Glycyrrhiza + hydrochlorothiazide / furosemide | interaction (pharmacodynamic) | 0.00 | 12th | expected miss: not a CYP mechanism |
| Curcuma + tacrolimus (case-level evidence) | interaction | 0.11 | 44th | weak |
| **Zingiber + warfarin** | **no interaction (RCT)** | **0.55** | **97th** | **silver false alarm** |
| Allium sativum + warfarin | no interaction (RCT) | 0.29 | 72nd | silver false alarm (milder) |

**Cold-herb folds (5 folds, 4 herbs held out each):** 108-303 compounds shared with training herbs are dropped per fold; test positives range from 412 to 3,373 per fold, so Phase 4 reports per-fold results. On the real graph, masking leaves 0 of 10,235 positives re-derivable.

**Take-away:** predicted CYP inhibition by compounds is not the same as a clinical interaction (dose, absorption and exposure matter). Ginger has many predicted CYP2C9 inhibitors but did not change warfarin PK/PD in a human trial. The silver labels are therefore a *mechanistic hypothesis*, and the paper should evaluate against clinical gold and discuss this gap openly. The gold set is small (12 pairs, 2 negatives), so it supports a qualitative case study, not statistical claims. Possible mitigations (Phase 4+): weight by compound abundance/potency, use experimental bioactivity where available, calibrate against gold.

### 7.4 Model results (Phase 4; `docs/phase4_results.md`)
Models: prior (per-drug base rate), matrix factorisation (reference only), Random Forest on chemistry + non-CYP targets + drug features, a heterogeneous GraphSAGE GNN, and the same network with **no message passing** (MLP ablation); a leakage-ablation RF given the unmasked CYP features. Mean AUROC over 5 folds (GNN/MLP: 3 seeds):

| Split | prior | RF | MLP (no graph) | GNN | RF with leaked CYP features |
|---|---|---|---|---|---|
| cold-compound | 0.714 | **0.945** | 0.906 | 0.903 | 0.998 |
| cold-herb (folds 0.795-0.930 for RF) | 0.704 | **0.885** | 0.842 | 0.837 | 0.999 |
| cold-drug (pooled) | 0.500 | **0.985** | 0.955 | 0.969 | 0.997 |
| random-pair (inflated reference) | 0.710 | 0.995 (MF 0.996) | 0.975 | 0.981 | 1.000 |

What this actually shows (state these plainly in any talk or paper):
1. **Compound-side skill is real:** chemistry plus targets predicts which unseen compounds look like CYP inhibitors (within-drug AUROC 0.94 cold-compound, 0.87 cold-herb). Caveat: the labels come from SwissADME predictions, which are themselves structure-derived, so this largely shows structure-to-predicted-activity learning, not new biology.
2. **Drug-side generalisation is weak:** cold-drug pooled AUROC (0.985) is almost entirely compound-side. Ranking unseen drugs for a given compound gives only 0.72 (RF), 0.60 (MLP), 0.53 (GNN, chance level).
3. **Graph structure adds nothing:** GNN minus MLP is -0.003, -0.005, +0.014 AUROC; the RF beats the GNN on every cold split. The GNN is not under-trained (several configurations gave the same validation AUROC).
4. **Leakage is a large effect:** unmasked CYP features inflate AUROC by +0.05 (cold-compound), +0.11 (cold-herb), +0.01 (cold-drug), which is why edge masking and cold splits are essential.
5. **Gold set:** both models inherit the silver false alarms (ginger-warfarin gets the top percentile, 1.00), and miss the pharmacodynamic licorice-diuretic pairs; they reproduce the mechanistic rule rather than clinical truth.
Planned improvements before the paper: drug chemistry features, experimental ChEMBL CYP bioactivity as an independent label source, and a scaffold split.

### 7.5 Safe-composition optimiser (Phase 5; `docs/phase5_results.md`)
A linear program re-weights the in-scope herbs of a real IMPPAT formulation to lower the predicted interaction risk against one patient drug, keeping each herb's share within bounds of its baseline and every therapeutic use above a coverage floor. IMPPAT gives ingredients but **no proportions**, so the baseline is an assumed equal-parts split.
- 1,573 formulations parsed; 777 contain at least one scoped herb; **529** contain two or more (the minimum for re-weighting). 18,515 scenarios (529 x 35 drugs), all solved.
- Median risk reduction **3.7%** (90th percentile 6.6%): small, because scoped herbs are only ~20% of a typical formulation. Constraints verified: coverage never below the 0.80 floor, no herb below half its share. Looser constraints reach 9.3%; random baselines give 2.7%, so the finding survives the unknown-proportions assumption.
- 57% of suggestions also improve under the independent silver risk (Spearman 0.61).
- Case studies make the weakness visible: ginger and garlic with warfarin (human-trial negatives) are lowered just like piperine with phenytoin (a true positive), because the risk proxy has known false alarms. Output is a hypothesis, not a dosing recommendation.

### 7.6 GraphRAG (Phase 6; `docs/phase6_results.md`)
Corpus: Kaviratna *Charaka-Samhita* and Bhishagratna *Sushruta Samhita* (public-domain translations), 6,493 passages. Components: alias-aware entity lexicon, a text graph (211 co-occurrence edges plus 532 LLM-extracted triples accepted by a grounding check, 710 rejected), hybrid retrieval (dense + alias expansion + text graph + KG facts, rank-fused), citation-constrained generation with `qwen3:8b` (local), automatic faithfulness proxies, and a disclaimer appended by code.
- **Retrieval (60 synthetic questions):** graph facts lift recall@5 0.450 to 0.533. **Vocabulary gap** (English herb name in the question, Sanskrit name in the passage): recall@5 plain 0.017, alias 0.067, graph 0.083, alias+graph 0.167: a large relative gain but still weak.
- **KG-grounded questions (graph advantage by construction):** full system answers 83% of risk-score questions, 75% of top-drugs and 97% of held-out top-herbs questions; plain RAG correctly refuses (0%).
- **Honest findings:** a small local model sometimes refuses despite having the answer; putting KG facts first in the prompt was worse, not better; the model turned "co-mentioned" into "balances" until the wording and prompt were fixed (1 of 16 sentences over-read before, 0 of 17 after; small samples); an irrelevant passage can still be blended into an answer, which the automatic proxy cannot see. An early automatic check wrongly reported the fix as harmful because it ignored negation; it was repaired and recounted.
- **Not claimed:** correctness on real questions. The expert-verified question set (template ready) and the AyurParam comparison (not approved for download) are pending.

### 7.7 Demo and paper drafts (Phase 8)
- **Demo:** a local Streamlit app (risk lookup with graph-path explanation and the published study where one exists; formulation re-weighting for several drugs; cited Q&A). Local only, because IMPPAT-derived data cannot be redistributed.
- **Drafts:** `docs/paper/paper1_draft.md` (knowledge graph, leakage-aware benchmark with honest negative results, composition case study) and `paper2_draft.md` (GraphRAG: vocabulary gap, over-reading, small-model failure modes). Rendered from templates so every number comes from the result files; a test asserts each directional claim against the real numbers.
- **Direction-of-effect check (composition):** for herbs with published pharmacokinetic interactions the optimiser lowers their share in 82% of scenarios on average (73% to 100% per pair), but it also lowers ginger in 99% of formulations although its warfarin trial was negative.

## 8. Data-quality bugs found by checking real data (not just unit tests)
1. Drug lookup returned a duplicate ChEMBL entry instead of the parent molecule (theophylline).
2. Enzyme families (ESTERASES, UGT) were created as fake "gene" nodes.
3. 11 drugs had zero DGIdb targets because of name variants (glyburide vs glibenclamide, salt forms); fixed with an alias table that refuses ambiguous aliases.
4. ChEMBL alone gave substrate information for only ~14 of 35 drugs; added TDC labels (also gives verified negatives) and a cited supplement for rivaroxaban, apixaban (CYP3A4; FDA labels) and theophylline (CYP1A2; PMID 7619675).
5. A cached empty file would never be re-downloaded (fixed, tested).
6. IMPPAT's advertised bulk files are dead links; replaced with a polite, resumable, scoped crawl (1,696 compounds, 3,392 pages, 1 request/second, cached, checksummed).

## 9. Evaluation plan (summary)
- Prediction: AUROC, AUPRC, precision@k on held-out compounds / herbs / drugs vs Random Forest and matrix-factorisation baselines; gold-set recall (PK subset reported separately); multi-seed means and standard deviations; ablations (with/without chemistry features, with/without masked edges to show leakage inflation).
- Composition: validity against IMPPAT formulations; sensitivity of suggestions to the risk model.
- GraphRAG: faithfulness, hallucination rate, retrieval P/R vs plain RAG on 50-100 expert questions.
- Jivha/Nadi: Cohen's kappa vs independent practitioners; inter-rater agreement as context.

## 10. Limitations, ethics and safety
- **Not medical advice.** Output is a research risk score; every demo and paper says so.
- Many edges are **in-silico predictions** (SwissADME, STITCH-style target links); the graph keeps the source on each edge so predicted and experimental evidence can be separated.
- **Silver labels are mechanistic hypotheses** with demonstrated false alarms (ginger-warfarin); gold set is tiny and pending expert review.
- 66% of compounds lack target edges; only 20 herbs and 35 drugs are in scope; five CYPs only (no UGT/transporter label logic yet); pharmacodynamic interactions are out of reach for a CYP-based model.
- **IMPPAT licence (CC BY-NC-ND):** the paper/repo release code and identifiers only, not IMPPAT-derived tables; permission/bulk files should be requested from IMSc before publication.
- Jivha/Nadi needs ethics approval and a clinical partner; no patient data is used.

## 11. Publication strategy
One system, several papers (five modules are too much evidence for one venue):
- **Paper 1 (primary):** KG + HDI-GNN + safe-composition (+ the leakage-aware label methodology and the gold-set case study).
- **Paper 2:** GraphRAG + patient-dosha personalisation.
- **Paper 3 (optional):** Jivha/Nadi, only if clinical data is secured.
Venue undecided; built to a bioinformatics-journal standard (candidates: Briefings in Bioinformatics, Journal of Cheminformatics, Computers in Biology and Medicine; workshop options at BIBM / ACM-BCB; domain journals such as Journal of Ethnopharmacology if the framing shifts to case studies). The source brief suggests posting an arXiv preprint the same week as submission.

## 12. Roadmap and status
| Phase | What | Status |
|---|---|---|
| 0 | Scaffold, manifest, downloader, access audit | Done |
| 1 | Raw data for scoped herbs/drugs | Done |
| 2 | Entity resolution + unified KG | Done |
| 3 | HDI labels, leakage masking, splits, gold set | **Done** (`docs/labels_report.md`) |
| 4 | Baselines, GNN, evaluation; Jivha/Nadi go/no-go | **Done** (`docs/phase4_results.md`); go/no-go awaits the project owner |
| 5 | Safe-composition optimiser | **Done** (`docs/phase5_results.md`) |
| 6 | GraphRAG + evaluation | **Done** (`docs/phase6_results.md`); expert question set pending |
| 7 | Jivha/Nadi pipeline (if go) | **Not built** (no clinical partner/ethics approval); protocol + tested kappa code ready |
| 8 | Demo + paper drafts | **Done**: local demo, two drafts rendered from result files, README, requirements |

External dependencies: a domain advisor (Ayurveda expert/pharmacist) for gold-set and GraphRAG question verification; a clinical partner + ethics approval for Jivha/Nadi; IMSc permission for IMPPAT-derived releases.

## 13. Reproducibility and repository map
- Environment: Python 3.12, local virtualenv `.venv`; hardware RTX 4060 (8 GB) laptop, ~15 GB RAM; local open LLMs only.
- Rebuild the KG from cached raw files (no network): `python -m ayurveda_kg.build` -> `data/processed/kg/` and `docs/kg_report.md`. Run tests: `python -m pytest -q`.
- Layout: `ayurveda_kg/` (code: `ingest/`, `resolve.py`, `kg.py`, `validate.py`, `build.py`, `labels.py`, `masking.py`, `splits.py`, `curated.py`, `report.py`), `config/scope.yaml` (fixed scope), `data/gold/` and `data/curated/` (cited, hand-curated), `data/manifest.json` (source, licence, checksum of every raw file), `docs/` (design spec, plans, schema, audits, reports), `tests/` (automated checks), `progress.md` (live status), this file.
- Raw and IMPPAT-derived data are **not** committed (licence); they are regenerated by the crawler and `build`.

## 14. Slide and report kit

**Suggested 12-slide outline:** (1) Problem: herbs + drugs, nobody checks; (2) Evidence of scale (LASI numbers) and the gap; (3) Idea: one knowledge graph, many tools; (4) Architecture diagram (section 4); (5) Data and scope (20 herbs, 35 drugs, sources); (6) Building the graph and the bugs real data revealed; (7) The label problem: why circular labels fool GNNs; (8) Our fix: silver vs gold, edge masking, herb-wise splits; (9) Results so far: KG numbers, label counts, plausibility; (10) Honest finding: ginger-warfarin false alarm and what it teaches; (11) Next: GNN, safe composition, GraphRAG, Jivha/Nadi gate; (12) Ethics, licence and limitations; publication plan.

**Figures worth drawing:** the architecture diagram; the KG schema (node/edge types); a worked example path *piperine -> CYP2C19 <- phenytoin*; a bar chart of compounds per herb; the gold-vs-silver percentile plot (section 7.3); a diagram of masked edges (what the model may and may not see).

**Quotable numbers:** 529 re-weightable real formulations, median 3.7% modelled risk reduction; 6,493 passages, vocabulary-gap recall@5 0.017 to 0.167 with alias+graph; 20 herbs, 1,696 compounds, 35 drugs, 1,667 target genes; 11,947 compound-target edges; 59,360 labelled pairs (10,235 positive); 12-pair cited gold set; 250 automated tests (as of this writing); 3,392 pages crawled politely at 1 request/second.

**Anticipated reviewer questions:** Does the graph help? (No: GNN = MLP; an honest negative result.) Are cold-drug numbers inflated? (Pooled yes; drug-side AUROC is reported separately.) Why are labels not circular? (masking + herb-wise splits + clinical gold.) Is this clinically valid? (No; research score; gold shows false alarms.) Why only 20 herbs? (scope control; extensible via config.) Why not DrugBank? (licence gate; optional plug-in.) Can you release the data? (code + identifiers; IMPPAT licence forbids derivatives.) How do proportions get optimised without dose data? (risk proxy, hypothesis-generating, stated plainly.)

## 15. Key verified references (read from primary records, 2026-10-02)
- Bano G et al. 1991, Eur J Clin Pharmacol 41:615-617 (PMID 1815977): piperine raises propranolol and theophylline exposure.
- Pattanaik S et al. 2006, Phytother Res 20:683-686 (PMID 16767797): piperine raises steady-state phenytoin AUC/Cmax. Pattanaik S et al. 2009, Phytother Res 23:1281-1286 (PMID 19283724): piperine raises carbamazepine AUC.
- Dalvi SS et al. 1994, J Assoc Physicians India 42:454-455 (PMID 7852226): gugulipid reduces propranolol and diltiazem Cmax/AUC.
- Jiang X et al. 2005, Br J Clin Pharmacol 59:425-432 (PMID 15801937): ginger does not affect warfarin PK/PD. Mohammed Abdul MI et al. 2008, Br J Pharmacol 154:1691-1700 (PMID 18516070): garlic does not significantly alter warfarin PK/PD.
- Lambert JP & Cormier J 2001, Pharmacotherapy 21:509-512 (PMID 11310527): warfarin and boldo-fenugreek case report (confounded).
- Licorice pseudoaldosteronism review, Front Nutr 2021 (PMC8484325): diuretics raise hypokalemia risk. Tacrolimus-herb review, Pharmaceutics 2022 (PMC9611668): turmeric and tacrolimus.
- Substrate supplement: ELIQUIS and rivaroxaban prescribing information (CYP3A4); theophylline CYP1A2, PMID 7619675.
- Context from the project brief: OpenTCM (GraphRAG for TCM, 2025); AyurParam (open Ayurveda LLM); TCM-Tongue (arXiv 2507.18288); LASI AYUSH-use study (2025). Re-verify any statistic before submission.

## 16. Glossary
**Herb / compound / target / drug:** a plant / a chemical in it / a protein it acts on / a prescription medicine. **CYP:** liver enzymes (cytochrome P450) that break down most drugs; **substrate** = broken down by it, **inhibitor** = blocks it. **P-gp (ABCB1):** a transporter that pushes drugs out of cells. **PK / PD:** pharmacokinetic (how much drug reaches the body) / pharmacodynamic (what it does) interaction. **KG:** knowledge graph. **GNN:** graph neural network. **Link prediction:** predicting whether an edge (here a herb/compound-drug interaction) exists. **RAG / GraphRAG:** retrieval-augmented generation (with a graph). **Silver / gold labels:** mechanistically derived training labels / hand-curated literature labels used only for evaluation. **Cold split:** test items (compounds, herbs, drugs) never seen in training. **Jivha / Nadi:** tongue (and nail) / pulse examination. **Dosha / prakriti:** Vata, Pitta, Kapha / a person's constitution. **AUROC / AUPRC:** ranking-quality metrics; AUPRC is more informative when positives are rare.
