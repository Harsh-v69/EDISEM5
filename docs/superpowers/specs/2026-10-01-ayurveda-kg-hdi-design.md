# Ayurveda Knowledge Graph, Herb-Drug Interaction GNN, Safe Composition, GraphRAG and Jivha/Nadi: Design Spec

Date: 2026-10-01. Status: design approved in chat, awaiting spec review.
Source brief: `Ayurveda_Deep_Dive_Ideas_3_4_5.pdf` (Ideas 3, 4, 5) plus the added "safe composition" module.

## 1. Goal

One system, built on a single Ayurvedic Knowledge Graph (KG), that:

1. Predicts herb-drug interaction (HDI) risk with a GNN.
2. Suggests formulation compositions and proportions that avoid predicted interactions with a patient's drugs.
3. Answers Ayurveda questions with cited, graph-grounded retrieval (GraphRAG).
4. Accepts Jivha (tongue and nail) and Nadi (pulse) observations as patient input that yields a dosha/prakriti estimate (gated module).

Outputs are research risk scores and decision support. They are never medical advice or diagnosis. Every demo and paper must say so.

## 2. Publication strategy

One system, several papers. Five modules are too much evidence for one venue.

| Paper | Content | Status |
|---|---|---|
| P1 (primary) | KG + HDI-GNN + safe-composition | Core |
| P2 | GraphRAG + patient-dosha personalisation | Core |
| P3 | Jivha/Nadi | Only if clinical data is secured (go/no-go gate) |

Venue is undecided. The evaluation is built to a bioinformatics-journal bar: baselines, ablations, seeded and reproducible runs, limitations stated honestly.

## 3. Fixed decisions (from user)

- Jivha/Nadi: clinical partner "maybe, in progress". The core is built first and Jivha/Nadi sits behind a go/no-go gate at the end of Phase 4.
- LLM: local open models only (AyurParam, Qwen or Llama via Ollama or HF). No API key in the repo.
- Hardware: RTX 4060 Laptop, 8 GB VRAM, about 15 GB RAM, about 41 GB free disk. Quantised 7-8B LLM and PyG GNNs fit. Neo4j-scale graphs do not, so use NetworkX + parquet.
- Data strategy: open-data-first (Approach 1). DrugBank is an optional plug-in if the user obtains the academic licence. The assistant never creates accounts or enters credentials.

## 4. Data sources and known access constraints

| Source | Gives | Access reality |
|---|---|---|
| IMPPAT 2.0 (IMSc) | Herb, compound, therapeutic use, formulation | No bulk download, browse-only, CC non-commercial. Ask IMSc for a dump. Fall back to rate-limited, polite scraping with a cached raw snapshot. |
| PubChem | Compound structures, identifiers | Open REST API |
| ChEMBL | Compound-target bioactivity (CYP, transporters) | Open download |
| DGIdb | Drug-gene interactions | Open TSV |
| DDInter 2.0 | Drug-drug interactions, drug info | Open |
| DrugBank | Drug-target, substrates | Academic application required, XML currently unreliable. Optional. |
| Classical texts (Charaka, Sushruta) | Passages for RAG | Public-domain or licensed translations only. Corpus provenance recorded. |

Every dataset gets a manifest entry: source URL, date fetched, licence, checksum. `data/raw` is never hand-edited.

## 5. Architecture

Package `ayurveda_kg/` with independent modules. Each module states its inputs and outputs and has its own tests.

```
ingest/   -> raw snapshots + manifest            (no logic beyond fetch/parse)
resolve/  -> canonical entity tables             (herb/compound/drug/target name + ID resolution)
kg/       -> unified heterogeneous graph         (NetworkX in memory, parquet on disk, PyG export)
hdi/      -> labels, splits, baselines, GNN, eval
compose/  -> safe-composition optimiser
rag/      -> text chunks, vector index, hybrid retrieval, cited generation, eval
vision_nadi/ -> Jivha/Nadi interface + pipeline (gated)
app/      -> demo (risk lookup, graph-path explanation, chat with citations)
```

Node types: Herb, Compound, Target (incl. CYP/transporters), Drug, Disease/TherapeuticUse, Dosha, Formulation, Passage, Patient (for dosha personalisation).
Edge types: contains, binds/modulates, substrate_of, treats/indicates, balances, part_of (formulation), cites, interacts_with (HDI labels; held separate from the training graph, see 6.2).

## 6. Module design

### 6.1 Graph
Entity resolution merges spelling variants of herb names (Latin binomial as anchor, Sanskrit/common names as aliases), compounds by InChIKey, drugs by a normalised name mapped to ChEMBL/DDInter IDs. The graph carries a documented schema (`docs/schema.md`) and a build report with counts, duplicate-merge rates and orphan checks.

### 6.2 HDI labels and leakage control (most important scientific risk)
- **Silver labels:** herb compound modulates CYP/transporter X (ChEMBL evidence threshold) and the drug is a known substrate of X, which implies a mechanistic interaction candidate. Used for training and cross-validation.
- **Gold labels:** hand-curated herb-drug pairs from published case reports and reviews, each with a citation. Used for evaluation only. Never in training.
- **Leakage rule:** the silver label is computed from CYP edges that also exist in the graph. During training these edges (and any edge path that trivially reconstructs the label) are masked, and splits are done by herb, not random pairs, so the model must generalise to unseen herbs. A test asserts that no masked edge appears in the training graph.
- Negative sampling strategy is documented, and the sparse positive labels limitation is reported in the paper.

### 6.3 Models
Baselines first: Random Forest on hand-built features and matrix factorisation. Then GraphSAGE and a GAT (or HGT/R-GCN) link predictor. Metrics: AUROC, AUPRC, precision@k, gold-set recall. K-fold CV, multiple seeds, mean and std reported.

### 6.4 Safe composition
Constrained optimisation over a formulation's herb or compound mix:
minimise aggregate predicted interaction risk against the patient's drug list, subject to therapeutic coverage of the original formulation (targets and dosha action) staying above a threshold and proportions summing to 1 within bounds. Start with LP or greedy search. Proportions are a risk-proxy, not pharmacokinetics (no public dose-response data). The output is hypothesis-generating and is labelled so. It is validated against IMPPAT's real formulations, for example by checking that it does not suggest removing the defining herb of a classical formulation without flagging it.

### 6.5 GraphRAG
Chunk texts, embed with a sentence-transformer, index in FAISS. Extract entities and relations with a local LLM into the same KG (source passage attached to every edge). Hybrid retrieval merges vector hits and multi-hop graph traversal, re-ranks, and generates with inline citations. Baselines: plain RAG, and a general open LLM versus AyurParam. Metrics: faithfulness, hallucination rate, retrieval precision and recall, on a 50-100 question set. **Dependency: a domain advisor to verify the answers**, and to spot-check extraction.

### 6.6 Jivha/Nadi (gated)
Interface is fixed now: image or signal in, dosha/prakriti score with uncertainty out. The score becomes a Patient node that biases graph queries. Pipeline is developed and tested on public proxy data (tongue datasets, public PPG signals). No Ayurvedic accuracy claim is made from proxy data. Deliverables that do not need a partner: capture protocol, annotation guide, ethics application draft, Cohen's kappa validation script. Real data collection starts only after the partner and ethics approval exist. Go/no-go checkpoint at the end of Phase 4.

## 7. Validation policy

After every logical step:
1. Unit and data-integrity tests pass (`pytest`). Examples: schema conformance, no dangling edges, dedup rate reported, label-leakage test, split-by-herb test.
2. A runnable smoke check on a tiny slice exercises the step end to end.
3. All experiments are seeded. Configs and metrics are logged under `runs/`.
4. `progress.md` is updated (what was done, result of validation, what is next) only after the checks pass.

## 8. Phases and deliverables

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Scaffold, `progress.md`, data manifest, access audit | Tests run; manifest created |
| 1 | Ingested raw snapshots (IMPPAT, PubChem, ChEMBL, DGIdb, DDInter) | Parsers tested on fixtures; counts logged |
| 2 | Resolved entities and unified KG | Schema and orphan tests pass |
| 3 | Silver and gold labels, leak-safe splits | Leakage test passes |
| 4 | Baselines then GNN, with results | GNN vs baselines on held-out herbs; **Jivha/Nadi go/no-go** |
| 5 | Safe-composition optimiser | Validated against IMPPAT formulations |
| 6 | GraphRAG and evaluation set | Eval table vs plain RAG |
| 7 | Jivha/Nadi pipeline (if go) | Kappa script on proxy data |
| 8 | Demo app and paper drafts | Reproducible from `README` |

Scope control: fix the herb and drug list in Phase 1 (one or two herb families and a defined drug-class list, for example anticoagulants, antidiabetics, antihypertensives, thyroid drugs) and do not expand it later. Keep the first GraphRAG corpus to 2-3 texts.

## 9. Risks

| Risk | Mitigation |
|---|---|
| IMPPAT not bulk-downloadable | Ask IMSc; polite scrape with cache; fixtures for tests |
| Sparse or circular labels | Two-tier labels, herb-wise splits, masking test, honest limitation section |
| Overclaiming clinical validity | "Research risk score" wording everywhere |
| Extraction errors in KG | Advisor spot-check sample; report precision |
| 8 GB VRAM ceiling | Quantised local LLM; small GNNs; CPU fallbacks |
| Jivha/Nadi data never arrives | Gate at Phase 4; future-work framing |
| No expert for GraphRAG question set | Flag early; it blocks P2 evaluation only |
| Project dir is not a git repo | Offer `git init` for versioning |

## 10. Out of scope (YAGNI)

Neo4j, a web front-end beyond a simple demo, clinical dosing advice, all-of-Ayurveda coverage, real patient data before ethics approval.
