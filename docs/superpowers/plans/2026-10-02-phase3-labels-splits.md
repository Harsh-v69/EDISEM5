# Phase 3 Plan: HDI labels, leakage control, splits, gold set

**Goal:** Produce labelled compound-drug pairs (silver, mechanistic), a verified literature gold set, leakage-safe splits, and a masked training graph, each with automated checks.

**Spec:** `docs/superpowers/specs/2026-10-01-ayurveda-kg-hdi-design.md` section 6.2. **Schema/leakage note:** `docs/schema.md`.

## Design decisions (made from Phase 2 findings)

1. **Label unit = (compound, drug)** (~59k pairs). Herb-drug risk is aggregated later (Phase 4/5); only 700 herb-drug pairs exist, too few to train or split.
2. **Enzymes in scope:** CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4 (the five SwissADME predicts; substrate data exists for them via ChEMBL + TDC).
3. **Silver label** for compound c, drug d (only compounds with SwissADME predictions are labelled):
   - `1`: some enzyme X where c is a predicted inhibitor of X AND d is a substrate of X.
   - `0`: for every X that c is predicted to inhibit, d is a verified non-substrate of X (vacuously true when c inhibits none of the five).
   - `-1` (unlabelled): c inhibits some X for which d's substrate status is unknown. Never guessed.
   - Substrate beats non-substrate if sources conflict; conflicts are counted and reported.
4. **Leakage control:** the label-defining edges (`predicted_cyp_inhibitor`, `substrate_of`, `non_substrate_of`, and any `modulates`/`targets` edge into the five CYPs) are removed from the training graph. A test proves that silver labels computed from the masked graph yield no positives.
5. **Splits:** (a) cold-compound K-fold (no compound in two folds), (b) cold-herb hold-out (test compounds belong only to held-out herbs; compounds shared with training herbs are dropped and counted), (c) cold-drug hold-out. Random-pair split is provided only as a labelled "inflated" reference.
6. **Gold set:** hand-curated herb-drug pairs from published studies, each with citation (PMID/DOI), mechanism type (`PK_CYP`, `PK_transporter`, `PD`, `none`), evidence level, and an explicit `verified_by: assistant_web_check, pending_expert_review` flag. Evaluation only; never used for training. PD interactions are kept but reported separately because a CYP-based model cannot predict them.

## Tasks (each: tests first, then code, then run, then progress.md)

- 3.1 `has_adme` compound flag (build.py) and KG rebuild.
- 3.2 `labels.py::silver_labels`.
- 3.3 `masking.py::mask_label_edges`, `assert_no_label_leak`.
- 3.4 `splits.py`: `cold_compound_folds`, `herb_holdout`, `cold_drug_holdout`, `assert_disjoint`.
- 3.5 `gold.py` + `data/gold/gold_hdi.csv` (web-verified citations) + validator.
- 3.6 Run on the real KG; `docs/labels_report.md`; sanity checks.
- 3.7 Gate: all tests pass, leakage tests pass on real data, `progress.md` updated.
