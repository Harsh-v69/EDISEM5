# Phase 4 Plan: baselines, GNN, evaluation

**Goal:** Train and honestly evaluate models that predict the Phase 3 silver labels from the **masked** graph, on leak-free cold splits, against baselines, with a leakage ablation and a herb-level check against the literature gold set.

**Spec:** `docs/superpowers/specs/2026-10-01-ayurveda-kg-hdi-design.md` section 6.3. **Inputs:** Phase 3 labels, masking and splits (`docs/superpowers/plans/2026-10-02-phase3-labels-splits.md`).

## Design decisions

1. **Task:** binary prediction of the silver label (labelled pairs only) for a (compound, drug) pair. Features may use only the masked graph: compound chemistry (RDKit Morgan fingerprint of the SMILES), compound non-CYP targets, predicted P-gp substrate flag, drug non-CYP targets (DGIdb), drug class, drug-drug interaction profile.
2. **Why this is not trivially circular:** the label depends on hidden CYP inhibition (compound side) and hidden CYP substrate status (drug side). The model must infer both from other evidence. Compound inhibition is structure-derived (SwissADME), so chemistry should help on cold compounds; the drug side is the hard part on cold drugs.
3. **Splits:** cold-compound 5-fold, cold-herb 5-fold (Phase 3 folds), cold-drug 5-fold, plus a random-pair split labelled "inflated reference". The training graph of every fold excludes the test nodes entirely (no test compounds, and no test drugs for cold-drug).
4. **Models:** (a) prior-only (drug base rate), (b) Random Forest on pair features, (c) matrix factorisation (only meaningful on random-pair: it cannot embed unseen compounds, reported as n/a on cold splits), (d) heterogeneous GraphSAGE link predictor with inductive node features.
5. **Metrics:** AUROC, AUPRC, precision@100, per fold; report mean and std over folds, and per-fold values (folds are very uneven). Multiple seeds for stochastic models.
6. **Ablation:** same RF with the CYP label-source features *unmasked* (compound CYP-inhibition vector + drug CYP-substrate vector), to quantify leakage inflation.
7. **Herb-level / gold:** aggregate predicted probabilities over a herb's exclusive (test-fold) compounds; report the percentile of each gold pair among all herb-drug pairs, descriptively (12 gold pairs, 2 negatives: no statistical claims).
8. **Jivha/Nadi go/no-go:** decided by the user at the end of this phase (needs a clinical partner + ethics approval).

## Tasks (tests first, validate, then progress.md)

- 4.0 Environment: PyTorch (CUDA), PyG, RDKit installed and verified.
- 4.1 `features.py`: fingerprints, compound/drug feature matrices from the masked graph.
- 4.2 `evaluate.py`: metrics + generic cross-validation runner over split types (asserts no train/test overlap).
- 4.3 `baselines.py`: prior, Random Forest, matrix factorisation, leaky ablation.
- 4.4 `gnn.py`: hetero GraphSAGE, training loop, inductive inference.
- 4.5 Run all experiments; `docs/phase4_results.md`; gold/herb-level check.
- 4.6 Gate: all tests pass, results reproducible from seeds, `progress.md` and `context.md` updated, go/no-go question asked.
