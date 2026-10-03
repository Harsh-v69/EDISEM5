# Phase 8 Plan: demo app and paper drafts

**Goal:** A local demo that lets a user look up a herb-drug research risk score with its graph-path explanation, re-weight a real formulation for a patient's drugs, and ask cited questions; plus two paper drafts whose every number is generated from the project's result files; plus a clean reproducible repository.

## Decisions
1. **Phase 7 (Jivha/Nadi) is not built** (no clinical partner or ethics approval). Partner-independent deliverables are written so the module can start the day a partner exists: capture protocol, annotation guide, ethics checklist, and a tested Cohen's kappa validation script. No Ayurvedic accuracy claim is made.
2. **The demo is local-only.** It reads IMPPAT-derived tables; IMPPAT's licence (CC BY-NC-ND) forbids redistributing them, so it must not be hosted publicly. Documented in the README and in the app.
3. **Thin UI, tested logic.** All behaviour lives in `ayurveda_kg/demo/service.py` as pure functions with tests; the Streamlit app only renders. The app is also smoke-tested headlessly and checked in a real browser.
4. **Explanations show real evidence:** for a herb-drug pair, the herb's compounds with the highest out-of-fold predicted risk, the CYP enzyme path (compound predicted inhibitor of X, drug substrate of X, with sources), and, where one exists, the published human study from the gold set.
5. **Papers are rendered from templates.** `docs/paper/*.md.tmpl` contain placeholders such as `{{p4.cold_compound.rf.auroc}}`; `ayurveda_kg/paper.py` fills them from the result files, so a number can never be mistyped, and a test fails if any placeholder is unresolved. References are limited to sources actually seen during the project; anything else is marked `[VERIFY]`.
6. **Every claim keeps its caveat:** silver labels are mechanistic hypotheses with known false alarms, the gold set is 12 pairs pending expert review, GraphRAG has no expert-verified set yet, graph structure added nothing over features.

## Tasks (tests first, validate, then progress.md)
- 8.1 `demo/service.py`: pair explanation, composition suggestion (multi-drug), retrieval/chat wrapper.
- 8.2 `demo/app.py` (Streamlit): three tabs; headless smoke test; live browser check.
- 8.3 Phase 7 partner-independent deliverables: protocol doc, kappa script and tests.
- 8.4 `paper.py` + two templates (Paper 1: KG, leakage-aware benchmark, composition; Paper 2: GraphRAG) + render test; rendered drafts.
- 8.5 README, docs index, reproduce-from-clean-clone check.
- 8.6 Final gate: all tests pass, clean-clone test run, `progress.md` / `context.md` updated, commit.
