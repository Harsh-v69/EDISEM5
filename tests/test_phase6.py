import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.phase6 import make_phase6_report, summarize_answers

ANS = pd.DataFrame([
    {"system": "plain", "qtype": "top_drugs", "correct": 0.0, "citation_validity": 1.0, "support_rate": 1.0, "refused": True},
    {"system": "plain", "qtype": "top_drugs", "correct": 0.0, "citation_validity": np.nan, "support_rate": np.nan, "refused": True},
    {"system": "full", "qtype": "top_drugs", "correct": 1.0, "citation_validity": 1.0, "support_rate": 1.0, "refused": False},
    {"system": "full", "qtype": "top_drugs", "correct": 2 / 3, "citation_validity": 1.0, "support_rate": 0.5, "refused": False},
    {"system": "plain", "qtype": "pair_score", "correct": 0.0, "citation_validity": np.nan, "support_rate": np.nan, "refused": True},
    {"system": "full", "qtype": "pair_score", "correct": 1.0, "citation_validity": 1.0, "support_rate": 1.0, "refused": False}])


def test_summarize_answers_means_ignore_nan_and_report_refusal_rate_and_counts():
    s = summarize_answers(ANS).set_index(["system", "qtype"])
    assert s.loc[("full", "top_drugs"), "correctness"] == pytest.approx((1 + 2 / 3) / 2)
    assert s.loc[("plain", "top_drugs"), "citation_validity"] == 1.0                    # NaN row ignored, not counted as 0
    assert s.loc[("plain", "top_drugs"), "refusal_rate"] == 1.0 and s.loc[("full", "pair_score"), "n"] == 1


def test_report_has_all_sections_with_computed_numbers_and_honest_caveats():
    stats = {"n_chunks": 6493, "per_text": {"Charaka": 3900, "Sushruta": 2593}, "herbs_found": 16, "herbs_total": 20,
             "cooccurrence_edges": 120, "triples_rows": 476, "triples_accepted": 300, "triples_rejected": 400, "triples_errors": 2,
             "relations": {"treats": 280, "pacifies": 20}, "grounded_entities_note": "x"}
    retr = pd.DataFrame([{"config": "plain", "recall@5": 0.5, "recall@10": 0.6, "mrr": 0.4, "n": 60},
                         {"config": "alias+graph", "recall@5": 0.55, "recall@10": 0.7, "mrr": 0.45, "n": 60}])
    examples = [{"question": "What treats cough?", "plain": "Plain answer [P1].", "full": "Full answer [P1][G1]."}]
    md = make_phase6_report(stats, retr, ANS, examples)
    for must in ["# Phase 6 results", "## Corpus and graph", "## Retrieval (synthetic questions)", "## KG-grounded multi-hop questions",
                 "## Qualitative examples", "## Limitations", "by construction", "expert", "AyurParam"]:
        assert must in md, must
    assert "6493" in md and "16 of 20" in md and "300" in md
    assert "alias+graph" in md and "0.700" in md
    assert "full" in md and "0.833" in md                                                # (1 + 2/3)/2 for full/top_drugs
    assert "What treats cough?" in md
