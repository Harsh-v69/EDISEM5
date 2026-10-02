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


def test_report_computes_alias_gap_refusal_despite_kg_fact_and_kfirst_comparison():
    stats = {"n_chunks": 10, "per_text": {"Charaka": 6, "Sushruta": 4}, "herbs_found": 16, "herbs_total": 20, "cooccurrence_edges": 5,
             "triples_rows": 4, "triples_accepted": 3, "triples_rejected": 2, "triples_errors": 0, "relations": {"treats": 3}}
    retr = pd.DataFrame([{"config": "plain", "recall@5": 0.45, "recall@10": 0.6, "mrr": 0.3, "n": 60}])
    gap = pd.DataFrame([{"config": "plain", "recall@5": 0.10, "recall@10": 0.20, "mrr": 0.08, "n": 40},
                        {"config": "alias", "recall@5": 0.50, "recall@10": 0.70, "mrr": 0.35, "n": 40}])
    rows = []
    for system, qtype, correct, refused, has_k in [("full", "top_drugs", 1.0, False, True), ("full", "top_drugs", 0.0, True, True),
                                                   ("full", "top_drugs", 0.0, True, True), ("full_kfirst", "top_drugs", 1.0, False, True),
                                                   ("full_kfirst", "top_drugs", 1.0, False, True), ("full_kfirst", "top_drugs", 1.0, False, True),
                                                   ("full", "top_herbs", 0.5, False, True), ("full_kfirst", "top_herbs", 1.0, False, True)]:
        rows.append({"system": system, "qtype": qtype, "correct": correct, "citation_validity": 1.0, "support_rate": 1.0, "refused": refused, "has_k": has_k})
    md = make_phase6_report(stats, retr, pd.DataFrame(rows), [], retr_gap=gap)
    assert "## Vocabulary-gap questions" in md and "0.500" in md and "0.100" in md
    assert "2 of 3" in md and "refused" in md.lower() and "answer-bearing" in md                  # 2 of the 3 'full' top_drugs answers refused despite a K fact
    assert "K-first" in md and "held-out" in md and "0.500" in md and "1.000" in md


def test_report_computes_the_overclaim_rate_from_the_answer_rows():
    stats = {"n_chunks": 10, "per_text": {"Charaka": 6, "Sushruta": 4}, "herbs_found": 16, "herbs_total": 20, "cooccurrence_edges": 5,
             "triples_rows": 4, "triples_accepted": 3, "triples_rejected": 2, "triples_errors": 0, "relations": {"treats": 3}}
    retr = pd.DataFrame([{"config": "plain", "recall@5": 0.45, "recall@10": 0.6, "mrr": 0.3, "n": 60}])
    rows = [{"system": s, "qtype": "top_drugs", "correct": 1.0, "citation_validity": 1.0, "support_rate": 1.0, "refused": False, "has_k": True,
             "n_cocite": c, "n_overclaim": o} for s, c, o in [("full", 4, 1), ("full", 2, 1), ("full_kfirst", 3, 0), ("plain", 5, 5)]]
    md = make_phase6_report(stats, retr, pd.DataFrame(rows), [])
    assert "sentences that cite only co-occurrence" in md and "2 of 6" in md and "33%" in md      # full system only: (1+1) of (4+2)


from ayurveda_kg.phase6 import summarize_overread


def test_summarize_overread_gives_counts_and_rate_per_variant():
    df = pd.DataFrame([{"variant": "before", "qid": "a", "n_cocite": 4, "n_overclaim": 3}, {"variant": "before", "qid": "b", "n_cocite": 2, "n_overclaim": 1},
                       {"variant": "after", "qid": "a", "n_cocite": 3, "n_overclaim": 0}, {"variant": "after", "qid": "b", "n_cocite": 0, "n_overclaim": 0}])
    s = summarize_overread(df).set_index("variant")
    assert s.loc["before", "n_cocite"] == 6 and s.loc["before", "n_overclaim"] == 4 and s.loc["before", "rate"] == pytest.approx(4 / 6)
    assert s.loc["after", "rate"] == 0.0 and s.loc["after", "questions"] == 2


def test_report_has_an_overreading_before_after_section_when_given():
    stats = {"n_chunks": 10, "per_text": {"Charaka": 6, "Sushruta": 4}, "herbs_found": 16, "herbs_total": 20, "cooccurrence_edges": 5,
             "triples_rows": 4, "triples_accepted": 3, "triples_rejected": 2, "triples_errors": 0, "relations": {"treats": 3}}
    retr = pd.DataFrame([{"config": "plain", "recall@5": 0.45, "recall@10": 0.6, "mrr": 0.3, "n": 60}])
    ov = pd.DataFrame([{"variant": "before", "qid": "a", "n_cocite": 6, "n_overclaim": 4}, {"variant": "after", "qid": "a", "n_cocite": 5, "n_overclaim": 1}])
    md = make_phase6_report(stats, retr, ANS, [], overread=ov)
    assert "## Over-reading of co-occurrence facts" in md and "4 of 6" in md and "1 of 5" in md
