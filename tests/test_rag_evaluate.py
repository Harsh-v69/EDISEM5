import csv

import pandas as pd
import pytest

from ayurveda_kg.rag.evaluate import (kg_questions, parse_question, retrieval_eval, score_pair_score, score_top_drugs, too_verbatim,
                                      write_expert_template)

RISK = pd.DataFrame([{"herb": "Curcuma longa", "drug": d, "risk_rf": r, "risk_silver": 0.1, "n_compounds": 50}
                     for d, r in [("warfarin", 0.30), ("phenytoin", 0.45), ("aspirin", 0.10), ("digoxin", 0.20), ("simvastatin", 0.40)]]
                    + [{"herb": "Piper nigrum", "drug": d, "risk_rf": r, "risk_silver": 0.2, "n_compounds": 60}
                       for d, r in [("warfarin", 0.35), ("phenytoin", 0.55), ("aspirin", 0.12), ("digoxin", 0.22), ("simvastatin", 0.42)]])
SCOPE = {"herbs": [{"imppat_name": "Curcuma longa", "common": "turmeric", "aliases": []}, {"imppat_name": "Piper nigrum", "common": "black pepper", "aliases": []}],
         "drugs": []}


def test_kg_questions_have_programmatic_gold_and_use_common_names():
    qs = kg_questions(RISK, SCOPE, pairs=[("Piper nigrum", "phenytoin")])
    top = [q for q in qs if q["type"] == "top_drugs"]
    pair = [q for q in qs if q["type"] == "pair_score"]
    assert len(top) == 2 and len(pair) == 1
    t = next(q for q in top if q["herb"] == "Curcuma longa")
    assert "turmeric" in t["question"].lower() and t["gold"] == ["phenytoin", "simvastatin", "warfarin"]      # top 3 by risk
    assert pair[0]["gold"] == pytest.approx(0.55) and "black pepper" in pair[0]["question"].lower() and "phenytoin" in pair[0]["question"]
    assert len({q["id"] for q in qs}) == len(qs)


def test_score_top_drugs_is_recall_of_the_gold_drugs_named_in_the_answer():
    assert score_top_drugs("Phenytoin, simvastatin and warfarin are highest [K1].", ["phenytoin", "simvastatin", "warfarin"]) == 1.0
    assert score_top_drugs("Phenytoin and aspirin.", ["phenytoin", "simvastatin", "warfarin"]) == pytest.approx(1 / 3)
    assert score_top_drugs("", ["phenytoin"]) == 0.0


def test_score_pair_score_finds_a_number_within_tolerance_and_ignores_citation_ids():
    assert score_pair_score("The research risk score is 0.55 [K1].", 0.55)
    assert score_pair_score("about 0.54", 0.55, tol=0.03) and not score_pair_score("about 0.40", 0.55, tol=0.03)
    assert not score_pair_score("See source [K1]; no number given.", 0.55)          # '1' from [K1] must not count as a score


def test_too_verbatim_flags_questions_that_copy_long_runs_of_the_passage():
    passage = "Haridra paste is applied on the part and cures skin disease and wounds when used regularly."
    assert too_verbatim("Haridra paste is applied on the part and cures skin disease", passage)
    assert not too_verbatim("Which remedy is used for skin ailments?", passage)


def test_parse_question_accepts_json_and_rejects_short_or_malformed_output():
    assert parse_question('{"question": "Which remedy is used for skin ailments in the text?"}') == "Which remedy is used for skin ailments in the text?"
    assert parse_question('{"question": "Too short?"}') is None and parse_question("nope") is None and parse_question('{"q": 1}') is None


def test_retrieval_eval_scores_each_config_with_recall_and_mrr():
    questions = [{"id": "q1", "question": "a", "gold": ["c1"]}, {"id": "q2", "question": "b", "gold": ["c2"]}]
    rankers = {"good": lambda q: ["c1", "x"] if q == "a" else ["c2", "y"], "bad": lambda q: ["x", "y"]}
    r = retrieval_eval(questions, rankers, ks=(1, 5)).set_index("config")
    assert r.loc["good", "recall@1"] == 1.0 and r.loc["good", "mrr"] == 1.0
    assert r.loc["bad", "recall@5"] == 0.0 and r.loc["bad", "mrr"] == 0.0


def test_expert_template_has_columns_for_the_advisor_and_a_draft_banner(tmp_path):
    p = tmp_path / "expert.csv"
    write_expert_template(p, [{"id": "E1", "question": "What does the text say about X?"}])
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert rows[0]["question_id"] == "E1" and rows[0]["expected_answer"] == "" and rows[0]["rater"] == ""
    assert "DRAFT" in rows[0]["status"] and {"gold_source_ids", "notes"} <= set(rows[0])
