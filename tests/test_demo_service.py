import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.demo.service import (CAVEAT, DemoData, ask, explain_pair, formulation_choices, list_drugs, list_herbs, risk_ranking, suggest)


def data():
    nodes = {"Herb": pd.DataFrame({"id": ["herb:Piper nigrum", "herb:Curcuma longa"], "name": ["Piper nigrum", "Curcuma longa"]}),
             "Compound": pd.DataFrame({"id": ["cpd:c1", "cpd:c2", "cpd:c3"], "name": ["piperine", "plain", "curcumin"]}),
             "Drug": pd.DataFrame({"id": ["drug:phenytoin", "drug:warfarin"], "name": ["phenytoin", "warfarin"]})}
    mk = lambda rows, cols=("src", "dst"): pd.DataFrame(rows, columns=list(cols))
    edges = {"contains": mk([("herb:Piper nigrum", "cpd:c1"), ("herb:Piper nigrum", "cpd:c2"), ("herb:Curcuma longa", "cpd:c3")]),
             "predicted_cyp_inhibitor": mk([("cpd:c1", "gene:CYP3A4", "SwissADME (via IMPPAT)")], cols=("src", "dst", "source")),
             "substrate_of": mk([("drug:phenytoin", "gene:CYP3A4", "ChEMBL metabolism; TDC")], cols=("src", "dst", "source"))}
    labels = pd.DataFrame([{"compound": "cpd:c1", "drug": "drug:phenytoin", "label": 1, "enzymes": "CYP3A4"},
                           {"compound": "cpd:c2", "drug": "drug:phenytoin", "label": 0, "enzymes": ""},
                           {"compound": "cpd:c3", "drug": "drug:phenytoin", "label": -1, "enzymes": "CYP2D6"}])
    cd = pd.DataFrame([{"compound": c, "drug": d, "p": p} for c, d, p in
                       [("cpd:c1", "drug:phenytoin", 0.9), ("cpd:c2", "drug:phenytoin", 0.1), ("cpd:c3", "drug:phenytoin", 0.3),
                        ("cpd:c1", "drug:warfarin", 0.4), ("cpd:c2", "drug:warfarin", 0.2), ("cpd:c3", "drug:warfarin", 0.2)]])
    hd = pd.DataFrame([{"herb": h, "drug": d, "risk_rf": r, "risk_silver": s, "n_compounds": n}
                       for h, d, r, s, n in [("Piper nigrum", "phenytoin", 0.50, 0.50, 2), ("Curcuma longa", "phenytoin", 0.30, 0.00, 1),
                                             ("Piper nigrum", "warfarin", 0.30, 0.20, 2), ("Curcuma longa", "warfarin", 0.20, 0.00, 1)]])
    gold = pd.DataFrame([{"herb": "Piper nigrum", "drug": "phenytoin", "label": 1, "mechanism": "PK", "evidence": "human_patient_pk", "confidence": "high",
                          "pmid": "16767797", "citation": "Pattanaik 2006", "finding": "piperine raised phenytoin AUC", "verification": "abstract read"}])
    scope = {"herbs": [{"imppat_name": "Piper nigrum", "common": "black pepper", "aliases": []}, {"imppat_name": "Curcuma longa", "common": "turmeric", "aliases": []}],
             "drugs": [{"name": "phenytoin", "cls": "x"}, {"name": "warfarin", "cls": "y"}]}
    uses = {"Piper nigrum": {"cough", "fever"}, "Curcuma longa": {"fever", "wounds"}}
    forms = pd.DataFrame([{"id": "F1", "name": "Test churna", "kind": "afi", "n_ingredients": 4, "in_scope": ["Piper nigrum", "Curcuma longa"]},
                          {"id": "F2", "name": "Single pepper", "kind": "api", "n_ingredients": 1, "in_scope": ["Piper nigrum"]},
                          {"id": "F3", "name": "No scoped herb", "kind": "afi", "n_ingredients": 3, "in_scope": []}])
    return DemoData(nodes, edges, labels, cd, hd, gold, scope, uses, forms)


def test_lists_use_common_names_and_sorted_scope():
    d = data()
    assert list_herbs(d) == [("Curcuma longa", "turmeric"), ("Piper nigrum", "black pepper")]
    assert list_drugs(d) == ["phenytoin", "warfarin"]


def test_explain_pair_gives_scores_ranked_compounds_enzyme_path_with_sources_and_the_gold_study():
    r = explain_pair(data(), "Piper nigrum", "phenytoin", top_n=2)
    assert r["scores"]["risk_rf"] == 0.50 and r["scores"]["n_compounds"] == 2
    assert r["scores"]["percentile_for_drug"] == 1.0                                   # highest of the 2 herbs for phenytoin
    top = r["compounds"]
    assert [c["name"] for c in top] == ["piperine", "plain"] and top[0]["p"] == 0.9 and top[0]["silver_label"] == 1
    path = top[0]["paths"][0]
    assert "piperine" in path and "CYP3A4" in path and "SwissADME" in path and "ChEMBL" in path and "phenytoin" in path
    assert top[1]["paths"] == [] and top[1]["silver_label"] == 0
    g = r["gold"][0]
    assert g["pmid"] == "16767797" and g["label"] == 1 and "AUC" in g["finding"]
    assert "not medical advice" in r["caveat"].lower() and r["caveat"] == CAVEAT


def test_explain_pair_without_published_evidence_says_so_and_unknown_inputs_return_an_error_not_a_crash():
    r = explain_pair(data(), "Curcuma longa", "phenytoin")
    assert r["gold"] == [] and r["compounds"][0]["silver_label"] == -1 and r["compounds"][0]["paths"] == []
    assert "error" in explain_pair(data(), "Ghost herb", "phenytoin")
    assert "error" in explain_pair(data(), "Piper nigrum", "ghostdrug")


def test_risk_ranking_orders_herbs_for_a_drug():
    r = risk_ranking(data(), "phenytoin")
    assert r["herb"].tolist() == ["Piper nigrum", "Curcuma longa"] and r["risk_rf"].tolist() == [0.50, 0.30]


def test_formulation_choices_lists_only_formulations_with_a_scoped_herb_and_flags_optimisable():
    c = formulation_choices(data()).set_index("id")
    assert set(c.index) == {"F1", "F2"} and bool(c.loc["F1", "optimisable"]) and not bool(c.loc["F2", "optimisable"])


def test_suggest_returns_shares_summing_to_baseline_and_per_drug_risk_before_after_for_several_drugs():
    r = suggest(data(), "F1", ["phenytoin", "warfarin"], tau=0.3, lo_frac=0.5, hi_mult=2.0)
    t = r["table"].set_index("herb")
    assert r["optimisable"] and t["baseline_share"].sum() == pytest.approx(0.5) and t["suggested_share"].sum() == pytest.approx(0.5)
    assert t.loc["Piper nigrum", "suggested_share"] < t.loc["Piper nigrum", "baseline_share"]       # riskier herb is reduced
    assert r["risk_after"]["phenytoin"] <= r["risk_before"]["phenytoin"] + 1e-9
    assert r["fixed_ingredients"] == 2 and "not a dosing recommendation" in r["caveat"].lower()
    assert r["min_coverage_ratio"] >= 0.3 - 1e-9


def test_suggest_handles_single_herb_unknown_formulation_and_unknown_drug_gracefully():
    r = suggest(data(), "F2", ["phenytoin"])
    assert r["optimisable"] is False and "only one" in r["message"].lower()
    assert "error" in suggest(data(), "NOPE", ["phenytoin"])
    assert "error" in suggest(data(), "F1", ["ghostdrug"])
    assert "error" in suggest(data(), "F1", [])


def test_ask_returns_items_and_degrades_to_retrieval_only_when_no_model_is_available():
    class R:
        def retrieve(self, q, **kw):
            return [{"id": "P1", "kind": "P", "chunk": "c1", "text": "Haridra cures skin disease.", "source": "Charaka"}]
    out = ask(R(), "What treats skin disease?", client=None)
    assert out["items"][0]["id"] == "P1" and out["answer"] is None and "retrieval only" in out["note"].lower()
    out = ask(R(), "What treats skin disease?", client=lambda p: "Haridra cures skin disease [P1].")
    assert out["answer"]["checks"]["valid"] == ["P1"] and out["answer"]["text"].strip().endswith("clinician.")

    def boom(p):
        raise RuntimeError("down")
    out = ask(R(), "q", client=boom)
    assert out["items"] and out["answer"]["error"]


def test_explain_pair_also_returns_structured_path_steps_for_drawing_the_chain():
    top = explain_pair(data(), "Piper nigrum", "phenytoin", top_n=2)["compounds"]
    step = top[0]["path_steps"][0]
    assert step == {"herb": "Piper nigrum", "compound": "piperine", "enzyme": "CYP3A4", "drug": "phenytoin",
                    "inhibitor_source": "SwissADME (via IMPPAT)", "substrate_source": "ChEMBL metabolism; TDC"}
    assert top[1]["path_steps"] == []
