import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.risk import herb_drug_risk, oof_risk


def toy_labels(n_c=30):
    rows = [{"compound": f"c{i}", "drug": d, "label": int((i + j) % 3 == 0) if i % 5 else -1, "enzymes": ""}
            for i in range(n_c) for j, d in enumerate(["d0", "d1", "d2"])]
    return pd.DataFrame(rows)


def test_oof_risk_scores_every_pair_once_from_a_model_that_never_saw_the_compound():
    lab = toy_labels()
    seen = []

    def spy(train, test, fold):
        seen.append((set(train.compound), set(test.compound)))
        assert (train.label.isin([0, 1])).all()                       # only labelled pairs are used for training
        return np.full(len(test), 0.3)

    r = oof_risk(lab, spy, k=4, seed=0)
    assert len(r) == 30 * 3 and not r.duplicated(["compound", "drug"]).any() and set(r.columns) == {"compound", "drug", "p"}
    for tr, te in seen:
        assert not tr & te                                             # cold-compound: test compounds never in training
    assert set().union(*[te for _, te in seen]) == {f"c{i}" for i in range(30)}
    assert (r.p == 0.3).all()
    # unlabelled pairs (label -1, every 5th compound) are still scored: the optimiser needs a risk for every compound
    assert r[r.compound == "c0"].shape[0] == 3


def test_oof_risk_passes_all_drugs_of_test_compounds_aligned():
    lab = toy_labels()
    def echo(train, test, fold):
        return test["drug"].map({"d0": 0.1, "d1": 0.5, "d2": 0.9}).to_numpy()
    r = oof_risk(lab, echo, k=3, seed=1).set_index(["compound", "drug"])["p"]
    assert r[("c7", "d0")] == 0.1 and r[("c7", "d2")] == 0.9


def test_herb_drug_risk_is_mean_over_compounds_with_counts_and_missing_herbs_absent():
    risk = pd.DataFrame({"compound": ["a", "b", "c", "a", "b", "c"], "drug": ["x"] * 3 + ["y"] * 3, "p": [0.2, 0.4, 0.9, 0.0, 0.2, 0.4]})
    contains = pd.DataFrame({"src": ["herb:H1", "herb:H1", "herb:H2"], "dst": ["a", "b", "c"]})
    h = herb_drug_risk(risk, contains).set_index(["herb", "drug"])
    assert h.loc[("H1", "x"), "risk"] == pytest.approx(0.3) and h.loc[("H1", "x"), "n_compounds"] == 2
    assert h.loc[("H2", "y"), "risk"] == pytest.approx(0.4)
    assert ("H3", "x") not in h.index
