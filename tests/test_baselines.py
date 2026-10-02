import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.baselines import mf_fit_predict, prior_fit_predict, rf_fit_predict
from ayurveda_kg.evaluate import metrics
from ayurveda_kg.features import Features


def synthetic(n_c=120, n_d=6, seed=0):
    """label = compound feature 0 AND drug feature 0 (a tiny stand-in for 'inhibits an enzyme the drug is a substrate of')."""
    rng = np.random.default_rng(seed)
    Xc = (rng.random((n_c, 8)) < 0.5).astype(np.float32)
    Xd = (rng.random((n_d, 4)) < 0.5).astype(np.float32)
    cids, dids = [f"c{i}" for i in range(n_c)], [f"d{j}" for j in range(n_d)]
    fe = Features(cids, dids, Xc, Xd, [f"x{i}" for i in range(8)], [f"y{i}" for i in range(4)], 0)
    rows = [{"compound": cids[i], "drug": dids[j], "label": int(Xc[i, 0] * Xd[j, 0]), "enzymes": ""} for i in range(n_c) for j in range(n_d)]
    return fe, pd.DataFrame(rows)


def split(lab, frac=0.7, seed=1):
    m = np.random.default_rng(seed).random(len(lab)) < frac
    return lab[m].reset_index(drop=True), lab[~m].reset_index(drop=True)


def test_prior_uses_per_drug_rate_and_global_rate_for_unseen_drug():
    tr = pd.DataFrame({"compound": list("abcd"), "drug": ["x", "x", "y", "y"], "label": [1, 1, 0, 1]})
    te = pd.DataFrame({"compound": list("ef"), "drug": ["x", "zz"], "label": [0, 0]})
    p = prior_fit_predict(tr, te, {})
    assert p.tolist() == [1.0, 0.75]                      # drug x rate; unseen drug falls back to the global rate


def test_rf_learns_the_synthetic_rule_on_unseen_pairs():
    fe, lab = synthetic()
    tr, te = split(lab)
    p = rf_fit_predict(fe, n_estimators=60, seed=0)(tr, te, {})
    assert metrics(te.label.values, p)["auroc"] > 0.95


def test_rf_is_deterministic_for_a_seed():
    fe, lab = synthetic()
    tr, te = split(lab)
    a = rf_fit_predict(fe, n_estimators=20, seed=3)(tr, te, {})
    b = rf_fit_predict(fe, n_estimators=20, seed=3)(tr, te, {})
    assert np.allclose(a, b, rtol=0, atol=1e-12)     # parallel trees add floats in varying order: ~1e-16 noise, identical forests


def test_rf_with_leaky_columns_beats_rf_without_when_label_depends_on_hidden_state():
    fe, _ = synthetic()
    rng = np.random.default_rng(5)
    hc, hd = rng.integers(0, 2, len(fe.compound_ids)), rng.integers(0, 2, len(fe.drug_ids))   # hidden CYP-style state, not in fe
    lab = pd.DataFrame([{"compound": c, "drug": d, "label": int(hc[i] * hd[j]), "enzymes": ""}
                        for i, c in enumerate(fe.compound_ids) for j, d in enumerate(fe.drug_ids)])
    tr, te = lab[lab.compound.isin(fe.compound_ids[:80])], lab[lab.compound.isin(fe.compound_ids[80:])]   # cold compounds
    clean = rf_fit_predict(fe, n_estimators=60, seed=0)(tr, te, {})
    leaky = rf_fit_predict(fe, n_estimators=60, seed=0, extra_compound=hc.reshape(-1, 1).astype(np.float32),
                           extra_drug=hd.reshape(-1, 1).astype(np.float32))(tr, te, {})
    assert metrics(te.label.values, leaky)["auroc"] > 0.95
    assert metrics(te.label.values, clean)["auroc"] < 0.75          # without the hidden state the cold compounds are unpredictable


def test_mf_learns_low_rank_structure_and_falls_back_to_prior_for_unseen_ids():
    rng = np.random.default_rng(0)
    n_c, n_d = 80, 12
    U, V = rng.normal(size=(n_c, 2)), rng.normal(size=(n_d, 2))
    y = ((U @ V.T) > 0).astype(int)
    lab = pd.DataFrame([{"compound": f"c{i}", "drug": f"d{j}", "label": int(y[i, j]), "enzymes": ""} for i in range(n_c) for j in range(n_d)])
    tr, te = split(lab, 0.7)
    p = mf_fit_predict(rank=4, epochs=400, seed=0)(tr, te, {})
    assert metrics(te.label.values, p)["auroc"] > 0.85
    cold = pd.DataFrame({"compound": ["unseen"], "drug": ["d0"], "label": [0]})
    assert mf_fit_predict(rank=4, epochs=50, seed=0)(tr, cold, {})[0] == pytest.approx(prior_fit_predict(tr, cold, {})[0])
