import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.evaluate import make_folds, metrics, precision_at_k, run_cv


def toy_labels(n_c=40, n_d=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"compound": f"c{i}", "drug": f"d{j}", "label": int(rng.random() < 0.3), "enzymes": ""} for i in range(n_c) for j in range(n_d)]
    return pd.DataFrame(rows)


def toy_contains(n_c=40):
    return pd.DataFrame({"src": [f"h{i % 8}" for i in range(n_c)], "dst": [f"c{i}" for i in range(n_c)]})   # each compound in one herb


def test_metrics_perfect_inverted_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["auroc"] == 0.0
    m = metrics(np.array([1, 1, 1]), np.array([0.2, 0.3, 0.4]))
    assert np.isnan(m["auroc"]) and np.isnan(m["auprc"]) and m["n_pos"] == 3        # undefined, not silently 0.5


def test_precision_at_k():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_k(y, p, 2) == 0.5 and precision_at_k(y, p, 3) == pytest.approx(2 / 3)
    assert precision_at_k(y, p, 100) == pytest.approx(2 / 5)                        # k larger than n: use all


@pytest.mark.parametrize("split", ["cold_compound", "cold_herb", "cold_drug", "random_pair"])
def test_folds_are_leak_free_for_each_split_type(split):
    lab, con = toy_labels(), toy_contains()
    folds = make_folds(split, lab, con, k=4, seed=0)
    assert len(folds) == 4
    for f in folds:
        tr, te = f["train_pairs"], f["test_pairs"]
        assert len(tr) > 0 and len(te) > 0 and set(tr.label) <= {0, 1}
        assert not set(map(tuple, tr[["compound", "drug"]].values)) & set(map(tuple, te[["compound", "drug"]].values))
        if split in ("cold_compound", "cold_herb"):
            assert not set(tr.compound) & set(te.compound)
        if split == "cold_drug":
            assert not set(tr.drug) & set(te.drug)
        assert set(tr.compound) <= f["train_compounds"] and set(tr.drug) <= f["train_drugs"]


def test_cold_herb_drops_compounds_shared_with_training_herbs():
    lab = toy_labels(n_c=6)
    con = pd.DataFrame({"src": ["hA", "hA", "hB", "hB", "hC", "hC"], "dst": ["c0", "c1", "c1", "c2", "c3", "c4"]})   # c1 shared; c5 in no herb
    for f in make_folds("cold_herb", lab, con, k=3, seed=0):
        assert not set(f["train_pairs"].compound) & set(f["test_pairs"].compound)
        assert "c5" not in set(f["test_pairs"].compound)


def test_unlabelled_pairs_never_reach_train_or_test():
    lab = toy_labels(); lab.loc[::7, "label"] = -1
    for f in make_folds("cold_compound", lab, toy_contains(), k=4, seed=0):
        assert (f["train_pairs"].label >= 0).all() and (f["test_pairs"].label >= 0).all()


def test_run_cv_returns_one_row_per_fold_and_passes_aligned_predictions():
    lab, con = toy_labels(), toy_contains()
    def prior(train, test, fold):
        return np.full(len(test), train.label.mean())
    r = run_cv("cold_compound", lab, con, prior, k=4, seed=0)
    assert len(r) == 4 and {"split", "fold", "auroc", "auprc", "p_at_100", "n_train", "n_test", "n_pos"} <= set(r.columns)
    def bad(train, test, fold):
        return np.zeros(len(test) + 1)
    with pytest.raises(AssertionError, match="aligned"):
        run_cv("cold_compound", lab, con, bad, k=4, seed=0)


# ---- grouped AUROC (separates compound-side from drug-side skill) ----
from ayurveda_kg.evaluate import grouped_auroc


def test_grouped_auroc_averages_per_group_and_skips_single_class_groups():
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4 + ["c"] * 3,
                       "label": [0, 0, 1, 1] + [0, 0, 1, 1] + [1, 1, 1],
                       "p": [0.1, 0.2, 0.8, 0.9] + [0.9, 0.8, 0.1, 0.2] + [0.5, 0.6, 0.7]})
    m, n = grouped_auroc(df, "g")
    assert n == 2 and m == pytest.approx((1.0 + 0.0) / 2)             # group c (all positive) is undefined and skipped, not counted as 0.5


def test_grouped_auroc_reveals_a_model_that_only_knows_the_group_base_rate():
    rng = np.random.default_rng(0)
    rows = [{"drug": d, "label": int(rng.random() < rate), "p": rate} for d, rate in [("hi", 0.8), ("lo", 0.1)] for _ in range(300)]
    df = pd.DataFrame(rows)
    pooled = metrics(df.label.values, df.p.values)["auroc"]
    within, _ = grouped_auroc(df, "drug")
    assert pooled > 0.7 and within == pytest.approx(0.5)               # pooled looks good; within-drug shows it learned nothing per drug
