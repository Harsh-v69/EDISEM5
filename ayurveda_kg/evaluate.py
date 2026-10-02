"""Evaluation harness: builds leak-free folds, enforces disjointness, and scores any model passed in as a function."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ayurveda_kg.splits import assert_disjoint, assign_folds, herb_group_folds, split_pairs

SPLITS = ("cold_compound", "cold_herb", "cold_drug", "random_pair")


def precision_at_k(y, p, k=100) -> float:
    k = min(k, len(y))
    return float(np.asarray(y)[np.argsort(-np.asarray(p), kind="stable")[:k]].mean())


def metrics(y, p, k=100) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    both = len(set(y.tolist())) == 2
    return {"auroc": roc_auc_score(y, p) if both else np.nan, "auprc": average_precision_score(y, p) if both else np.nan,
            f"p_at_{k}": precision_at_k(y, p, k), "n_test": len(y), "n_pos": int((y == 1).sum())}


def grouped_auroc(df: pd.DataFrame, by: str, y="label", p="p"):
    """Mean AUROC over groups (e.g. drugs or compounds) that contain both classes. Returns (mean, n_groups).
    Pooled AUROC can look strong when a model only knows each group's base rate; this removes that effect."""
    vals = [roc_auc_score(g[y], g[p]) for _, g in df.groupby(by) if g[y].nunique() == 2]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def make_folds(split, labels, contains, k=5, seed=0) -> list[dict]:
    """Each fold: train_pairs, test_pairs (labelled 0/1 only) and the node sets the training graph may contain."""
    lab = labels[labels["label"].isin([0, 1])]
    comps, drugs = set(labels["compound"]), set(labels["drug"])
    folds = []
    if split == "cold_compound":
        a = assign_folds(comps, k, seed)
        for f in range(k):
            test_c = {c for c, x in a.items() if x == f}
            tr, te = split_pairs(lab, comps - test_c, test_c)
            folds.append({"train_pairs": tr, "test_pairs": te, "train_compounds": comps - test_c, "train_drugs": drugs})
    elif split == "cold_herb":
        for test_herbs, train_c, test_c, shared in herb_group_folds(contains, k, seed):
            assert_disjoint(train_c, test_c)
            tr, te = split_pairs(lab, train_c, test_c)
            folds.append({"train_pairs": tr, "test_pairs": te, "train_compounds": train_c, "train_drugs": drugs, "test_herbs": test_herbs})
    elif split == "cold_drug":
        a = assign_folds(drugs, k, seed)
        for f in range(k):
            test_d = {d for d, x in a.items() if x == f}
            tr, te = split_pairs(lab, comps, comps, train_drugs=drugs - test_d, test_drugs=test_d)
            folds.append({"train_pairs": tr, "test_pairs": te, "train_compounds": comps, "train_drugs": drugs - test_d})
    elif split == "random_pair":     # inflated reference only: compounds and drugs are seen in training
        idx = np.random.default_rng(seed).permutation(len(lab))
        for f in range(k):
            te_i = idx[f::k]
            mask = np.zeros(len(lab), bool); mask[te_i] = True
            folds.append({"train_pairs": lab[~mask], "test_pairs": lab[mask], "train_compounds": comps, "train_drugs": drugs})
    else:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    return folds


def run_cv(split, labels, contains, fit_predict, k=5, seed=0, k_top=100) -> pd.DataFrame:
    """fit_predict(train_pairs, test_pairs, fold) -> probabilities aligned with test_pairs."""
    rows = []
    for i, f in enumerate(make_folds(split, labels, contains, k, seed)):
        p = np.asarray(fit_predict(f["train_pairs"], f["test_pairs"], f))
        assert len(p) == len(f["test_pairs"]), "predictions must be aligned with test_pairs"
        rows.append({"split": split, "fold": i, "n_train": len(f["train_pairs"]),
                     **metrics(f["test_pairs"]["label"].values, p, k_top)})
    return pd.DataFrame(rows)
