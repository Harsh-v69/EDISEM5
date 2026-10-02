"""Compound-drug and herb-drug risk scores for the composition optimiser. Out-of-fold only: each compound is scored by a model
that never saw it, so the optimiser is never steered by memorised scores."""
import pandas as pd

from ayurveda_kg.splits import assign_folds


def oof_risk(labels: pd.DataFrame, fit_predict, k=5, seed=0) -> pd.DataFrame:
    """Cold-compound out-of-fold probability for EVERY (compound, drug) pair, including pairs with no silver label.
    fit_predict(train_pairs, test_pairs, fold) -> probabilities aligned with test_pairs (see evaluate.run_cv)."""
    comps = sorted(set(labels["compound"]))
    folds = assign_folds(comps, k, seed)
    labelled = labels[labels["label"].isin([0, 1])]
    out = []
    for f in range(k):
        test_c = {c for c, x in folds.items() if x == f}
        test_pairs = labels[labels["compound"].isin(test_c)][["compound", "drug"]].reset_index(drop=True)
        train = labelled[~labelled["compound"].isin(test_c)]
        p = fit_predict(train, test_pairs, {"train_compounds": set(comps) - test_c, "train_drugs": set(labels["drug"])})
        out.append(test_pairs.assign(p=p))
    return pd.concat(out, ignore_index=True)


def herb_drug_risk(risk: pd.DataFrame, contains: pd.DataFrame) -> pd.DataFrame:
    """Herb-drug risk = mean compound risk. IMPPAT has no compound abundance, so every compound counts equally (a stated assumption)."""
    m = contains.rename(columns={"src": "herb", "dst": "compound"})[["herb", "compound"]].merge(risk, on="compound")
    g = m.groupby(["herb", "drug"], as_index=False).agg(risk=("p", "mean"), n_compounds=("compound", "nunique"))
    g["herb"] = g["herb"].str.removeprefix("herb:")
    g["drug"] = g["drug"].str.removeprefix("drug:")
    return g
