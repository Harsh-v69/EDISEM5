"""Leakage-safe splits. Cold-compound folds, cold-herb hold-out (compounds shared with training herbs are dropped, not guessed),
cold-drug folds. Note: cold-compound still lets close chemical analogues straddle folds; a scaffold split is a Phase 4 option."""
import random

import pandas as pd


def assign_folds(ids, k=5, seed=0) -> dict:
    """Deterministic, balanced assignment of ids (compounds or drugs) to k folds."""
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    return {i: n % k for n, i in enumerate(ids)}


def herb_holdout(contains: pd.DataFrame, test_herbs):
    """Returns (train_compounds, test_compounds, shared_compounds).
    test = belongs ONLY to held-out herbs; train = belongs to NO held-out herb; shared = both (dropped, reported)."""
    test_herbs = set(test_herbs)
    herbs_of = contains.groupby("dst")["src"].agg(set)
    train = {c for c, h in herbs_of.items() if not h & test_herbs}
    test = {c for c, h in herbs_of.items() if h <= test_herbs}
    shared = set(herbs_of.index) - train - test
    return train, test, shared


def herb_group_folds(contains: pd.DataFrame, k=5, seed=0):
    """Yield (test_herbs, train, test, shared) so every herb is held out exactly once."""
    folds = assign_folds(contains["src"].unique(), k, seed)
    for f in range(k):
        test_herbs = {h for h, x in folds.items() if x == f}
        if test_herbs:
            yield (test_herbs, *herb_holdout(contains, test_herbs))


def assert_disjoint(a, b):
    both = set(a) & set(b)
    assert not both, f"{len(both)} ids overlap between splits, e.g. {sorted(both)[:3]}"
    return True


def split_pairs(labels: pd.DataFrame, train_compounds, test_compounds, train_drugs=None, test_drugs=None):
    """Select labelled (0/1) pairs for train and test. Unlabelled (-1) pairs are never used."""
    lab = labels[labels["label"].isin([0, 1])]
    pick = lambda comps, drugs: lab[lab["compound"].isin(comps) & (lab["drug"].isin(drugs) if drugs is not None else True)]
    return pick(train_compounds, train_drugs), pick(test_compounds, test_drugs)
