import pandas as pd
import pytest

from ayurveda_kg.splits import assert_disjoint, assign_folds, herb_group_folds, herb_holdout, split_pairs


def test_assign_folds_covers_everything_once_balanced_and_deterministic():
    ids = [f"c{i}" for i in range(23)]
    f = assign_folds(ids, k=5, seed=1)
    assert set(f) == set(ids) and set(f.values()) == set(range(5))
    sizes = pd.Series(f).value_counts()
    assert sizes.max() - sizes.min() <= 1
    assert f == assign_folds(ids, k=5, seed=1) and f != assign_folds(ids, k=5, seed=2)


CONTAINS = pd.DataFrame({"src": ["herb:A", "herb:A", "herb:B", "herb:B", "herb:C"],
                         "dst": ["cpd:1", "cpd:2", "cpd:2", "cpd:3", "cpd:4"]})   # cpd:2 is shared by A and B


def test_herb_holdout_keeps_only_exclusive_compounds_and_reports_shared():
    train, test, shared = herb_holdout(CONTAINS, test_herbs={"herb:A"})
    assert test == {"cpd:1"} and train == {"cpd:3", "cpd:4"} and shared == {"cpd:2"}
    assert_disjoint(train, test)


def test_herb_holdout_with_two_test_herbs_makes_shared_compound_exclusive():
    train, test, shared = herb_holdout(CONTAINS, test_herbs={"herb:A", "herb:B"})
    assert test == {"cpd:1", "cpd:2", "cpd:3"} and train == {"cpd:4"} and shared == set()


def test_herb_group_folds_every_herb_is_test_exactly_once_and_never_leaks():
    seen = []
    for test_herbs, train, test, shared in herb_group_folds(CONTAINS, k=3, seed=0):
        seen += sorted(test_herbs)
        assert_disjoint(train, test)
        assert_disjoint(train, shared) and assert_disjoint(test, shared)
    assert sorted(seen) == ["herb:A", "herb:B", "herb:C"]


def test_assert_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError, match="overlap"):
        assert_disjoint({"a", "b"}, {"b"})


def test_split_pairs_drops_unlabelled_and_respects_both_axes():
    lab = pd.DataFrame({"compound": ["c1", "c1", "c2", "c2", "c3"], "drug": ["a", "b", "a", "b", "a"], "label": [1, 0, -1, 0, 1]})
    tr, te = split_pairs(lab, train_compounds={"c1", "c3"}, test_compounds={"c2"})
    assert tr.label.isin([0, 1]).all() and te.label.isin([0, 1]).all()
    assert set(tr.compound) <= {"c1", "c3"} and set(te.compound) == {"c2"} and len(te) == 1
    tr, te = split_pairs(lab, train_compounds={"c1", "c2", "c3"}, test_compounds={"c1", "c2", "c3"}, train_drugs={"a"}, test_drugs={"b"})
    assert set(tr.drug) == {"a"} and set(te.drug) == {"b"}
