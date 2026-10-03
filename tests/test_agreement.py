import math

import pandas as pd
import pytest

from ayurveda_kg.vision_nadi.agreement import cohens_kappa, pairwise_kappa, validate_annotations


def test_kappa_is_one_for_perfect_agreement_zero_for_chance_and_negative_for_systematic_disagreement():
    a = ["x", "y", "x", "y", "x", "y"]
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert cohens_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)           # same marginals, agreement exactly at chance level
    assert cohens_kappa(["x", "y", "x", "y"], ["y", "x", "y", "x"]) == pytest.approx(-1.0)


def test_kappa_matches_a_hand_computed_example():
    # 10 items, 2 raters, 2 categories: observed agreement 0.8, expected 0.5 -> kappa 0.6
    r1 = ["a"] * 5 + ["b"] * 5
    r2 = ["a"] * 4 + ["b"] + ["b"] * 4 + ["a"]
    assert cohens_kappa(r1, r2) == pytest.approx(0.6)


def test_kappa_is_undefined_not_a_fake_number_when_there_is_no_variation_or_no_items():
    assert math.isnan(cohens_kappa(["a", "a", "a"], ["a", "a", "a"]))          # both raters always say 'a': chance agreement is 1, kappa undefined
    assert math.isnan(cohens_kappa([], []))
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa(["a"], ["a", "b"])


def test_pairwise_kappa_covers_every_rater_pair_and_ignores_items_either_rater_skipped():
    df = pd.DataFrame({"image": [1, 2, 3, 4, 5, 6], "rater": ["A"] * 3 + ["B"] * 3 + ["A"] * 0,
                       "label": ["x", "y", "x", "x", "y", "x"]})
    long = pd.DataFrame([{"item": i, "rater": r, "label": l} for i, r, l in
                         [(1, "A", "x"), (2, "A", "y"), (3, "A", "x"), (4, "A", "y"),
                          (1, "B", "x"), (2, "B", "y"), (3, "B", "y"),                      # B skipped item 4
                          (1, "C", "x"), (2, "C", "x"), (3, "C", "x"), (4, "C", "y")]])
    k = pairwise_kappa(long).set_index(["rater_1", "rater_2"])
    assert set(k.index) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert k.loc[("A", "B"), "n_items"] == 3 and k.loc[("A", "C"), "n_items"] == 4


def test_validate_annotations_flags_duplicates_unknown_labels_and_single_rater_items():
    long = pd.DataFrame([{"item": 1, "rater": "A", "label": "pale"}, {"item": 1, "rater": "B", "label": "red"},
                         {"item": 2, "rater": "A", "label": "pale"}, {"item": 2, "rater": "A", "label": "red"},       # duplicate A on item 2
                         {"item": 3, "rater": "A", "label": "purple?"}])                                               # unknown label, single rater
    probs = "\n".join(validate_annotations(long, allowed=["pale", "red", "dark"]))
    assert "duplicate" in probs and "unknown label" in probs and "purple?" in probs and "single rater" in probs
    ok = pd.DataFrame([{"item": 1, "rater": "A", "label": "pale"}, {"item": 1, "rater": "B", "label": "red"}])
    assert validate_annotations(ok, allowed=["pale", "red"]) == []
