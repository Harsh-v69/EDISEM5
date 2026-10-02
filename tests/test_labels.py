import pandas as pd

from ayurveda_kg.labels import CYP5, silver_labels


def kg(inh, sub, non, has_adme=("c1", "c2", "c3", "c4")):
    ids = ["c1", "c2", "c3", "c4"]
    nodes = {"Compound": pd.DataFrame({"id": [f"cpd:{i}" for i in ids], "has_adme": [i in has_adme for i in ids]}),
             "Drug": pd.DataFrame({"id": ["drug:a", "drug:b"], "name": ["a", "b"]})}
    mk = lambda pairs: pd.DataFrame(pairs, columns=["src", "dst"])
    edges = {"predicted_cyp_inhibitor": mk([(f"cpd:{c}", f"gene:{g}") for c, g in inh]),
             "substrate_of": mk([(f"drug:{d}", f"gene:{g}") for d, g in sub]),
             "non_substrate_of": mk([(f"drug:{d}", f"gene:{g}") for d, g in non])}
    return nodes, edges


def lab(df, c, d):
    r = df[(df.compound == f"cpd:{c}") & (df.drug == f"drug:{d}")]
    assert len(r) == 1
    return int(r.label.iloc[0]), r.enzymes.iloc[0]


def test_positive_when_inhibitor_meets_substrate_of_same_enzyme():
    df = silver_labels(*kg(inh=[("c1", "CYP3A4"), ("c1", "CYP2D6")], sub=[("a", "CYP3A4")], non=[]))
    assert lab(df, "c1", "a") == (1, "CYP3A4")           # only the matching enzyme is the mechanism


def test_negative_when_compound_inhibits_nothing():
    df = silver_labels(*kg(inh=[], sub=[("a", "CYP3A4")], non=[]))
    assert lab(df, "c2", "a")[0] == 0 and lab(df, "c2", "b")[0] == 0


def test_negative_when_drug_is_verified_non_substrate_of_every_inhibited_enzyme():
    df = silver_labels(*kg(inh=[("c1", "CYP2C9")], sub=[], non=[("b", "CYP2C9")]))
    assert lab(df, "c1", "b")[0] == 0


def test_unlabelled_when_substrate_status_unknown_for_an_inhibited_enzyme():
    df = silver_labels(*kg(inh=[("c1", "CYP2C19")], sub=[], non=[]))
    assert lab(df, "c1", "a") == (-1, "CYP2C19")


def test_substrate_wins_over_non_substrate_conflict_and_is_reported():
    df = silver_labels(*kg(inh=[("c1", "CYP3A4")], sub=[("a", "CYP3A4")], non=[("a", "CYP3A4")]))
    assert lab(df, "c1", "a")[0] == 1
    assert df.attrs["conflicts"] == [("drug:a", "CYP3A4")]


def test_compound_without_predictions_is_never_labelled():
    df = silver_labels(*kg(inh=[], sub=[], non=[], has_adme=("c1",)))
    assert (df[df.compound == "cpd:c3"].label == -1).all()


def test_only_the_five_cyps_count_and_all_pairs_present():
    df = silver_labels(*kg(inh=[("c1", "CYP2E1")], sub=[("a", "CYP2E1")], non=[]))   # CYP2E1 is outside CYP5
    assert lab(df, "c1", "a")[0] == 0
    assert len(df) == 4 * 2 and set(df.label) <= {1, 0, -1}
    assert CYP5 == ("CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4")


# ---- herb-level aggregation ----
from ayurveda_kg.labels import herb_drug_scores


def test_herb_drug_scores_counts_positive_and_labelled_compounds_only():
    labels = pd.DataFrame({"compound": ["c1", "c2", "c3", "c1", "c2", "c3"], "drug": ["a"] * 3 + ["b"] * 3,
                           "label": [1, 0, -1, 0, 0, 1], "enzymes": [""] * 6})
    contains = pd.DataFrame({"src": ["herb:H", "herb:H", "herb:H", "herb:K"], "dst": ["c1", "c2", "c3", "c3"]})
    s = herb_drug_scores(labels, contains).set_index(["herb", "drug"])
    assert s.loc[("herb:H", "a"), ["n_pos", "n_labelled"]].tolist() == [1, 2]      # c3 is unlabelled for drug a
    assert s.loc[("herb:H", "a"), "frac_pos"] == 0.5
    assert s.loc[("herb:H", "b"), ["n_pos", "n_labelled"]].tolist() == [1, 3]
    assert s.loc[("herb:K", "a"), "n_labelled"] == 0 and pd.isna(s.loc[("herb:K", "a"), "frac_pos"])   # no evidence: NaN, not 0


# ---- gold vs silver comparison ----
from ayurveda_kg.labels import gold_vs_silver


def test_gold_vs_silver_adds_score_and_percentile_among_all_scored_pairs():
    scores = pd.DataFrame({"herb": ["herb:A", "herb:A", "herb:B", "herb:B"], "drug": ["drug:x", "drug:y", "drug:x", "drug:y"],
                           "n_pos": [0, 1, 2, 4], "n_labelled": [4, 4, 4, 4], "n_compounds": [4] * 4,
                           "frac_pos": [0.0, 0.25, 0.5, 1.0]})
    gold = pd.DataFrame({"herb": ["A", "B", "A"], "drug": ["y", "x", "zz"], "label": [1, 0, 1], "mechanism": ["PK", "none", "PK"]})
    out = gold_vs_silver(gold, scores).set_index(["herb", "drug"])
    assert out.loc[("A", "y"), "frac_pos"] == 0.25 and out.loc[("A", "y"), "percentile"] == 0.5
    assert out.loc[("B", "x"), "percentile"] == 0.75
    assert pd.isna(out.loc[("A", "zz"), "frac_pos"])          # gold pair absent from the scores stays visible, as NaN
