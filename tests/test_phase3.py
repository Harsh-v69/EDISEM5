import pandas as pd

from ayurveda_kg.phase3 import make_labels_report


def test_labels_report_has_counts_per_drug_split_table_and_gold_table_with_caveat():
    labels = pd.DataFrame({"compound": ["c1", "c1", "c2", "c2"], "drug": ["drug:a", "drug:b", "drug:a", "drug:b"],
                           "label": [1, 0, -1, 0], "enzymes": ["CYP3A4", "", "CYP2D6", ""]})
    drug_names = {"drug:a": "alpha", "drug:b": "beta"}
    splits = [{"fold": 0, "test_herbs": 2, "train_compounds": 10, "test_compounds": 4, "shared_dropped": 3,
               "train_pairs": 30, "test_pairs": 12, "test_pos": 5}]
    gold = pd.DataFrame([{"herb": "Zingiber officinale", "drug": "warfarin", "label": 0, "mechanism": "none", "evidence": "human_crossover_pk",
                          "n_pos": 184, "n_labelled": 334, "frac_pos": 0.55, "percentile": 0.97}])
    md = make_labels_report(labels, drug_names, splits, gold)
    assert "| positive (1) | 1 |" in md and "| negative (0) | 2 |" in md and "| unlabelled (-1) | 1 |" in md
    assert "| alpha | 1 | 0 | 1 |" in md                      # per-drug: pos, neg, unlabelled
    assert "shared_dropped" in md and "| 0 | 2 | 10 | 4 | 3 |" in md
    assert "Zingiber officinale" in md and "0.97" in md
    assert "mechanistic hypothesis" in md                      # the honest caveat must always be in the report
