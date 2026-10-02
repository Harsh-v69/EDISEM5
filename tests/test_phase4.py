import numpy as np
import pandas as pd

from ayurveda_kg.phase4 import gold_model_scores, summarize


def test_summarize_averages_seeds_within_fold_then_reports_mean_std_min_max_over_folds():
    rows = []
    for fold, a in enumerate([0.6, 0.8]):
        for seed, d in enumerate([-0.1, 0.1]):                       # seeds average out to a
            rows.append({"split": "cold_herb", "model": "gnn", "fold": fold, "seed": seed, "auroc": a + d, "auprc": 0.5, "p_at_100": 1.0,
                         "n_test": 10, "n_pos": 4, "n_train": 20})
    s = summarize(pd.DataFrame(rows)).iloc[0]
    assert s["auroc_mean"] == pytest_approx(0.7) and s["auroc_min"] == pytest_approx(0.6) and s["auroc_max"] == pytest_approx(0.8)
    assert s["auroc_std"] == pytest_approx(np.std([0.6, 0.8], ddof=1)) and s["n_folds"] == 2


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-9)


def test_gold_model_scores_herb_drug_mean_and_percentile():
    preds = pd.DataFrame({"compound": ["c1", "c2", "c3", "c4"], "drug": ["drug:a", "drug:a", "drug:a", "drug:a"], "p": [0.9, 0.7, 0.2, 0.1]})
    contains = pd.DataFrame({"src": ["herb:H1", "herb:H1", "herb:H2", "herb:H2"], "dst": ["c1", "c2", "c3", "c4"]})
    gold = pd.DataFrame({"herb": ["H1", "H2", "H3"], "drug": ["a", "a", "a"], "label": [1, 0, 1], "mechanism": ["PK", "none", "PK"]})
    out = gold_model_scores(preds, contains, gold).set_index("herb")
    assert out.loc["H1", "model_score"] == pytest_approx(0.8) and out.loc["H2", "model_score"] == pytest_approx(0.15)
    assert out.loc["H1", "model_percentile"] == 1.0 and out.loc["H2", "model_percentile"] == 0.5
    assert pd.isna(out.loc["H3", "model_score"])                       # herb with no held-out predictions stays visible as NaN


# ---- report ----
from ayurveda_kg.phase4 import make_phase4_report


def fake_results():
    rows = []
    for split in ["cold_compound", "cold_herb", "cold_drug"]:
        for model, base in [("prior", 0.55), ("rf", 0.9 if split != "cold_drug" else 0.6), ("gnn", 0.92 if split != "cold_drug" else 0.5),
                            ("rf_leaky", 0.999)]:
            for fold in range(2):
                for seed in ([0, 1] if model == "gnn" else [0]):
                    rows.append({"split": split, "model": model, "seed": seed, "fold": fold, "n_train": 10, "auroc": base + 0.01 * fold,
                                 "auprc": base / 2, "p_at_100": 0.9, "n_test": 20, "n_pos": 5, "seconds": 1.0})
    return pd.DataFrame(rows)


def test_report_computes_winners_and_leakage_gap_from_the_numbers():
    gold = pd.DataFrame([{"herb": "Piper nigrum", "drug": "phenytoin", "label": 1, "mechanism": "PK", "evidence": "human_patient_pk",
                          "frac_pos": 0.48, "percentile": 0.94, "model": "rf", "model_score": 0.7, "model_percentile": 0.9}])
    md = make_phase4_report(fake_results(), gold)
    assert "cold_compound" in md and "| rf |" in md and "| gnn |" in md
    assert "GNN has the higher mean AUROC than the Random Forest on 2 of 3 cold splits" in md      # gnn wins compound+herb, loses drug
    assert "cold_drug: rf" in md                                                                    # the loser is named, not hidden
    assert "leakage" in md.lower() and "0.999" in md
    assert "Piper nigrum" in md and "mechanistic hypothesis" in md
    assert "per-fold" in md.lower()


def test_report_includes_grouped_auroc_and_mlp_ablation_when_given():
    grouped = pd.DataFrame([{"split": "cold_drug", "model": "rf", "within_drug_auroc": 0.88, "n_drugs": 20,
                             "within_compound_auroc": 0.61, "n_compounds": 1500}])
    res = fake_results()
    res = pd.concat([res, res[res.model == "gnn"].assign(model="mlp", auroc=lambda d: d.auroc - 0.002)], ignore_index=True)
    gold = pd.DataFrame(columns=["herb", "drug", "label", "mechanism", "evidence", "frac_pos", "percentile", "model", "model_score", "model_percentile"])
    md = make_phase4_report(res, gold, grouped)
    assert "within-drug" in md.lower() and "0.610" in md and "0.880" in md
    assert "no message passing" in md.lower() and "mlp" in md


def test_report_states_drug_side_vs_pooled_and_gold_negatives_vs_pk_positives():
    grouped = pd.DataFrame([{"split": "cold_drug", "model": "rf", "within_drug_auroc": 0.99, "n_drugs": 20, "within_compound_auroc": 0.60, "n_compounds": 99},
                            {"split": "cold_drug", "model": "gnn", "within_drug_auroc": 0.98, "n_drugs": 20, "within_compound_auroc": 0.50, "n_compounds": 99}])
    g = lambda herb, label, mech, pct: {"herb": herb, "drug": "warfarin", "label": label, "mechanism": mech, "evidence": "e", "frac_pos": 0.5, "percentile": 0.5,
                                         "model": "rf", "model_score": 0.5, "model_percentile": pct}
    gold = pd.DataFrame([g("A", 1, "PK", 0.9), g("B", 1, "PK", 0.5), g("C", 0, "none", 0.95), g("D", 0, "none", 0.2)])
    md = make_phase4_report(fake_results(), gold, grouped)
    assert "drug-side (within-compound) AUROC 0.600" in md and "mostly compound-side" in md
    assert "rf: gold negatives at percentiles 0.95, 0.20; PK positives 0.50-0.90 (median 0.70); 1 of 2 negatives rank above every PK positive" in md
