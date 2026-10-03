"""Guards the NARRATIVE of the paper drafts: each directional claim in the prose is asserted on the real numbers, so a re-run that changes a result
fails here instead of leaving the text stale. Skipped where the local processed data is absent (IMPPAT-derived data is not committed)."""
import re
from pathlib import Path

import pytest

NEEDED = ["data/processed/phase5/facts.json", "data/processed/rag/facts.json", "data/processed/phase4/results.csv", "data/processed/kg/nodes_Herb.parquet"]
pytestmark = pytest.mark.skipif(not all(Path(p).exists() for p in NEEDED), reason="local processed data not present")


@pytest.fixture(scope="module")
def n():
    from ayurveda_kg.paper import gather_numbers
    return gather_numbers()


COLD = ("cold_compound", "cold_herb", "cold_drug")


def test_random_forest_beats_the_gnn_on_every_cold_split(n):
    assert all(n[f"p4_{s}_rf_auroc"] > n[f"p4_{s}_gnn_auroc"] for s in COLD)


def test_graph_message_passing_adds_little_on_compound_and_herb_splits(n):
    assert all(abs(n[f"p4_gnn_minus_mlp_{s}"]) < 0.02 for s in ("cold_compound", "cold_herb"))


def test_leakage_inflates_every_cold_split_and_most_for_cold_herb(n):
    assert all(n[f"p4_leak_{s}"] > 0 for s in COLD)
    assert n["p4_leak_cold_herb"] == max(n[f"p4_leak_{s}"] for s in COLD)


def test_pooled_cold_drug_score_hides_weak_drug_side_skill(n):
    assert n["p4_cold_drug_rf_auroc"] > 0.9 and 0.5 <= n["p4_cold_drug_rf_within_compound"] < 0.8
    assert n["p4_cold_drug_gnn_within_compound"] < n["p4_cold_drug_rf_within_compound"]
    assert n["p4_cold_drug_rf_within_drug"] > n["p4_cold_drug_rf_within_compound"]


def test_alias_and_graph_help_on_the_vocabulary_gap_in_the_stated_order(n):
    assert n["p6_gap_plain_r5"] < n["p6_gap_alias_r5"] < n["p6_gap_aliasgraph_r5"]
    assert n["p6_gap_plain_r5"] < n["p6_gap_graph_r5"] < n["p6_gap_aliasgraph_r5"]
    assert n["p6_retr_graph_r5"] > n["p6_retr_plain_r5"]


def test_kg_facts_make_the_difference_and_k_first_does_not_help(n):
    for q in ("pair_score", "top_drugs", "top_herbs"):
        assert n[f"p6_kg_full_{q}"] > n[f"p6_kg_plain_{q}"]
    assert n["p6_kg_full_kfirst_top_drugs"] < n["p6_kg_full_top_drugs"] and n["p6_kg_full_kfirst_top_herbs"] < n["p6_kg_full_top_herbs"]
    assert n["p6_refusal_full_top_drugs"] > 0                                           # the prose says the model refused despite having the answer


def test_the_overreading_fix_did_not_make_things_worse(n):
    before = n["p6_over_before_n"] / n["p6_over_before_d"]
    after = n["p6_over_after_n"] / n["p6_over_after_d"]
    assert after <= before


def test_composition_constraints_hold_and_direction_is_mixed_as_stated(n):
    assert n["p5_min_coverage"] >= 0.8 - 1e-9 and n["p5_n_flags"] == 0
    assert 0 < n["p5_median_red"] < 0.1                                                  # "modest"
    assert n["p5_dir_pos_mean"] > 0.5 and n["p5_dir_neg_mean"] > 0                       # mostly right direction, but also lowers trial-negative herbs
    assert n["p5_random_median"] < n["p5_equal_median"]


def test_both_drafts_render_with_no_unresolved_placeholder(n):
    from ayurveda_kg.paper import render
    for t in Path("docs/paper").glob("*.md.tmpl"):
        out = render(t.read_text(encoding="utf-8"), n)
        assert "{{" not in out and not re.search(r"\b(None|nan|NaN)\b", out), t.name          # exact Python/pandas tokens; the English word 'none' and names such as Ananth are fine
