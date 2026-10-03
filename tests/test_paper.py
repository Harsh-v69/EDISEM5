import re
from pathlib import Path

import pandas as pd
import pytest

from ayurveda_kg.paper import kg_facts, label_facts, phase4_facts, placeholders, render


def test_render_fills_values_with_filters_and_leaves_no_placeholder():
    nums = {"a": 0.1234, "b": 6493, "c": 0.4567, "d": "text"}
    out = render("x={{a|f3}} y={{b|int}} z={{c|pct}} w={{c|pct1}} v={{a|f2}} u={{d}} t={{b}}", nums)
    assert out == "x=0.123 y=6,493 z=46% w=45.7% v=0.12 u=text t=6493"
    assert "{{" not in out


def test_render_refuses_when_any_key_is_missing_and_names_every_missing_key():
    with pytest.raises(KeyError) as e:
        render("{{a}} {{missing_one}} {{missing_two|pct}}", {"a": 1})
    assert "missing_one" in str(e.value) and "missing_two" in str(e.value)


def test_render_refuses_none_and_unknown_filter():
    with pytest.raises(KeyError):
        render("{{a|pct}}", {"a": None})                       # a missing number must never silently become 'None%'
    with pytest.raises(ValueError, match="filter"):
        render("{{a|bogus}}", {"a": 1})


def test_placeholders_lists_unique_keys():
    assert placeholders("{{a}} {{b|pct}} {{a|f3}}") == {"a", "b"}


def test_kg_and_label_facts_count_nodes_edges_and_label_classes():
    nodes = {"Herb": pd.DataFrame({"id": ["h1", "h2"]}), "Compound": pd.DataFrame({"id": list("abc")}), "Drug": pd.DataFrame({"id": ["d"]}), "Target": pd.DataFrame({"id": list("xyzw")})}
    edges = {"contains": pd.DataFrame({"src": ["h1"] * 3, "dst": list("abc")}), "ddi": pd.DataFrame({"src": ["d"], "dst": ["d2"]})}
    k = kg_facts(nodes, edges)
    assert k["kg_herbs"] == 2 and k["kg_compounds"] == 3 and k["kg_drugs"] == 1 and k["kg_targets"] == 4
    assert k["kg_edges_contains"] == 3 and k["kg_edges_ddi"] == 1
    lab = pd.DataFrame({"compound": list("aabb"), "drug": ["d"] * 4, "label": [1, 0, -1, 1], "enzymes": [""] * 4})
    gold = pd.DataFrame({"label": [1, 1, 0], "mechanism": ["PK", "PD", "none"]})
    l = label_facts(lab, gold)
    assert l["lab_pairs"] == 4 and l["lab_pos"] == 2 and l["lab_neg"] == 1 and l["lab_unl"] == 1 and l["lab_pos_compounds"] == 2
    assert l["gold_n"] == 3 and l["gold_neg"] == 1 and l["gold_pk_pos"] == 1


def test_phase4_facts_has_summary_grouped_leakage_and_graph_minus_mlp_keys():
    rows = []
    for split in ["cold_compound", "cold_herb", "cold_drug", "random_pair"]:
        for model, auc in [("prior", 0.7), ("rf", 0.9), ("mlp", 0.8), ("gnn", 0.78), ("rf_leaky", 0.99)]:
            for fold in range(2):
                rows.append({"split": split, "model": model, "seed": 0, "fold": fold, "auroc": auc + 0.01 * fold, "auprc": 0.5, "p_at_100": 1.0,
                             "n_test": 10, "n_pos": 4, "n_train": 20})
    grouped = pd.DataFrame([{"split": "cold_drug", "model": "rf", "within_drug_auroc": 0.98, "n_drugs": 27, "within_compound_auroc": 0.72, "n_compounds": 1127}])
    f = phase4_facts(pd.DataFrame(rows), grouped)
    assert f["p4_cold_compound_rf_auroc"] == pytest.approx(0.905) and f["p4_cold_compound_rf_auroc_std"] == pytest.approx(0.0070711, abs=1e-4)
    assert f["p4_cold_drug_rf_within_compound"] == pytest.approx(0.72) and f["p4_cold_drug_rf_within_drug"] == pytest.approx(0.98)
    assert f["p4_leak_cold_compound"] == pytest.approx(0.99 - 0.9) and f["p4_gnn_minus_mlp_cold_compound"] == pytest.approx(0.78 - 0.8)


def test_every_placeholder_in_the_real_templates_uses_a_known_prefix():
    prefixes = ("kg_", "lab_", "gold_", "p4_", "p5_", "p6_")
    for t in Path("docs/paper").glob("*.md.tmpl"):
        bad = [k for k in placeholders(t.read_text(encoding="utf-8")) if not k.startswith(prefixes)]
        assert not bad, (t.name, bad)
