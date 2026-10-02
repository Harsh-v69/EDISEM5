import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.phase5 import build_problem, herb_lookup, parse_formulation_records, run_scenarios

RISK = pd.DataFrame([{"herb": h, "drug": d, "risk_rf": r, "risk_silver": s}
                     for h, d, r, s in [("A", "x", 0.9, 0.8), ("A", "y", 0.1, 0.1), ("B", "x", 0.1, 0.2), ("B", "y", 0.5, 0.4),
                                        ("C", "x", 0.4, 0.4), ("C", "y", 0.4, 0.3)]])
USES = {"A": {"u1", "u2"}, "B": {"u2", "u3"}, "C": {"u3"}}


def test_herb_lookup_resolves_aliases_case_insensitively():
    scope = {"herbs": [{"imppat_name": "Tinospora sinensis", "aliases": ["Tinospora cordifolia", "Giloy"]}]}
    m = herb_lookup(scope)
    assert m["tinospora cordifolia"] == "Tinospora sinensis" and m["giloy"] == "Tinospora sinensis" and m["tinospora sinensis"] == "Tinospora sinensis"


def test_parse_formulation_records_marks_in_scope_and_keeps_total_ingredient_count(tmp_path):
    d = tmp_path / "formulations"
    d.mkdir()
    page = ('<html><body><div>Formulation identifier: AFI000001 Dosage (according to X): 1 g Ingredients of Test Formulation name Ingredient name Plant part</div>'
            '<table><thead><tr><th>Formulation name</th><th>Ingredient name</th><th>Plant part</th></tr></thead><tbody>'
            '<tr><td>T</td><td>Piper nigrum</td><td>fruit</td></tr><tr><td>T</td><td>Borax</td><td></td></tr>'
            '<tr><td>T</td><td>Tinospora cordifolia</td><td>stem</td></tr></tbody></table></body></html>')
    (d / "afi_1.html").write_text(page, encoding="utf-8")
    scope = {"herbs": [{"imppat_name": "Piper nigrum", "aliases": []}, {"imppat_name": "Tinospora sinensis", "aliases": ["Tinospora cordifolia"]}]}
    recs = parse_formulation_records(d, herb_lookup(scope))
    assert len(recs) == 1
    r = recs.iloc[0]
    assert r["n_ingredients"] == 3 and sorted(r["in_scope"]) == ["Piper nigrum", "Tinospora sinensis"] and r["kind"] == "afi"


def test_build_problem_uses_equal_parts_baseline_and_union_use_matrix():
    p = build_problem(["A", "B"], 4, RISK, ["x", "y"], USES, "risk_rf")
    assert p["herbs"] == ["A", "B"] and p["w0"].tolist() == [0.25, 0.25]            # 2 of 4 ingredients are in scope, equal parts
    assert p["risk"].tolist() == [[0.9, 0.1], [0.1, 0.5]]
    assert p["A"].shape == (2, 3) and p["A"].sum() == 4                                  # uses u1,u2,u3; four herb-use ones
    dup = build_problem(["A", "A"], 4, RISK, ["x"], USES, "risk_rf")                       # same herb listed twice (two plant parts)
    assert dup["herbs"] == ["A"] and dup["w0"].tolist() == [0.5]


def test_run_scenarios_only_optimises_formulations_with_two_or_more_in_scope_herbs_and_never_gets_worse():
    forms = pd.DataFrame([{"id": "F1", "name": "n1", "kind": "afi", "n_ingredients": 3, "in_scope": ["A", "B"]},
                          {"id": "F2", "name": "n2", "kind": "afi", "n_ingredients": 2, "in_scope": ["A"]},
                          {"id": "F3", "name": "n3", "kind": "api", "n_ingredients": 4, "in_scope": ["A", "B", "C"]}])
    out = run_scenarios(forms, RISK, USES, ["x", "y"], risk_col="risk_rf", eval_col="risk_silver", tau=0.5, lo_frac=0.5, hi_mult=2.0)
    assert set(out["formulation_id"]) == {"F1", "F3"}                                      # F2 has a single in-scope herb: nothing to re-weight
    assert len(out) == 2 * 2                                                               # two formulations x two single-drug scenarios
    assert (out["objective"] <= out["baseline_objective"] + 1e-9).all() and (out["rel_reduction"] >= -1e-9).all()
    assert {"eval_baseline", "eval_new", "eval_rel_reduction", "min_coverage_ratio", "status"} <= set(out.columns)
    assert (out["status"] == "optimal").all()
