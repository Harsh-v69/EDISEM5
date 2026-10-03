import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.phase5 import make_phase5_report, run_scenarios, select_cases, sweep_constraints

RISK = pd.DataFrame([{"herb": h, "drug": d, "risk_rf": r, "risk_silver": s}
                     for h, d, r, s in [("A", "x", 0.9, 0.8), ("A", "y", 0.1, 0.1), ("B", "x", 0.1, 0.2), ("B", "y", 0.5, 0.4),
                                        ("C", "x", 0.4, 0.4), ("C", "y", 0.4, 0.3)]])
USES = {"A": {"u1", "u2"}, "B": {"u2", "u3"}, "C": {"u3"}}
FORMS = pd.DataFrame([{"id": "F1", "name": "n1", "kind": "afi", "n_ingredients": 3, "in_scope": ["A", "B"]},
                      {"id": "F3", "name": "n3", "kind": "api", "n_ingredients": 4, "in_scope": ["A", "B", "C"]}])


def test_random_baseline_is_valid_reproducible_and_differs_from_equal_parts():
    eq = run_scenarios(FORMS, RISK, USES, ["x"], tau=0.5)
    r1 = run_scenarios(FORMS, RISK, USES, ["x"], tau=0.5, baseline_seed=1)
    r2 = run_scenarios(FORMS, RISK, USES, ["x"], tau=0.5, baseline_seed=1)
    assert r1["w0"].tolist() == r2["w0"].tolist() and r1["w0"].tolist() != eq["w0"].tolist()
    assert (r1["status"] == "optimal").all() and (r1["objective"] <= r1["baseline_objective"] + 1e-9).all()


def test_sweep_constraints_returns_one_row_per_setting_and_looser_bounds_never_help_less():
    sw = sweep_constraints(FORMS, RISK, USES, ["x", "y"], settings=[(0.9, 0.9, 1.1), (0.5, 0.5, 2.0), (0.5, 0.1, 4.0)])
    assert len(sw) == 3 and {"tau", "lo_frac", "hi_mult", "median_rel_reduction", "mean_rel_reduction", "n_scenarios"} <= set(sw.columns)
    assert sw["mean_rel_reduction"].is_monotonic_increasing or sw["mean_rel_reduction"].iloc[-1] >= sw["mean_rel_reduction"].iloc[0]


def test_select_cases_finds_gold_linked_scenarios_and_reports_the_share_change_of_the_named_herb():
    main = run_scenarios(FORMS, RISK, USES, ["x", "y"], tau=0.3, lo_frac=0.0, hi_mult=3.0)
    cases = select_cases(main, [("A", "x", "gold: interaction (human PK)")])
    assert len(cases) >= 1 and (cases["herb"] == "A").all() and (cases["drug"] == "x").all()
    assert "share_before" in cases.columns and "share_after" in cases.columns
    assert (cases["share_after"] <= cases["share_before"] + 1e-9).all()          # A is the risky herb for drug x: its share must not rise
    assert "gold" in cases["note"].iloc[0]


def test_report_has_all_sections_with_computed_numbers_and_the_proxy_caveat():
    main = run_scenarios(FORMS, RISK, USES, ["x", "y"], eval_col="risk_silver", tau=0.5)
    sw = sweep_constraints(FORMS, RISK, USES, ["x", "y"], settings=[(0.8, 0.5, 2.0), (0.95, 0.5, 2.0)])
    rand = {"equal_median": 0.2, "random_median": 0.15, "n": 10}
    cases = select_cases(main, [("A", "x", "gold: interaction")])
    md = make_phase5_report(FORMS, main, sw, rand, cases, n_formulations_total=5)
    for must in ["# Phase 5 results", "## Coverage of the formulation set", "## Optimiser behaviour", "## Does a suggestion transfer",
                 "## Sensitivity to constraints", "## Sensitivity to the equal-parts assumption", "## Literature-linked case studies",
                 "risk proxy", "not a dosing recommendation"]:
        assert must in md, must
    assert "5 formulations" in md and "2 of 5" in md                                     # 2 formulations have >= 2 in-scope herbs


# ---- direction-of-effect check against the published studies ----
from ayurveda_kg.phase5 import direction_check


def test_direction_check_reports_how_often_the_published_herbs_share_is_reduced():
    main = pd.DataFrame([
        {"formulation_id": "F1", "drug": "x", "herbs": "A; B", "w0": "0.250; 0.250", "w": "0.125; 0.375"},      # A reduced by 50%
        {"formulation_id": "F2", "drug": "x", "herbs": "B; A", "w0": "0.200; 0.200", "w": "0.150; 0.250"},      # A increased by 25%
        {"formulation_id": "F3", "drug": "x", "herbs": "A; C", "w0": "0.300; 0.300", "w": "0.150; 0.450"},      # A reduced by 50%
        {"formulation_id": "F4", "drug": "y", "herbs": "A; B", "w0": "0.250; 0.250", "w": "0.250; 0.250"}])      # other drug: ignored for pair (A, x)
    gold = pd.DataFrame([{"herb": "A", "drug": "x", "label": 1, "mechanism": "PK"}, {"herb": "Z", "drug": "x", "label": 1, "mechanism": "PK"}])
    out = direction_check(main, gold).set_index(["herb", "drug"])
    r = out.loc[("A", "x")]
    assert r["n_formulations"] == 3 and r["frac_reduced"] == pytest.approx(2 / 3) and r["frac_increased"] == pytest.approx(1 / 3)
    assert r["median_change"] == pytest.approx(-0.5)
    assert ("Z", "x") not in out.index                                              # herb in no scenario: omitted, not reported as 0%


def test_report_includes_the_direction_check_with_both_the_good_and_the_bad_news():
    forms = pd.DataFrame([{"id": "F1", "name": "n1", "kind": "afi", "n_ingredients": 3, "in_scope": ["A", "B"]}])
    main = run_scenarios(forms, RISK, USES, ["x", "y"], eval_col="risk_silver", tau=0.5)
    sw = sweep_constraints(forms, RISK, USES, ["x", "y"], settings=[(0.8, 0.5, 2.0)])
    d = pd.DataFrame([{"herb": "A", "drug": "x", "gold_label": 1, "mechanism": "PK", "n_formulations": 10, "frac_reduced": 0.9, "frac_increased": 0.1, "median_change": -0.3},
                      {"herb": "B", "drug": "y", "gold_label": 0, "mechanism": "none", "n_formulations": 8, "frac_reduced": 0.75, "frac_increased": 0.25, "median_change": -0.2}])
    md = make_phase5_report(forms, main, sw, {"equal_median": 0.1, "random_median": 0.1, "n": 2}, select_cases(main, []), 5, direction=d)
    assert "## Direction check against the published studies" in md and "90%" in md and "75%" in md
    assert "published interaction" in md and "no interaction" in md


# ---- facts export (numbers the paper drafts are rendered from) ----
from ayurveda_kg.phase5 import phase5_facts


def test_phase5_facts_are_flat_json_safe_numbers_matching_the_report_inputs():
    forms = pd.DataFrame([{"id": "F1", "name": "n1", "kind": "afi", "n_ingredients": 3, "in_scope": ["A", "B"]},
                          {"id": "F2", "name": "n2", "kind": "api", "n_ingredients": 1, "in_scope": ["A"]},
                          {"id": "F3", "name": "n3", "kind": "afi", "n_ingredients": 2, "in_scope": []}])
    main = run_scenarios(forms, RISK, USES, ["x", "y"], eval_col="risk_silver", tau=0.5)
    sw = sweep_constraints(forms, RISK, USES, ["x", "y"], settings=[(0.8, 0.5, 2.0), (0.5, 0.0, 4.0)])
    d = pd.DataFrame([{"herb": "A", "drug": "x", "gold_label": 1, "mechanism": "PK", "n_formulations": 10, "frac_reduced": 0.8, "frac_increased": 0.2, "median_change": -0.3},
                      {"herb": "B", "drug": "y", "gold_label": 0, "mechanism": "none", "n_formulations": 8, "frac_reduced": 0.5, "frac_increased": 0.5, "median_change": 0.0}])
    f = phase5_facts(forms, main, sw, {"equal_median": 0.04, "random_median": 0.03, "n": 5}, d)
    import json
    json.dumps(f)                                                                      # must be serialisable
    assert f["forms"] == 3 and f["forms_scoped"] == 2 and f["forms_opt"] == 1 and f["scenarios"] == len(main) == 2
    assert f["median_red"] == pytest.approx(main["rel_reduction"].median()) and 0 <= f["transfer_improve_frac"] <= 1
    assert f["sweep_max_red"] == pytest.approx(sw["median_rel_reduction"].max()) and f["sweep_min_red"] == pytest.approx(sw["median_rel_reduction"].min())
    assert f["random_median"] == 0.03 and f["equal_median"] == 0.04
    assert f["dir_pos_mean"] == pytest.approx(0.8) and f["dir_neg_mean"] == pytest.approx(0.5) and f["dir_pos_min"] == pytest.approx(0.8)
