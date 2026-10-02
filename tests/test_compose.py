import numpy as np
import pytest

from ayurveda_kg.compose import optimise


def test_shifts_share_to_the_lower_risk_herb_up_to_the_bound():
    risk = np.array([[0.9], [0.1]])                                     # herb 0 risky, herb 1 safe, one patient drug
    r = optimise(risk, np.array([0.5, 0.5]), lo_frac=0.5, hi_mult=1.5)
    assert r["status"] == "optimal"
    assert r["w"] == pytest.approx([0.25, 0.75])                        # herb 0 at its lower bound (0.5 x 0.5), herb 1 takes the rest
    assert r["objective"] < r["baseline_objective"]


def test_bounds_and_sum_hold_and_baseline_is_always_feasible():
    risk = np.array([[0.2, 0.8], [0.7, 0.1], [0.4, 0.4]])
    w0 = np.array([0.5, 0.3, 0.2])
    r = optimise(risk, w0, lo_frac=0.4, hi_mult=2.0)
    assert r["w"].sum() == pytest.approx(1.0)
    assert (r["w"] >= 0.4 * w0 - 1e-9).all() and (r["w"] <= np.minimum(1, 2.0 * w0) + 1e-9).all()


def test_coverage_constraint_limits_the_shift():
    risk = np.array([[0.9], [0.1]])
    A = np.array([[1, 0], [0, 1]])                                       # each herb is the only provider of its own therapeutic use
    r = optimise(risk, np.array([0.5, 0.5]), A=A, tau=0.9, lo_frac=0.0, hi_mult=2.0)
    assert r["w"][0] >= 0.9 * 0.5 - 1e-9                                 # herb 0's use must keep 90% of its baseline coverage
    assert r["min_coverage_ratio"] >= 0.9 - 1e-9
    free = optimise(risk, np.array([0.5, 0.5]), lo_frac=0.0, hi_mult=2.0)
    assert free["w"][0] < r["w"][0]                                      # without the constraint the risky herb would be cut further


def test_max_objective_protects_the_worst_drug_where_sum_would_not():
    risk = np.array([[1.0, 0.0], [0.0, 0.6]])                           # herb 0 hurts drug 0, herb 1 hurts drug 1
    r_max = optimise(risk, np.array([0.5, 0.5]), objective="max", lo_frac=0.0, hi_mult=2.0)
    worst = (r_max["w"] @ risk).max()
    assert worst <= (np.array([0.5, 0.5]) @ risk).max() + 1e-9
    assert r_max["w"] == pytest.approx([0.375, 0.625], abs=1e-6)         # equalises 1.0*w0 and 0.6*w1 -> w0=0.375


def test_drug_weights_change_the_answer():
    risk = np.array([[0.9, 0.0], [0.0, 0.9]])
    a = optimise(risk, np.array([0.5, 0.5]), drug_weights=np.array([1.0, 0.0]), lo_frac=0.0, hi_mult=2.0)
    b = optimise(risk, np.array([0.5, 0.5]), drug_weights=np.array([0.0, 1.0]), lo_frac=0.0, hi_mult=2.0)
    assert a["w"][0] < 0.5 < b["w"][0]


def test_flags_report_herbs_reduced_below_half_or_removed():
    risk = np.array([[0.9], [0.1], [0.1]])
    r = optimise(risk, np.array([1 / 3] * 3), lo_frac=0.0, hi_mult=3.0)
    assert "removed" in r["flags"][0] and r["w"][0] == pytest.approx(0.0, abs=1e-9)
    kept = optimise(risk, np.array([1 / 3] * 3), lo_frac=0.5, hi_mult=3.0)
    assert not any("removed" in f for f in kept["flags"])                # default bounds never delete a herb


def test_shares_less_than_one_are_preserved_for_fixed_out_of_scope_ingredients():
    risk = np.array([[0.9], [0.1]])
    r = optimise(risk, np.array([0.3, 0.3]), lo_frac=0.5, hi_mult=2.0)    # the optimisable herbs hold 0.6 of the formulation
    assert r["w"].sum() == pytest.approx(0.6)


@pytest.mark.parametrize("objective", ["sum", "max"])
def test_random_problems_never_violate_constraints_or_get_worse(objective):
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, d, u = rng.integers(2, 7), rng.integers(1, 5), rng.integers(1, 5)
        risk, A = rng.random((n, d)), (rng.random((n, u)) < 0.5).astype(float)
        w0 = rng.random(n) + 0.05
        w0 = w0 / w0.sum() * rng.uniform(0.3, 1.0)
        lo, hi, tau = rng.uniform(0, 1), rng.uniform(1, 3), rng.uniform(0.5, 1.0)
        r = optimise(risk, w0, A=A, tau=tau, lo_frac=lo, hi_mult=hi, objective=objective)
        assert r["status"] == "optimal"
        assert r["w"].sum() == pytest.approx(w0.sum())
        assert (r["w"] >= lo * w0 - 1e-7).all() and (r["w"] <= np.minimum(w0.sum(), hi * w0) + 1e-7).all()
        base = A.T @ w0
        assert ((A.T @ r["w"])[base > 0] >= tau * base[base > 0] - 1e-7).all()
        assert r["objective"] <= r["baseline_objective"] + 1e-9
