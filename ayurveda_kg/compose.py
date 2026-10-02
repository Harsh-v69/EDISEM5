"""Safe-composition optimiser: re-weight a formulation's (in-scope) herbs to minimise predicted interaction risk against a patient's
drugs, keeping each herb's share within bounds of the baseline and each therapeutic use above a coverage floor.
IMPORTANT: shares are a risk PROXY. IMPPAT has no proportions or abundances, so the baseline is an assumption and the output is
hypothesis-generating, not a dosing recommendation."""
import numpy as np
from scipy.optimize import linprog


def optimise(risk, w0, drug_weights=None, A=None, tau=0.8, lo_frac=0.5, hi_mult=2.0, objective="sum") -> dict:
    """risk [n_herbs, n_drugs]; w0 [n_herbs] baseline shares of the optimisable herbs (their total S may be < 1: the rest is fixed);
    A [n_herbs, n_uses] 0/1 therapeutic-use matrix; objective 'sum' (weighted total risk) or 'max' (worst single drug)."""
    risk, w0 = np.asarray(risk, float), np.asarray(w0, float)
    n, d = risk.shape
    S = w0.sum()
    dw = np.ones(d) if drug_weights is None else np.asarray(drug_weights, float)
    lo, hi = lo_frac * w0, np.minimum(S, hi_mult * w0)
    hi = np.maximum(hi, w0)                                          # the baseline itself must stay feasible
    A_ub, b_ub = [], []
    if A is not None:
        A = np.asarray(A, float)
        base = A.T @ w0
        for u in np.where(base > 0)[0]:
            row = np.zeros(n + (objective == "max")); row[:n] = -A[:, u]
            A_ub.append(row); b_ub.append(-tau * base[u])
    if objective == "sum":
        c, bounds = risk @ dw, list(zip(lo, hi))
        A_eq = [np.ones(n)]
    elif objective == "max":
        c = np.zeros(n + 1); c[-1] = 1
        for j in range(d):
            row = np.zeros(n + 1); row[:n] = risk[:, j]; row[-1] = -1
            A_ub.append(row); b_ub.append(0.0)
        bounds = list(zip(lo, hi)) + [(0, None)]
        A_eq = [np.append(np.ones(n), 0.0)]
    else:
        raise ValueError(objective)
    res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq), b_eq=[S], bounds=bounds, method="highs")
    ok = res.status == 0
    w = res.x[:n] if ok else w0.copy()
    obj = (lambda x: float((x @ risk @ dw))) if objective == "sum" else (lambda x: float((x @ risk).max()))
    flags = [f"herb {i} removed" if w[i] < 1e-9 < w0[i] else f"herb {i} reduced to {w[i] / w0[i]:.0%} of baseline"
             for i in range(n) if w[i] < 0.5 * w0[i] - 1e-9]
    cov = float("nan")
    if A is not None and (A.T @ w0 > 0).any():
        base = A.T @ w0; k = base > 0
        cov = float(((A.T @ w)[k] / base[k]).min())
    return {"w": w, "w0": w0, "objective": obj(w), "baseline_objective": obj(w0), "status": "optimal" if ok else "infeasible",
            "per_drug_risk": w @ risk, "baseline_per_drug_risk": w0 @ risk, "min_coverage_ratio": cov, "flags": flags}
