"""Baselines to beat: per-drug prior, Random Forest on pair features, matrix factorisation.
Each *_fit_predict returns f(train_pairs, test_pairs, fold) -> probabilities aligned with test_pairs (or is such a function)."""
import numpy as np
import pandas as pd


def prior_fit_predict(train, test, fold) -> np.ndarray:
    """Per-drug positive rate from training pairs; unseen drug -> global rate."""
    rate, g = train.groupby("drug")["label"].mean(), train["label"].mean()
    return test["drug"].map(rate).fillna(g).to_numpy(dtype=float)


def pair_matrix(fe, pairs, extra_compound=None, extra_drug=None) -> np.ndarray:
    ci = {c: i for i, c in enumerate(fe.compound_ids)}
    di = {d: i for i, d in enumerate(fe.drug_ids)}
    a, b = pairs["compound"].map(ci).to_numpy(), pairs["drug"].map(di).to_numpy()
    Xc = fe.Xc if extra_compound is None else np.hstack([fe.Xc, extra_compound])
    Xd = fe.Xd if extra_drug is None else np.hstack([fe.Xd, extra_drug])
    return np.hstack([Xc[a], Xd[b]])


def rf_fit_predict(fe, n_estimators=200, seed=0, extra_compound=None, extra_drug=None, min_samples_leaf=2):
    """Random Forest on [compound features | drug features]. extra_* are for the leakage ablation only."""
    from sklearn.ensemble import RandomForestClassifier

    def f(train, test, fold):
        rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                                    n_jobs=-1, random_state=seed)
        rf.fit(pair_matrix(fe, train, extra_compound, extra_drug), train["label"].values)
        return rf.predict_proba(pair_matrix(fe, test, extra_compound, extra_drug))[:, 1]
    return f


def mf_fit_predict(rank=8, epochs=300, lr=0.05, reg=1e-3, seed=0):
    """Logistic matrix factorisation (+ biases). Cannot score a compound or drug it has not seen: those fall back to the prior,
    which is exactly why it is only a meaningful reference on the random-pair split."""
    def f(train, test, fold):
        cs, ds = sorted(set(train["compound"])), sorted(set(train["drug"]))
        ci, di = {c: i for i, c in enumerate(cs)}, {d: i for i, d in enumerate(ds)}
        a, b, y = train["compound"].map(ci).to_numpy(), train["drug"].map(di).to_numpy(), train["label"].to_numpy(float)
        rng = np.random.default_rng(seed)
        U, V = rng.normal(0, 0.1, (len(cs), rank)), rng.normal(0, 0.1, (len(ds), rank))
        bc, bd, b0 = np.zeros(len(cs)), np.zeros(len(ds)), 0.0
        m = {k: 0.0 for k in ("U", "V", "bc", "bd")}; v = {k: 0.0 for k in m}      # Adam state
        sig = lambda z: 1 / (1 + np.exp(-z))
        for t in range(1, epochs + 1):
            err = sig((U[a] * V[b]).sum(1) + bc[a] + bd[b] + b0) - y
            gU, gV = np.zeros_like(U), np.zeros_like(V)
            np.add.at(gU, a, err[:, None] * V[b]); np.add.at(gV, b, err[:, None] * U[a])
            gbc, gbd = np.zeros_like(bc), np.zeros_like(bd)
            np.add.at(gbc, a, err); np.add.at(gbd, b, err)
            grads = {"U": gU / len(y) + reg * U, "V": gV / len(y) + reg * V, "bc": gbc / len(y), "bd": gbd / len(y)}
            params = {"U": U, "V": V, "bc": bc, "bd": bd}
            for k, g in grads.items():
                m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g
                params[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            b0 -= lr * err.mean()
        base = prior_fit_predict(train, test, fold)
        ta, tb = test["compound"].map(ci), test["drug"].map(di)
        known = (ta.notna() & tb.notna()).to_numpy()
        out = base.copy()
        ia, ib = ta[known].astype(int).to_numpy(), tb[known].astype(int).to_numpy()
        out[known] = sig((U[ia] * V[ib]).sum(1) + bc[ia] + bd[ib] + b0)
        return out
    return f
