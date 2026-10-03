"""Inter-rater agreement for the Jivha/Nadi module (practitioner labels). Needs no clinical partner to test; real use needs real annotations.
Cohen's kappa is undefined (NaN) when chance agreement is 1 or there are no items: never a fake number."""
import itertools

import pandas as pd


def cohens_kappa(r1, r2) -> float:
    if len(r1) != len(r2):
        raise ValueError("the two label lists must have the same length")
    n = len(r1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(r1, r2)) / n
    pe = sum((sum(a == c for a in r1) / n) * (sum(b == c for b in r2) / n) for c in set(r1) | set(r2))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pairwise_kappa(long: pd.DataFrame) -> pd.DataFrame:
    """long: one row per (item, rater, label). Kappa for every rater pair on the items BOTH labelled."""
    wide = long.pivot(index="item", columns="rater", values="label")
    rows = []
    for a, b in itertools.combinations(sorted(wide.columns), 2):
        both = wide[[a, b]].dropna()
        rows.append({"rater_1": a, "rater_2": b, "n_items": len(both), "kappa": cohens_kappa(both[a].tolist(), both[b].tolist()),
                     "observed_agreement": float((both[a] == both[b]).mean()) if len(both) else float("nan")})
    return pd.DataFrame(rows, columns=["rater_1", "rater_2", "n_items", "kappa", "observed_agreement"])


def validate_annotations(long: pd.DataFrame, allowed) -> list[str]:
    out = []
    for (i, r), n in long.groupby(["item", "rater"]).size().items():
        if n > 1:
            out.append(f"duplicate label: rater {r} labelled item {i} {n} times")
    out += [f"unknown label {l!r} (item {i}, rater {r})" for i, r, l in zip(long["item"], long["rater"], long["label"]) if l not in set(allowed)]
    out += [f"item {i} has a single rater" for i, n in long.groupby("item")["rater"].nunique().items() if n < 2]
    return out
