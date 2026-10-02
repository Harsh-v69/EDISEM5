"""Silver HDI labels at (compound, drug) level, derived mechanistically from the KG. See docs/superpowers/plans/2026-10-02-phase3-labels-splits.md."""
import pandas as pd

CYP5 = ("CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4")  # the enzymes SwissADME predicts


def _by_src(df, enzymes):
    out = {}
    for s, d in zip(df["src"], df["dst"]):
        g = d[5:]
        if d.startswith("gene:") and g in enzymes:
            out.setdefault(s, set()).add(g)
    return out


def silver_labels(nodes, edges, enzymes=CYP5) -> pd.DataFrame:
    """label 1 = compound predicted to inhibit an enzyme the drug is a substrate of;
    0 = for every enzyme the compound inhibits, the drug is a verified non-substrate (vacuous if it inhibits none);
    -1 = undetermined (never guessed). Compounds with no SwissADME predictions are always -1."""
    inh = _by_src(edges["predicted_cyp_inhibitor"], enzymes)
    sub = _by_src(edges["substrate_of"], enzymes)
    non = _by_src(edges["non_substrate_of"], enzymes)
    conflicts = sorted((d, g) for d in sub for g in sub[d] & non.get(d, set()))
    rows = []
    for c, ok in zip(nodes["Compound"]["id"], nodes["Compound"]["has_adme"]):
        ci = inh.get(c, set())
        for d in nodes["Drug"]["id"]:
            ds, dn = sub.get(d, set()), non.get(d, set()) - sub.get(d, set())  # substrate beats non-substrate
            hit, unknown = ci & ds, ci - ds - dn
            if not ok:
                label, enz = -1, set()
            elif hit:
                label, enz = 1, hit
            elif unknown:
                label, enz = -1, unknown
            else:
                label, enz = 0, set()
            rows.append({"compound": c, "drug": d, "label": label, "enzymes": ";".join(sorted(enz))})
    df = pd.DataFrame(rows, columns=["compound", "drug", "label", "enzymes"])
    df.attrs["conflicts"] = conflicts
    return df


def herb_drug_scores(labels: pd.DataFrame, contains: pd.DataFrame) -> pd.DataFrame:
    """Aggregate compound-drug silver labels to herb-drug level. frac_pos is NaN (not 0) when no compound is labelled."""
    lab = labels.assign(pos=(labels["label"] == 1).astype(int), known=labels["label"].isin([0, 1]).astype(int))
    m = contains.rename(columns={"src": "herb", "dst": "compound"})[["herb", "compound"]].merge(lab, on="compound")
    g = m.groupby(["herb", "drug"]).agg(n_pos=("pos", "sum"), n_labelled=("known", "sum"), n_compounds=("compound", "nunique")).reset_index()
    g["frac_pos"] = g["n_pos"] / g["n_labelled"].where(g["n_labelled"] > 0)
    return g


def gold_vs_silver(gold: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Attach the silver herb-drug score and its percentile (among all scored herb-drug pairs) to each gold pair."""
    s = scores.assign(herb=scores["herb"].str.removeprefix("herb:"), drug=scores["drug"].str.removeprefix("drug:"))
    s = s[s["n_labelled"] > 0].assign(percentile=lambda d: d["frac_pos"].rank(pct=True))
    return gold.merge(s[["herb", "drug", "n_pos", "n_labelled", "frac_pos", "percentile"]], on=["herb", "drug"], how="left")
