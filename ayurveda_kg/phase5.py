"""Phase 5: risk matrices, formulation loading, composition scenarios and validation. Artifacts in data/processed/phase5 (IMPPAT-derived: local only)."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ayurveda_kg import baselines, curated, phase4
from ayurveda_kg.compose import optimise
from ayurveda_kg.ingest import imppat
from ayurveda_kg.labels import herb_drug_scores
from ayurveda_kg.risk import herb_drug_risk, oof_risk
from ayurveda_kg.scope import load_scope

OUT = Path("data/processed/phase5")
RAW = Path("data/raw/imppat")


def build_risk(seed=0) -> pd.DataFrame:
    """Out-of-fold RF herb-drug risk and the Phase 3 silver herb-drug score (alternative risk model for cross-checks)."""
    nodes, edges, masked, labels, fe, _ = phase4.load_inputs()
    risk = oof_risk(labels, baselines.rf_fit_predict(fe, seed=seed), k=5, seed=seed)
    h_rf = herb_drug_risk(risk, edges["contains"]).rename(columns={"risk": "risk_rf"})
    h_sil = herb_drug_scores(labels, edges["contains"]).rename(columns={"frac_pos": "risk_silver"})
    h_sil["herb"], h_sil["drug"] = h_sil["herb"].str.removeprefix("herb:"), h_sil["drug"].str.removeprefix("drug:")
    out = h_rf.merge(h_sil[["herb", "drug", "risk_silver"]], on=["herb", "drug"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    risk.to_parquet(OUT / "compound_drug_risk.parquet", index=False)
    out.to_parquet(OUT / "herb_drug_risk.parquet", index=False)
    return out


def load_uses(scope=None) -> dict:
    """herb (IMPPAT name) -> set of therapeutic uses, from the cached therapeutics pages."""
    scope = scope or load_scope()
    uses = {}
    for h in scope["herbs"]:
        p = RAW / "therapeutics" / f"{h['imppat_name'].replace(' ', '_')}.html"
        uses[h["imppat_name"]] = set(imppat.parse_therapeutics_page(p.read_text(encoding="utf-8", errors="ignore"))) if p.exists() else set()
    return uses


def herb_lookup(scope=None) -> dict:
    """lowercase ingredient name -> scoped herb IMPPAT name (canonical name or any alias)."""
    scope = scope or load_scope()
    m = {}
    for h in scope["herbs"]:
        for n in [h["imppat_name"], *h.get("aliases", [])]:
            m[n.lower()] = h["imppat_name"]
    return m


def parse_formulation_records(form_dir=RAW / "formulations", lookup=None) -> pd.DataFrame:
    """One row per cached formulation page: id, name, kind (afi/api), n_ingredients, in_scope (canonical scoped herb names)."""
    lookup = lookup or herb_lookup()
    rows = []
    for p in sorted(Path(form_dir).glob("*.html")):
        f = imppat.parse_formulation_page(p.read_text(encoding="utf-8", errors="ignore"))
        if not f["ingredients"]:
            continue
        herbs = [lookup[i["ingredient"].lower()] for i in f["ingredients"] if i["ingredient"].lower() in lookup]
        rows.append({"id": f["id"] or p.stem, "name": f["name"], "kind": p.stem.split("_")[0], "n_ingredients": len(f["ingredients"]),
                     "in_scope": herbs})
    return pd.DataFrame(rows, columns=["id", "name", "kind", "n_ingredients", "in_scope"])


def load_formulation_records(cache=OUT / "formulations.parquet", form_dir=RAW / "formulations", lookup=None, refresh=False) -> pd.DataFrame:
    """parse_formulation_records with an on-disk cache (parsing ~1,600 pages takes minutes). refresh=True re-parses."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    recs = parse_formulation_records(form_dir, lookup)
    cache.parent.mkdir(parents=True, exist_ok=True)
    recs.to_parquet(cache, index=False)
    return recs


def build_problem(in_scope, n_ingredients, risk, drugs, uses, risk_col, pivot=None) -> dict:
    """Optimisation inputs for one formulation. Baseline = equal parts across ALL ingredients (IMPPAT gives no proportions: an assumption);
    out-of-scope ingredients have no risk data and stay fixed. A herb listed twice (e.g. two plant parts) counts with double share."""
    counts = pd.Series(in_scope).value_counts(sort=False)
    herbs = list(dict.fromkeys(in_scope))
    pivot = risk.pivot(index="herb", columns="drug", values=risk_col) if pivot is None else pivot     # pass a precomputed pivot in loops
    r = pivot.reindex(index=herbs, columns=drugs)
    all_uses = sorted(set().union(*[uses.get(h, set()) for h in herbs]))
    A = np.array([[1.0 if u in uses.get(h, set()) else 0.0 for u in all_uses] for h in herbs])
    return {"herbs": herbs, "risk": r.to_numpy(float), "w0": np.array([counts[h] / n_ingredients for h in herbs]), "A": A, "uses": all_uses}


def run_scenarios(forms, risk, uses, drugs, risk_col="risk_rf", eval_col=None, tau=0.8, lo_frac=0.5, hi_mult=2.0, objective="sum",
                  baseline_seed=None) -> pd.DataFrame:
    """Single-drug scenarios for every formulation with >= 2 distinct in-scope herbs. eval_col = an independent risk model used only to
    evaluate how well a suggestion transfers (optimise with risk_col, score with eval_col). baseline_seed: draw the baseline shares at random
    (Dirichlet) instead of equal parts, to test how much the unknown-proportions assumption matters."""
    rows = []
    rng = None if baseline_seed is None else np.random.default_rng(baseline_seed)
    piv = risk.pivot(index="herb", columns="drug", values=risk_col)                     # pivot ONCE; the old per-scenario pivot made this very slow
    piv_eval = risk.pivot(index="herb", columns="drug", values=eval_col) if eval_col else None
    for f in forms.itertuples():
        if len(set(f.in_scope)) < 2:
            continue
        draw = None if rng is None else rng.dirichlet(np.ones(f.n_ingredients))
        base = build_problem(f.in_scope, f.n_ingredients, risk, list(drugs), uses, risk_col, pivot=piv)    # herb x all-drugs matrix, built once per formulation
        eval_all = build_problem(f.in_scope, f.n_ingredients, risk, list(drugs), uses, eval_col, pivot=piv_eval)["risk"] if eval_col else None
        for j, d in enumerate(drugs):
            p = {**base, "risk": base["risk"][:, [j]]}
            if draw is not None:
                p["w0"] = draw[:len(p["herbs"])]
            if np.isnan(p["risk"]).any():
                continue
            r = optimise(p["risk"], p["w0"], A=p["A"], tau=tau, lo_frac=lo_frac, hi_mult=hi_mult, objective=objective)
            row = {"formulation_id": f.id, "formulation": f.name, "drug": d, "n_in_scope": len(p["herbs"]), "n_ingredients": f.n_ingredients,
                   "baseline_objective": r["baseline_objective"], "objective": r["objective"],
                   "rel_reduction": 1 - r["objective"] / r["baseline_objective"] if r["baseline_objective"] > 0 else 0.0,
                   "min_coverage_ratio": r["min_coverage_ratio"], "status": r["status"], "n_flags": len(r["flags"]),
                   "n_removed": sum("removed" in x for x in r["flags"]),
                   "herbs": "; ".join(p["herbs"]), "w0": "; ".join(f"{x:.3f}" for x in r["w0"]), "w": "; ".join(f"{x:.3f}" for x in r["w"])}
            if eval_col:
                q = eval_all[:, [j]]
                b, n = float(r["w0"] @ q[:, 0]), float(r["w"] @ q[:, 0])
                row.update(eval_baseline=b, eval_new=n, eval_rel_reduction=1 - n / b if b > 0 else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def sweep_constraints(forms, risk, uses, drugs, settings, risk_col="risk_rf", objective="sum") -> pd.DataFrame:
    """settings: list of (tau, lo_frac, hi_mult). One summary row per setting."""
    rows = []
    for tau, lo, hi in settings:
        s = run_scenarios(forms, risk, uses, drugs, risk_col=risk_col, tau=tau, lo_frac=lo, hi_mult=hi, objective=objective)
        rows.append({"tau": tau, "lo_frac": lo, "hi_mult": hi, "n_scenarios": len(s),
                     "median_rel_reduction": float(s["rel_reduction"].median()) if len(s) else float("nan"),
                     "mean_rel_reduction": float(s["rel_reduction"].mean()) if len(s) else float("nan"),
                     "median_min_coverage": float(s["min_coverage_ratio"].median()) if len(s) else float("nan"),
                     "frac_with_removal": float((s["n_removed"] > 0).mean()) if len(s) else float("nan")})
    return pd.DataFrame(rows)


def select_cases(main: pd.DataFrame, picks, per_pick=3) -> pd.DataFrame:
    """picks: [(herb, drug, note)]. For each, up to `per_pick` scenarios containing that herb, with that herb's share before/after."""
    rows = []
    for herb, drug, note in picks:
        sub = main[(main["drug"] == drug) & main["herbs"].str.split("; ").map(lambda h: herb in h)]
        for r in sub.sort_values(["n_in_scope", "formulation_id"], ascending=[False, True]).head(per_pick).itertuples():
            i = r.herbs.split("; ").index(herb)
            rows.append({"formulation_id": r.formulation_id, "formulation": r.formulation, "herb": herb, "drug": drug,
                         "share_before": float(r.w0.split("; ")[i]), "share_after": float(r.w.split("; ")[i]),
                         "n_in_scope": r.n_in_scope, "rel_reduction": r.rel_reduction, "note": note})
    return pd.DataFrame(rows, columns=["formulation_id", "formulation", "herb", "drug", "share_before", "share_after", "n_in_scope",
                                       "rel_reduction", "note"])


def direction_check(main: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """For each gold herb-drug pair: across all scenarios (formulations) containing the herb for that drug, how often is the herb's share
    reduced / increased? Pairs whose herb appears in no scenario are omitted (never reported as 0%)."""
    rows = []
    for g in gold.itertuples():
        sub = main[(main["drug"] == g.drug) & main["herbs"].str.split("; ").map(lambda h: g.herb in h)]
        if sub.empty:
            continue
        ch = []
        for r in sub.itertuples():
            i = r.herbs.split("; ").index(g.herb)
            ch.append(float(r.w.split("; ")[i]) / float(r.w0.split("; ")[i]) - 1)
        ch = np.array(ch)
        rows.append({"herb": g.herb, "drug": g.drug, "gold_label": g.label, "mechanism": g.mechanism, "n_formulations": len(ch),
                     "frac_reduced": float((ch < -1e-6).mean()), "frac_increased": float((ch > 1e-6).mean()), "median_change": float(np.median(ch))})
    return pd.DataFrame(rows, columns=["herb", "drug", "gold_label", "mechanism", "n_formulations", "frac_reduced", "frac_increased", "median_change"])


def phase5_facts(forms, main, sweep, rand, direction) -> dict:
    """Flat, JSON-safe numbers from a Phase 5 run. The paper drafts are rendered from these, so no number is ever retyped by hand."""
    nan_to_none = lambda v: None if (isinstance(v, float) and v != v) else v
    r = main["rel_reduction"]
    e = main["eval_rel_reduction"] if "eval_rel_reduction" in main else pd.Series(dtype=float)
    pos = direction[(direction["gold_label"] == 1) & (direction["mechanism"] == "PK")] if direction is not None and len(direction) else pd.DataFrame(columns=["frac_reduced"])
    neg = direction[direction["gold_label"] == 0] if direction is not None and len(direction) else pd.DataFrame(columns=["frac_reduced"])
    f = {"forms": len(forms), "forms_scoped": int((forms["in_scope"].map(len) >= 1).sum()),
         "forms_opt": int(forms["in_scope"].map(lambda x: len(set(x)) >= 2).sum()), "scenarios": len(main),
         "median_red": float(r.median()), "mean_red": float(r.mean()), "p90_red": float(r.quantile(0.9)), "frac_improve_1pct": float((r > 0.01).mean()),
         "min_coverage": float(main["min_coverage_ratio"].min()), "n_flags": int((main["n_flags"] > 0).sum()),
         "transfer_median": float(e.median()) if len(e) else None, "transfer_improve_frac": float((e > 0).mean()) if len(e) else None,
         "spearman": nan_to_none(float(r.corr(e, method="spearman"))) if len(e) > 1 else None,
         "sweep_min_red": float(sweep["median_rel_reduction"].min()), "sweep_max_red": float(sweep["median_rel_reduction"].max()),
         "random_median": float(rand["random_median"]), "equal_median": float(rand["equal_median"]),
         "dir_pos_mean": float(pos["frac_reduced"].mean()) if len(pos) else None, "dir_pos_min": float(pos["frac_reduced"].min()) if len(pos) else None,
         "dir_pos_max": float(pos["frac_reduced"].max()) if len(pos) else None, "n_dir_pos": int(len(pos)),
         "dir_neg_mean": float(neg["frac_reduced"].mean()) if len(neg) else None, "n_dir_neg": int(len(neg))}
    return {k: nan_to_none(v) for k, v in f.items()}


def make_phase5_report(forms, main, sweep, rand, cases, n_formulations_total, tau=0.8, lo_frac=0.5, hi_mult=2.0, direction=None) -> str:
    """Markdown report. Every number is computed from the inputs."""
    n_any = int((forms["in_scope"].map(len) >= 1).sum())
    n_two = int(forms["in_scope"].map(lambda x: len(set(x)) >= 2).sum())
    frac = (main.drop_duplicates("formulation_id").eval("n_in_scope / n_ingredients")) if len(main) else pd.Series(dtype=float)
    L = ["# Phase 5 results", "", "Generated by `python -m ayurveda_kg.phase5` (report step). Do not edit by hand.", "",
         "**What this is.** An optimiser that re-weights the in-scope herbs of a real IMPPAT formulation to lower the predicted interaction risk "
         "against one patient drug, while keeping every therapeutic use above a coverage floor and every share within bounds of its baseline. "
         "The risk is a **risk proxy** (Phase 4 Random Forest, out-of-fold, mean over a herb's compounds). IMPPAT gives no proportions, so the "
         "baseline is an assumption (equal parts). Output is hypothesis-generating and **not a dosing recommendation**.", "",
         "## Coverage of the formulation set", "",
         f"- {n_formulations_total} formulations were parsed from IMPPAT; {n_any} contain at least one of the 20 scoped herbs and "
         f"{n_two} of {n_formulations_total} contain two or more, which is the minimum for any re-weighting.",
         f"- Among the optimisable formulations, the scoped herbs make up a median of {frac.median():.0%} of the ingredients (the rest are fixed and have "
         "no risk estimate)." if len(frac) else "- No optimisable formulations.", ""]
    L += ["## Optimiser behaviour", ""]
    if len(main):
        r = main["rel_reduction"]
        L += [f"- {len(main)} single-drug scenarios over {main['formulation_id'].nunique()} formulations; "
              f"{int((main['status'] == 'optimal').sum())} solved to optimality.",
              f"- Relative risk reduction vs baseline: median {r.median():.1%}, mean {r.mean():.1%}, 90th percentile {r.quantile(.9):.1%}; "
              f"{(r > 0.01).mean():.0%} of scenarios improve by more than 1%.",
              f"- Main setting: coverage floor tau = {tau}, each herb's share kept within {lo_frac}x to {hi_mult}x of its baseline. Lowest coverage ratio "
              f"actually kept: {main['min_coverage_ratio'].min():.3f} (must be >= {tau}).",
              f"- Scenarios where some herb was cut below half its baseline share: {int((main['n_flags'] > 0).sum())}."]
    L += ["", "## Does a suggestion transfer to an independent risk model?", ""]
    if "eval_rel_reduction" in main:
        e = main["eval_rel_reduction"]
        rho = main["rel_reduction"].corr(e, method="spearman")
        L += ["Suggestions are optimised with the Random Forest risk and then scored with the Phase 3 silver herb-drug score (a different, "
              "mechanistic estimate).", "",
              f"- Median reduction under the independent model: {e.median():.1%} (mean {e.mean():.1%}); "
              f"{(e > 0).mean():.0%} of scenarios also improve under it.",
              f"- Spearman correlation between the two reductions: {rho:.2f}."]
    L += ["", "## Sensitivity to constraints", "", "| coverage floor tau | min share (x baseline) | max share (x baseline) | scenarios | median reduction | mean reduction | median min coverage | scenarios with a herb removed (flagged) |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r.tau} | {r.lo_frac} | {r.hi_mult} | {r.n_scenarios} | {r.median_rel_reduction:.1%} | {r.mean_rel_reduction:.1%} | {r.median_min_coverage:.3f} | {r.frac_with_removal:.1%} |"
          for r in sweep.itertuples()]
    L += ["", "## Sensitivity to the equal-parts assumption", "",
          f"Median reduction with equal-parts baselines: {rand['equal_median']:.1%}; with {rand['n']} random Dirichlet baselines per formulation: "
          f"{rand['random_median']:.1%}. A large gap would mean the conclusion depends on the unknown proportions.", "",
          "## Literature-linked case studies", ""]
    if len(cases):
        L += ["Formulations containing a herb with a published human interaction (Phase 3 gold set). A good suggestion lowers that herb's share.", "",
              "| formulation | herb | drug | share before | share after | in-scope herbs | scenario reduction | gold note |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r.formulation} | {r.herb} | {r.drug} | {r.share_before:.3f} | {r.share_after:.3f} | {r.n_in_scope} | {r.rel_reduction:.1%} | {r.note} |"
              for r in cases.itertuples()]
    else:
        L.append("No matching formulations.")
    if direction is not None and len(direction):
        L += ["", "## Direction check against the published studies", "",
              "For every herb-drug pair with a published human study (Phase 3 gold set), across all formulations that contain the herb: how often does the optimiser lower "
              "that herb's share? Lowering is the right direction where the study found a published interaction and the wrong direction where it found no interaction.", "",
              "| herb | drug | gold | mechanism | formulations | share reduced | share increased | median change |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r.herb} | {r.drug} | {'interaction' if r.gold_label == 1 else 'no interaction'} | {r.mechanism} | {r.n_formulations} | {r.frac_reduced:.0%} | "
              f"{r.frac_increased:.0%} | {r.median_change:+.0%} |" for r in direction.itertuples()]
        pos = direction[(direction["gold_label"] == 1) & (direction["mechanism"] == "PK")]
        neg = direction[direction["gold_label"] == 0]
        if len(pos):
            L += ["", f"Computed: for the {len(pos)} pairs with a published interaction (pharmacokinetic), the herb's share is reduced in {pos['frac_reduced'].mean():.0%} of scenarios on average "
                      f"(per-pair range {pos['frac_reduced'].min():.0%} to {pos['frac_reduced'].max():.0%})."]
        if len(neg):
            L += [f"For the {len(neg)} pairs with no interaction in the published trial, the herb is nevertheless reduced in {neg['frac_reduced'].mean():.0%} of scenarios on average (a false alarm inherited from the risk proxy)."]
    L += ["", "## Limitations", "",
          "- The risk is a mechanistic-hypothesis proxy with known false alarms (e.g. ginger-warfarin, a human-trial negative, scores high), so the "
          "optimiser can lower the share of a herb that is actually safe.",
          "- Shares are a risk proxy: IMPPAT has no proportions or abundances, and compounds are weighted equally within a herb.",
          "- Only the 20 scoped herbs can be re-weighted; other ingredients are fixed and unscored. Coverage uses IMPPAT therapeutic-use labels "
          "(binary, literature-derived), not efficacy.",
          "- Pharmacodynamic interactions (e.g. licorice with diuretics) are invisible to this CYP-based risk model."]
    return "\n".join(L) + "\n"


CASE_PICKS = [
    ("Piper nigrum", "phenytoin", "gold: interaction (human PK, Pattanaik 2006)"),
    ("Piper nigrum", "carbamazepine", "gold: interaction (human PK, Pattanaik 2009)"),
    ("Piper nigrum", "propranolol", "gold: interaction (human PK, Bano 1991)"),
    ("Piper nigrum", "theophylline", "gold: interaction (human PK, Bano 1991)"),
    ("Commiphora wightii", "diltiazem", "gold: interaction (human PK, Dalvi 1994)"),
    ("Zingiber officinale", "warfarin", "gold: NO interaction (RCT, Jiang 2005); a model false alarm"),
    ("Allium sativum", "warfarin", "gold: NO interaction (RCT, Mohammed Abdul 2008); a model false alarm"),
]
SWEEP = [(0.95, 0.5, 1.5), (0.8, 0.5, 2.0), (0.6, 0.25, 3.0), (0.5, 0.0, 4.0)]


def run_all(report_path="docs/phase5_results.md", n_random=5):
    risk_path = OUT / "herb_drug_risk.parquet"
    risk = pd.read_parquet(risk_path) if risk_path.exists() else build_risk()
    uses, forms = load_uses(), parse_formulation_records()
    drugs = sorted(risk["drug"].unique())
    main = run_scenarios(forms, risk, uses, drugs, risk_col="risk_rf", eval_col="risk_silver", tau=0.8, lo_frac=0.5, hi_mult=2.0)
    sweep = sweep_constraints(forms, risk, uses, drugs, SWEEP)
    rnd = pd.concat([run_scenarios(forms, risk, uses, drugs, tau=0.8, baseline_seed=s) for s in range(n_random)])
    rand = {"equal_median": float(main["rel_reduction"].median()), "random_median": float(rnd["rel_reduction"].median()), "n": n_random}
    cases = select_cases(main, CASE_PICKS)
    OUT.mkdir(parents=True, exist_ok=True)
    main.to_parquet(OUT / "scenarios.parquet", index=False)
    direction = direction_check(main, curated.load_gold())
    import json
    (OUT / "facts.json").write_text(json.dumps(phase5_facts(forms, main, sweep, rand, direction)), encoding="utf-8")
    Path(report_path).write_text(make_phase5_report(forms, main, sweep, rand, cases, len(forms), direction=direction), encoding="utf-8")
    return forms, main, sweep, rand, cases


if __name__ == "__main__":
    forms, main, sweep, rand, cases = run_all()
    print(f"{len(forms)} formulations, {len(main)} scenarios; wrote docs/phase5_results.md")
