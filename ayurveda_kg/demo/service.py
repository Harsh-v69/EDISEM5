"""Demo logic as pure functions (the Streamlit app only renders their output). Local use only: it reads IMPPAT-derived tables whose licence
(CC BY-NC-ND) forbids redistribution, so the demo must not be hosted publicly."""
from dataclasses import dataclass

import pandas as pd

CAVEAT = ("Research risk score from predicted CYP inhibition by the herb's compounds versus the drug's known CYP substrate status. "
          "It is a mechanistic hypothesis with known false alarms (for example ginger with warfarin), not medical advice.")
SUGGEST_CAVEAT = ("Hypothesis-generating risk proxy, not a dosing recommendation. IMPPAT gives no proportions, so the baseline assumes equal parts across "
                  "all ingredients; only the scoped herbs are re-weighted and other ingredients stay fixed and unscored.")


@dataclass
class DemoData:
    nodes: dict
    edges: dict
    labels: pd.DataFrame          # silver labels: compound, drug, label, enzymes (ids with prefixes)
    cd_risk: pd.DataFrame         # out-of-fold compound-drug probability: compound, drug, p
    hd_risk: pd.DataFrame         # herb-drug: herb, drug, risk_rf, risk_silver, n_compounds
    gold: pd.DataFrame
    scope: dict
    uses: dict
    forms: pd.DataFrame

    @classmethod
    def load(cls):
        from ayurveda_kg import curated, phase5
        from ayurveda_kg.kg import load_tables
        from ayurveda_kg.scope import load_scope
        nodes, edges = load_tables("data/processed/kg")
        scope = load_scope()
        return cls(nodes, edges, pd.read_parquet("data/processed/labels/silver.parquet"),
                   pd.read_parquet("data/processed/phase5/compound_drug_risk.parquet"),
                   pd.read_parquet("data/processed/phase5/herb_drug_risk.parquet"), curated.load_gold(), scope,
                   phase5.load_uses(scope), phase5.load_formulation_records())


def list_herbs(d: DemoData) -> list[tuple[str, str]]:
    return sorted((h["imppat_name"], h.get("common") or h["imppat_name"]) for h in d.scope["herbs"])


def list_drugs(d: DemoData) -> list[str]:
    return sorted(x["name"] for x in d.scope["drugs"])


def risk_ranking(d: DemoData, drug: str, top=10) -> pd.DataFrame:
    r = d.hd_risk[d.hd_risk["drug"] == drug].sort_values(["risk_rf", "herb"], ascending=[False, True])
    return r[["herb", "risk_rf", "risk_silver", "n_compounds"]].head(top).reset_index(drop=True)


def _source(edges: pd.DataFrame, src: str, dst: str) -> str:
    if edges is None or not len(edges):
        return "n/a"
    m = edges[(edges["src"] == src) & (edges["dst"] == dst)]
    return str(m["source"].iloc[0]) if len(m) and "source" in m else "n/a"


def explain_pair(d: DemoData, herb: str, drug: str, top_n=5) -> dict:
    """Score, its rank, the compounds driving it, the CYP path with sources for each, and any published human study for the pair."""
    if herb not in {h for h, _ in list_herbs(d)}:
        return {"error": f"Unknown herb: {herb}"}
    if drug not in list_drugs(d):
        return {"error": f"Unknown drug: {drug}"}
    row = d.hd_risk[(d.hd_risk["herb"] == herb) & (d.hd_risk["drug"] == drug)]
    if row.empty:
        return {"error": f"No risk score for {herb} with {drug}."}
    row = row.iloc[0]
    same_drug = d.hd_risk[d.hd_risk["drug"] == drug]["risk_rf"]
    scores = {"risk_rf": float(row["risk_rf"]), "risk_silver": float(row["risk_silver"]), "n_compounds": int(row["n_compounds"]),
              "percentile_for_drug": float((same_drug <= row["risk_rf"]).mean()),
              "percentile_overall": float((d.hd_risk["risk_rf"] <= row["risk_rf"]).mean())}
    cids = d.edges["contains"].loc[d.edges["contains"]["src"] == f"herb:{herb}", "dst"]
    names = dict(zip(d.nodes["Compound"]["id"], d.nodes["Compound"]["name"]))
    drug_id = f"drug:{drug}"
    r = d.cd_risk[(d.cd_risk["drug"] == drug_id) & d.cd_risk["compound"].isin(set(cids))].sort_values(["p", "compound"], ascending=[False, True]).head(top_n)
    lab = d.labels[d.labels["drug"] == drug_id].set_index("compound")
    comps = []
    for c, p in zip(r["compound"], r["p"]):
        silver = int(lab.loc[c, "label"]) if c in lab.index else None
        enzymes = [e for e in str(lab.loc[c, "enzymes"]).split(";") if e] if c in lab.index else []
        paths, steps = [], []
        if silver == 1:
            for x in enzymes:
                steps.append({"herb": herb, "compound": names.get(c, c), "enzyme": x, "drug": drug,
                              "inhibitor_source": _source(d.edges.get("predicted_cyp_inhibitor"), c, f"gene:{x}"),
                              "substrate_source": _source(d.edges.get("substrate_of"), drug_id, f"gene:{x}")})
                paths.append(f"{herb} → {names.get(c, c)} → predicted inhibitor of {x} [{_source(d.edges.get('predicted_cyp_inhibitor'), c, f'gene:{x}')}]"
                             f" ← known substrate of {x} [{_source(d.edges.get('substrate_of'), drug_id, f'gene:{x}')}] ← {drug}")
        comps.append({"name": names.get(c, c), "p": float(p), "silver_label": silver, "enzymes": enzymes, "paths": paths, "path_steps": steps})
    g = d.gold[(d.gold["herb"] == herb) & (d.gold["drug"] == drug)]
    gold = [{k: r_[k] for k in ("pmid", "label", "mechanism", "evidence", "confidence", "citation", "finding")} for _, r_ in g.iterrows()]
    return {"herb": herb, "drug": drug, "scores": scores, "compounds": comps, "gold": gold, "caveat": CAVEAT}


def formulation_choices(d: DemoData) -> pd.DataFrame:
    f = d.forms.copy()
    f["n_in_scope"] = f["in_scope"].map(lambda x: len(set(x)))
    f = f[f["n_in_scope"] >= 1]
    f["optimisable"] = f["n_in_scope"] >= 2
    return f[["id", "name", "kind", "n_ingredients", "n_in_scope", "optimisable"]].sort_values("name").reset_index(drop=True)


def suggest(d: DemoData, formulation_id: str, drugs: list[str], tau=0.8, lo_frac=0.5, hi_mult=2.0, objective="sum") -> dict:
    from ayurveda_kg.compose import optimise
    from ayurveda_kg.phase5 import build_problem
    row = d.forms[d.forms["id"] == formulation_id]
    if row.empty:
        return {"error": f"Unknown formulation: {formulation_id}"}
    if not drugs:
        return {"error": "Choose at least one drug."}
    if any(x not in list_drugs(d) for x in drugs):
        return {"error": "Unknown drug in selection."}
    f = row.iloc[0]
    herbs = list(dict.fromkeys(f["in_scope"]))
    base = {"formulation": f["name"], "id": formulation_id, "n_ingredients": int(f["n_ingredients"]), "caveat": SUGGEST_CAVEAT}
    if len(herbs) < 2:
        return {**base, "optimisable": False, "message": "This formulation has only one scoped herb, so there is nothing to re-weight."}
    p = build_problem(list(f["in_scope"]), int(f["n_ingredients"]), d.hd_risk, list(drugs), d.uses, "risk_rf")
    if pd.isna(p["risk"]).any():
        return {**base, "error": "No risk estimate for one of the herb-drug combinations."}
    r = optimise(p["risk"], p["w0"], A=p["A"], tau=tau, lo_frac=lo_frac, hi_mult=hi_mult, objective=objective)
    table = pd.DataFrame({"herb": p["herbs"], "baseline_share": r["w0"], "suggested_share": r["w"]})
    table["change"] = table["suggested_share"] / table["baseline_share"] - 1
    return {**base, "optimisable": True, "table": table, "risk_before": dict(zip(drugs, map(float, r["baseline_per_drug_risk"]))),
            "risk_after": dict(zip(drugs, map(float, r["per_drug_risk"]))), "objective_before": r["baseline_objective"],
            "objective_after": r["objective"], "min_coverage_ratio": r["min_coverage_ratio"], "flags": r["flags"],
            "fixed_ingredients": int(f["n_ingredients"]) - len(f["in_scope"]), "status": r["status"]}


def ask(retriever, question: str, client=None) -> dict:
    """Retrieve cited context and, if a model client is given, answer. Without a client it degrades to retrieval only."""
    from ayurveda_kg.rag.generate import answer
    items = retriever.retrieve(question)
    if client is None:
        return {"items": items, "answer": None, "note": "Retrieval only: no language model is connected."}
    return {"items": items, "answer": answer(question, items, client), "note": ""}


def build_retriever():
    """Heavy: loads the embedding model, index and graphs (a few seconds). Needs the processed RAG data to exist locally."""
    from ayurveda_kg import phase6
    sys_ = phase6.load_system()
    return phase6.retriever(sys_, "full")
