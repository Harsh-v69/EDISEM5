"""Phase 2 pipeline: raw files -> resolved tables -> validated KG on disk (data/processed/kg)."""
import glob
import json
import re
from pathlib import Path

import pandas as pd

from ayurveda_kg import curated, kg, report, resolve, validate
from ayurveda_kg.ingest import chembl, imppat
from ayurveda_kg.resolve import alias_map, is_gene_symbol
from ayurveda_kg.scope import load_scope

LEVEL_RANK = {"Unknown": 0, "Minor": 1, "Moderate": 2, "Major": 3}


# ---------- pure assembly functions (unit-tested) ----------

def herb_edges(plant_rows, herb_ids, phy2cid) -> pd.DataFrame:
    agg = {}
    for r in plant_rows:
        cid = phy2cid.get(r["phy_id"])
        if cid is None or r["plant"] not in herb_ids:
            continue  # compound page not crawled yet: reported by the pipeline, never invented
        a = agg.setdefault((herb_ids[r["plant"]], cid), {"parts": set(), "refs": set()})
        if r["part"]:
            a["parts"].add(r["part"])
        a["refs"].add(r["reference"])
    return pd.DataFrame([{"src": s, "dst": d, "parts": "; ".join(sorted(a["parts"])), "n_refs": len(a["refs"])}
                         for (s, d), a in agg.items()], columns=["src", "dst", "parts", "n_refs"])


def compound_target_edges(targets_by_phy, phy2cid):
    mod, cyp, pgp = {}, set(), set()
    for phy, t in targets_by_phy.items():
        cid = phy2cid.get(phy)
        if cid is None:
            continue
        for x in t["targets"]:
            mod.setdefault((cid, f"gene:{x['gene']}"), set()).add(x["source"])
        for k, v in t["adme"].items():
            m = re.fullmatch(r"(CYP\w+) inhibitor", k)
            if m and v == "Yes":
                cyp.add((cid, f"gene:{m.group(1)}"))
            if k == "P-glycoprotein substrate" and v == "Yes":
                pgp.add((cid, "gene:ABCB1"))
    mod_df = pd.DataFrame([{"src": s, "dst": d, "sources": "; ".join(sorted(v))} for (s, d), v in sorted(mod.items())],
                          columns=["src", "dst", "sources"])

    def plain(pairs):
        return pd.DataFrame([{"src": s, "dst": d, "source": "SwissADME (via IMPPAT)"} for s, d in sorted(pairs)],
                            columns=["src", "dst", "source"])

    return mod_df, plain(cyp), plain(pgp)


def target_nodes(edge_frames) -> pd.DataFrame:
    genes = sorted({d for df in edge_frames for d in df["dst"] if d.startswith("gene:")})
    return pd.DataFrame([{"id": g, "symbol": g[5:], "is_cyp": g[5:].startswith("CYP")} for g in genes],
                        columns=["id", "symbol", "is_cyp"])


def dgidb_edges(df, drugs) -> pd.DataFrame:
    by_chembl = {c.upper(): i for i, c in zip(drugs["id"], drugs["chembl_id"]) if c}
    by_name = alias_map(drugs)
    df = df[df["gene_name"].notna()]
    concept = df["drug_concept_id"].fillna("").str.replace("chembl:", "", regex=False).str.upper()
    drug_id = concept.map(by_chembl).fillna(df["drug_name"].fillna("").str.upper().map(by_name))
    df = df.assign(drug=drug_id).dropna(subset=["drug"])
    rows = []
    for (d, g), grp in df.groupby(["drug", "gene_name"]):
        scores = pd.to_numeric(grp["interaction_score"], errors="coerce")
        rows.append({"src": d, "dst": f"gene:{g}",
                     "sources": "; ".join(sorted(grp["interaction_source_db_name"].dropna().unique())),
                     "interaction_types": "; ".join(sorted(grp["interaction_type"].dropna().unique())),
                     "score": float(scores.max()) if scores.notna().any() else float("nan")})
    return pd.DataFrame(rows, columns=["src", "dst", "sources", "interaction_types", "score"])


def metabolism_edges(rows, id_to_parent, drugs) -> pd.DataFrame:
    by_chembl = {c: i for i, c in zip(drugs["id"], drugs["chembl_id"]) if c}
    pairs = set()
    for r in rows:
        drug = by_chembl.get(id_to_parent.get(r["substrate_chembl_id"]))
        enzyme = (r["enzyme_name"] or "").strip().upper()
        if drug and is_gene_symbol(enzyme):
            pairs.add((drug, f"gene:{enzyme}"))
    return pd.DataFrame([{"src": s, "dst": d, "source": "ChEMBL metabolism"} for s, d in sorted(pairs)],
                        columns=["src", "dst", "source"])


def ddi_edges(df, drugs) -> pd.DataFrame:
    by_name = alias_map(drugs)
    best = {}
    for a, b, lvl in zip(df["Drug_A"], df["Drug_B"], df["Level"]):
        ia, ib = by_name.get(str(a).strip().upper()), by_name.get(str(b).strip().upper())
        if ia and ib and ia != ib:
            k = tuple(sorted((ia, ib)))
            if LEVEL_RANK.get(lvl, 0) >= LEVEL_RANK.get(best.get(k, "Unknown"), 0):
                best[k] = lvl
    return pd.DataFrame([{"src": s, "dst": d, "level": lv, "source": "DDInter 2.0"} for (s, d), lv in sorted(best.items())],
                        columns=["src", "dst", "level", "source"])


def with_adme_flag(compounds, targets_by_phy) -> pd.DataFrame:
    """has_adme = at least one merged IMPPAT page carried SwissADME predictions (so a missing 'Yes' really means 'No')."""
    flag = [any(targets_by_phy.get(p, {}).get("adme") for p in phys) for phys in compounds["phy_ids"]]
    return compounds.assign(has_adme=flag)


def tdc_edges(frames, drugs):
    """frames: {gene symbol: TDC DataFrame[Drug_ID, Y]}. Returns (substrate_of, non_substrate_of) edge tables."""
    by_name = alias_map(drugs)
    pos, neg = set(), set()
    for gene, df in frames.items():
        for name, y in zip(df["Drug_ID"], df["Y"]):
            d = by_name.get(str(name).strip().upper())
            if d:
                (pos if int(y) == 1 else neg).add((d, f"gene:{gene}"))
    mk = lambda pairs: pd.DataFrame([{"src": s, "dst": d, "source": "TDC Carbon-Mangels"} for s, d in sorted(pairs)],
                                    columns=["src", "dst", "source"])
    return mk(pos), mk(neg)


def merge_edge_sources(*frames) -> pd.DataFrame:
    """Union edge tables; the same (src, dst) from several sources becomes one edge listing all sources."""
    df = pd.concat(frames, ignore_index=True)
    return (df.groupby(["src", "dst"], sort=True)["source"].agg(lambda x: "; ".join(sorted(set(x)))).reset_index())


# ---------- loaders (touch disk) ----------

def _read(p):
    return Path(p).read_text(encoding="utf-8", errors="ignore")


def load_imppat(raw="data/raw/imppat"):
    plant_rows = []
    for p in sorted(glob.glob(f"{raw}/plants/*.html")):
        plant_rows += imppat.parse_plant_page(_read(p))
    details, targets = [], {}
    for p in sorted(glob.glob(f"{raw}/compounds/*.detail.html")):
        d = imppat.parse_detail_page(_read(p))
        if d["phy_id"]:
            details.append(d)
            tp = Path(p.replace(".detail.html", ".targets.html"))
            if tp.exists():
                targets[d["phy_id"]] = imppat.parse_targets_page(_read(tp))
    return plant_rows, details, targets


def load_chembl(raw="data/raw/chembl"):
    ids = json.loads(_read(f"{raw}/drug_ids.json"))
    parents = set(filter(None, ids.values()))
    id_to_parent, names = {}, {}
    for p in glob.glob(f"{raw}/molecule_search/*.json"):
        for m in json.loads(_read(p)).get("molecules", []):
            par = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id")
            if par in parents:
                id_to_parent[m["molecule_chembl_id"]] = par
                names.setdefault(par, set()).update([m.get("pref_name") or ""] + [x["molecule_synonym"] for x in m.get("molecule_synonyms") or []])
    aliases = {n: sorted(names.get(pid, [])) for n, pid in ids.items() if pid}
    rows = []
    for p in sorted(glob.glob(f"{raw}/metabolism_*.json")):
        rows += chembl.parse_metabolism(json.loads(_read(p)))
    return ids, id_to_parent, rows, aliases


def load_tdc(raw="data/raw/tdc_cyp"):
    return {Path(p).name.split("_")[0]: pd.read_csv(p, sep="\t") for p in sorted(glob.glob(f"{raw}/*.tab"))}


def build_all(out_dir="data/processed/kg", scope_path="config/scope.yaml"):
    scope = load_scope(scope_path)
    plant_rows, details, targets = load_imppat()
    herbs = resolve.herb_table(scope)
    compounds, phy2cid = resolve.resolve_compounds(details)
    compounds = with_adme_flag(compounds, targets)
    ids, id_to_parent, met_rows, aliases = load_chembl()
    drugs, unresolved = resolve.resolve_drugs(scope, ids, aliases)

    contains = herb_edges(plant_rows, dict(zip(herbs["name"], herbs["id"])), phy2cid)
    modulates, cyp_inh, pgp = compound_target_edges(targets, phy2cid)
    dg = pd.read_csv("data/raw/dgidb/interactions.tsv", sep="\t", dtype=str, na_values=["NULL"])
    dd = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(glob.glob("data/raw/ddinter/*.csv"))])

    tdc_pos, tdc_neg = tdc_edges(load_tdc(), drugs)
    edges = {"contains": contains, "modulates": modulates, "predicted_cyp_inhibitor": cyp_inh,
             "predicted_pgp_substrate": pgp, "targets": dgidb_edges(dg, drugs),
             "substrate_of": merge_edge_sources(metabolism_edges(met_rows, id_to_parent, drugs), tdc_pos,
                                                curated.load_substrate_supplement("data/curated/substrate_supplement.csv", drugs)),
             "non_substrate_of": tdc_neg, "ddi": ddi_edges(dd, drugs)}
    nodes = {"Herb": herbs, "Compound": compounds, "Drug": drugs, "Target": target_nodes(list(edges.values()))}
    problems = validate.validate_all(nodes, edges, scope)
    kg.save_tables(nodes, edges, out_dir)
    meta = {"compounds_crawled": len(details), "compounds_expected": len({r["phy_id"] for r in plant_rows}),
            "compound_merges": int((compounds["n_merged"] > 1).sum()) if len(compounds) else 0,
            "drugs_without_chembl": unresolved, "problems": problems}
    Path("data/processed/kg_meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "problems"}), encoding="utf-8")
    Path("docs/kg_report.md").write_text(report.make_report(nodes, edges, meta), encoding="utf-8")
    return {"nodes": {k: len(v) for k, v in nodes.items()}, "edges": {k: len(v) for k, v in edges.items()}, **meta}


if __name__ == "__main__":
    r = build_all()
    print(json.dumps({k: v for k, v in r.items() if k != "problems"}, indent=1))
    print(f"{len(r['problems'])} validation problems")
    for p in r["problems"][:40]:
        print(" -", p)
