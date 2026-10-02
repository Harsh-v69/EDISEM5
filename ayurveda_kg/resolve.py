"""Entity resolution: one canonical node per herb, compound (InChIKey) and drug. Nothing is silently dropped."""
import re

import pandas as pd


def herb_table(scope) -> pd.DataFrame:
    return pd.DataFrame([{"id": f"herb:{h['imppat_name']}", "name": h["imppat_name"],
                          "aliases": h.get("aliases", []), "common": h.get("common", "")} for h in scope["herbs"]])


def resolve_compounds(details):
    """details: parse_detail_page() dicts. Returns (compound table, phy_id -> compound id).
    Key = full InChIKey (stereo-aware). No key -> fall back to the IMPPAT id so the compound is kept."""
    phy2cid, groups = {}, {}
    for d in details:
        cid = f"cpd:{d['inchikey']}" if d["inchikey"] else f"cpd:{d['phy_id']}"
        phy2cid[d["phy_id"]] = cid
        groups.setdefault(cid, []).append(d)
    rows = []
    for cid, ds in groups.items():
        first = ds[0]
        rows.append({"id": cid, "name": first["name"], "inchikey": first["inchikey"], "smiles": first["smiles"],
                     "pubchem_cid": first["cid"], "chembl_id": first["chembl_id"],
                     "phy_ids": [d["phy_id"] for d in ds], "n_merged": len(ds),
                     "synonyms": sorted({s for d in ds for s in d["synonyms"]})})
    return pd.DataFrame(rows), phy2cid


def resolve_drugs(scope, chembl_ids, aliases=None):
    """Returns (drug table, names with no ChEMBL id). Unresolved drugs stay in the table.
    aliases: {scope name: [other names, e.g. ChEMBL synonyms]} used to match DGIdb/DDInter spellings."""
    aliases = aliases or {}
    rows = [{"id": f"drug:{d['name']}", "name": d["name"], "cls": d["cls"], "chembl_id": chembl_ids.get(d["name"]) or "",
             "aliases": sorted({a for a in aliases.get(d["name"], []) if a and a.lower() != d["name"].lower()})}
            for d in scope["drugs"]]
    return pd.DataFrame(rows), [d["name"] for d in scope["drugs"] if not chembl_ids.get(d["name"])]


def alias_map(drugs) -> dict:
    """UPPERCASE name/alias -> drug id. An alias claimed by two different drugs is dropped (it would attach evidence to the wrong drug)."""
    seen, bad = {}, set()
    has_alias = "aliases" in drugs.columns
    for i, n, al in zip(drugs["id"], drugs["name"], drugs["aliases"] if has_alias else [[]] * len(drugs)):
        for a in {n, *al}:
            k = a.strip().upper()
            if seen.setdefault(k, i) != i:
                bad.add(k)
    return {k: v for k, v in seen.items() if k not in bad}


_GENE = re.compile(r"[A-Z][A-Z0-9]*\d[A-Z0-9]*")


def is_gene_symbol(name) -> bool:
    """True for symbols like CYP3A4 / UGT1A9 / NAT2; False for enzyme families like ESTERASES or UGT."""
    return bool(name) and _GENE.fullmatch(name) is not None
