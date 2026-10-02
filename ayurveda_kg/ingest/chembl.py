"""ChEMBL (open) access: resolve scoped drug names to ChEMBL ids and download the full metabolism (enzyme) table."""
import json
import urllib.parse
from pathlib import Path

from ayurveda_kg.ingest.fetch import fetch

API = "https://www.ebi.ac.uk/chembl/api/data/"
LICENCE = "ChEMBL CC BY-SA 3.0"


def pick_molecule(search_json, name):
    """ChEMBL id of the parent molecule (salts/duplicate entries map to their parent) whose pref_name or a synonym equals `name` (case-insensitive), else None."""
    want = name.strip().lower()
    hits = []
    for m in search_json.get("molecules", []):
        names = {(m.get("pref_name") or "").lower()} | {s["molecule_synonym"].lower() for s in m.get("molecule_synonyms") or []}
        if want in names:
            parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id")
            hits.append((parent == m["molecule_chembl_id"], m.get("max_phase") or 0, parent or m["molecule_chembl_id"]))
    return max(hits)[2] if hits else None


KEEP = ("drug_chembl_id", "substrate_chembl_id", "enzyme_name", "enzyme_chembl_id", "substrate_name", "metabolite_name")


def parse_metabolism(page) -> list[dict]:
    return [{k: m.get(k) for k in KEEP} for m in page.get("metabolisms", [])]


def resolve_drug_ids(names, raw_dir, manifest_path="data/manifest.json", delay=1.0) -> dict:
    out = {}
    for n in names:
        p = fetch(f"{API}molecule/search.json?q={urllib.parse.quote(n)}&limit=20", Path(raw_dir) / "molecule_search" / f"{n}.json",
                  name=f"chembl_search_{n}", licence=LICENCE, manifest_path=manifest_path, delay=delay)
        out[n] = pick_molecule(json.loads(p.read_text(encoding="utf-8")), n)
    return out


def fetch_metabolism_all(raw_dir, manifest_path="data/manifest.json", delay=1.0, page=1000) -> list[dict]:
    rows, offset = [], 0
    while True:
        p = fetch(f"{API}metabolism.json?limit={page}&offset={offset}", Path(raw_dir) / f"metabolism_{offset}.json",
                  name=f"chembl_metabolism_{offset}", licence=LICENCE, manifest_path=manifest_path, delay=delay)
        d = json.loads(p.read_text(encoding="utf-8"))
        rows += parse_metabolism(d)
        offset += page
        if offset >= d["page_meta"]["total_count"]:
            return rows
