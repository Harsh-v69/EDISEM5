"""Probe every planned data source so the plan rests on evidence, not assumptions."""
from datetime import datetime, timezone
from pathlib import Path

import requests

from ayurveda_kg.ingest.fetch import UA

SOURCES = [
    {"name": "IMPPAT 2.0", "url": "https://cb.imsc.res.in/imppat/", "needs": "herb, compound, use, formulation"},
    {"name": "PubChem PUG REST", "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/curcumin/property/InChIKey/JSON", "needs": "compound identifiers"},
    {"name": "ChEMBL API", "url": "https://www.ebi.ac.uk/chembl/api/data/status.json", "needs": "compound-target bioactivity"},
    {"name": "DGIdb downloads", "url": "https://dgidb.org/downloads", "needs": "drug-gene interactions"},
    {"name": "DDInter 2.0", "url": "https://ddinter2.scbdd.com/", "needs": "drug-drug interactions, drug info"},
    {"name": "DrugBank academic", "url": "https://go.drugbank.com/academic_research/", "needs": "optional: drug-target, substrates"},
    {"name": "AyurParam (HF search)", "url": "https://huggingface.co/api/models?search=ayurparam", "needs": "local Ayurveda LLM"},
]


def probe(url, session=None) -> dict:
    try:
        r = (session or requests).get(url, headers={"User-Agent": UA}, timeout=30, stream=True)
        try:
            code, ctype = r.status_code, r.headers.get("Content-Type", "")
        finally:
            r.close()
        return {"status": "OK" if code < 400 else f"HTTP {code}", "http": code, "type": ctype, "note": ""}
    except Exception as e:  # network problems are a result, not a crash
        return {"status": "ERROR", "http": None, "type": "", "note": f"{type(e).__name__}: {e}"}


def write_report(results, path):
    rows = [
        f"| {r['name']} | {r['url']} | {r['needs']} | {r['status']} | {r['http']} | {r['type']} | {r['note']} |"
        for r in results
    ]
    head = (
        "# Data access audit\n\n"
        f"Probed {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        "OK means the URL answered, not that bulk download or licensing is settled. "
        "See the manual notes section for what that means.\n\n"
        "| Source | URL | Needed for | Status | HTTP | Content-Type | Note |\n|---|---|---|---|---|---|---|\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(head + "\n".join(rows) + "\n", encoding="utf-8")


def main():
    write_report([{**s, **probe(s["url"])} for s in SOURCES], "docs/data_access_audit.md")


if __name__ == "__main__":
    main()
