"""One command to fetch every open drug-side file and the public-domain texts (idempotent: cached files are never refetched, and every file is
recorded in data/manifest.json with source, licence and checksum). IMPPAT pages are fetched by the crawlers instead
(python -m ayurveda_kg.ingest.crawl_imppat, crawl_formulations). Licence strings are as recorded at fetch time; verify exact terms before any release."""
import json
from pathlib import Path

from ayurveda_kg.ingest.fetch import fetch
from ayurveda_kg.scope import load_scope

MANIFEST = "data/manifest.json"
DGIDB = "DGIdb open data (verify terms)"
DDINTER = "DDInter 2.0 public; cite"
TDC = "TDC / Carbon-Mangels & Hutter 2011; cite both"
DLI = "Public domain (US); scan on archive.org"

FILES = (
    [{"name": f"dgidb_{f}", "url": f"https://dgidb.org/data/latest/{f}.tsv", "dest": f"data/raw/dgidb/{f}.tsv", "licence": DGIDB}
     for f in ("interactions", "drugs", "genes")]
    + [{"name": f"ddinter_{c}", "url": f"https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_{c}.csv",
        "dest": f"data/raw/ddinter/ddinter_code_{c}.csv", "licence": DDINTER} for c in "ABDHLPRV"]
    + [{"name": f"tdc_{g}_substrate", "url": f"https://dataverse.harvard.edu/api/access/datafile/{i}",
        "dest": f"data/raw/tdc_cyp/{g}_substrate_carbonmangels.tab", "licence": TDC}
       for g, i in (("CYP2C9", 4259584), ("CYP2D6", 4259578), ("CYP3A4", 4259581))]
    + [{"name": f"text_{k}", "url": f"https://archive.org/download/{path}", "dest": f"data/raw/texts/{k}.txt", "licence": f"{DLI}: {path.split('/')[0]}"}
       for k, path in (("charaka_kaviratna", "BIUSante_47357/BIUSante_47357_djvu.txt"),
                       ("sushruta_v1", "in.ernet.dli.2015.43171/2015.43171.The-Sushruta-Samhita--Vol1_djvu.txt"),
                       ("sushruta_v2", "in.ernet.dli.2015.39322/2015.39322.An-English-Translation-Of-The-Sushruta-Samhita--Vol2_djvu.txt"),
                       ("sushruta_v3", "in.ernet.dli.2015.39323/2015.39323.An-English-Translation-Of-The-Sushruta-Samhita--Vol3_djvu.txt"))]
)


def fetch_chembl():
    """Resolve the scoped drugs to ChEMBL parent ids (drug_ids.json) and download the whole metabolism table."""
    from ayurveda_kg.ingest.chembl import fetch_metabolism_all, resolve_drug_ids
    raw = "data/raw/chembl"
    ids = resolve_drug_ids([d["name"] for d in load_scope()["drugs"]], raw, manifest_path=MANIFEST)
    Path(raw).mkdir(parents=True, exist_ok=True)
    Path(f"{raw}/drug_ids.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")
    fetch_metabolism_all(raw, manifest_path=MANIFEST)
    return ids


def main():
    for f in FILES:
        fetch(f["url"], f["dest"], name=f["name"], licence=f["licence"], manifest_path=MANIFEST, delay=1.0)
        print("ok", f["dest"], flush=True)
    ids = fetch_chembl()
    print(f"ChEMBL: {sum(1 for v in ids.values() if v)}/{len(ids)} scoped drugs resolved", flush=True)


if __name__ == "__main__":
    main()
