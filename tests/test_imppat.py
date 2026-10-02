from pathlib import Path

import pytest

from ayurveda_kg.ingest.imppat import parse_detail_page, parse_plant_page, parse_targets_page

F = Path(__file__).parent / "fixtures"


def read(name):
    f = F / name
    if not f.exists():  # IMPPAT page excerpts are CC BY-NC-ND and are not committed; they exist only locally
        pytest.skip(f"fixture {name} not present (IMPPAT licence: not redistributed)")
    return f.read_text(encoding="utf-8")


def test_parse_plant_page_rows_and_part():
    rows = parse_plant_page(read("imppat_plant_Curcuma_longa.html"))
    assert len(rows) == 4
    assert rows[0] == {"plant": "Curcuma longa", "part": "", "phy_id": "IMPPAT3_PHYID000017",
                       "phy_name": "Vitamin E", "reference": "ISBN:9780387706375"}
    assert rows[-1]["part"] == "flower"
    assert rows[-1]["reference"].startswith("DOI:")


def test_parse_detail_page_identifiers():
    d = parse_detail_page(read("imppat_detail_PHYID000017.html"))
    assert d["phy_id"] == "IMPPAT3_PHYID000017"
    assert d["name"] == "Vitamin E"
    assert d["inchikey"] == "GVJHHUAWPYXKBD-IEOSBIPESA-N"
    assert d["chembl_id"] == "CHEMBL47"
    assert d["cid"] == "14985"
    assert d["smiles"].startswith("C[C@@H](CCC[C@]1(C)CCc2c(O1)")
    assert "alpha tocopherol" in d["synonyms"]


def test_parse_targets_page_targets_and_adme():
    t = parse_targets_page(read("imppat_targets_PHYID000017.html"))
    genes = {x["gene"]: x for x in t["targets"]}
    assert genes["CYP3A4"]["entrez"] == "1576"
    assert genes["CYP3A4"]["ensembl"] == "ENSG00000160868"
    assert genes["CYP3A4"]["source"] == "ChEMBL"
    assert t["adme"]["P-glycoprotein substrate"] == "Yes"
    assert t["adme"]["CYP3A4 inhibitor"] == "No"
    assert len(t["targets"]) > 5


def test_parsers_fail_soft_on_empty_page():
    assert parse_plant_page("<html></html>") == []
    assert parse_targets_page("<html></html>") == {"targets": [], "adme": {}}
    assert parse_detail_page("<html></html>")["inchikey"] == ""


# ---- crawler ----
import csv

from ayurveda_kg import manifest
from ayurveda_kg.ingest.imppat import crawl_compounds, plant_ids


class _Resp:
    def __init__(self, content=b"<html>x</html>", status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


class _Sess:
    def __init__(self, bad=()):
        self.urls, self.bad = [], bad

    def get(self, url, **kw):
        self.urls.append(url)
        return _Resp(status=500 if any(b in url for b in self.bad) else 200)


def test_plant_ids_unique_and_ordered():
    rows = parse_plant_page(read("imppat_plant_Curcuma_longa.html"))
    ids = plant_ids(rows)
    assert ids == sorted(set(ids)) and "IMPPAT3_PHYID000023" in ids and len(ids) == 3


def test_crawl_compounds_index_resume_and_failures(tmp_path):
    s = _Sess(bad=("PHYID2/",))  # no real id ends with '/', so use a tail match below
    s = _Sess(bad=("humantargets/B",))
    mp = tmp_path / "m.json"
    out = crawl_compounds(["A", "B"], tmp_path / "raw", delay=0, session=s, manifest_path=mp)
    assert out["ok"] == 3 and out["failed"] == 1
    idx = list(csv.DictReader(open(tmp_path / "raw" / "crawl_index.tsv", encoding="utf-8"), delimiter="\t"))
    assert len(idx) == 3 and all(len(r["sha256"]) == 64 for r in idx)
    assert "imppat_crawl_index" in manifest.load(mp)
    n = len(s.urls)
    crawl_compounds(["A"], tmp_path / "raw", delay=0, session=s, manifest_path=mp)  # cached: no refetch
    assert len(s.urls) == n
