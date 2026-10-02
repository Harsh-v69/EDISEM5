"""Gate on the REAL built KG: known pharmacology must be present. Skipped until the full crawl has been built."""
import json
from pathlib import Path

import pytest

from ayurveda_kg.kg import load_tables

KG = Path("data/processed/kg")
META = Path("data/processed/kg_meta.json")

pytestmark = pytest.mark.skipif(not META.exists() or json.loads(META.read_text())["compounds_crawled"] < json.loads(META.read_text())["compounds_expected"],
                                reason="full IMPPAT crawl not yet built")


def tables():
    return load_tables(KG)


def test_every_scoped_herb_has_many_compounds():
    n, e = tables()
    per = e["contains"].groupby("src").size()
    assert len(per) == 20 and per.min() >= 20, per.sort_values().head()


def test_piperine_is_linked_to_cyp3a4_or_pgp():
    n, e = tables()
    c = n["Compound"]
    piperine = c[c["name"].str.lower() == "piperine"]
    assert len(piperine) == 1
    cid = piperine.iloc[0]["id"]
    hits = set(e["modulates"].query("src == @cid").dst) | set(e["predicted_cyp_inhibitor"].query("src == @cid").dst) | set(e["predicted_pgp_substrate"].query("src == @cid").dst)
    assert {"gene:CYP3A4", "gene:ABCB1"} & hits, hits


def test_black_pepper_contains_piperine_and_curcuma_contains_curcumin():
    n, e = tables()
    c = n["Compound"].set_index("id")["name"].str.lower()
    for herb, cpd in [("herb:Piper nigrum", "piperine"), ("herb:Curcuma longa", "curcumin")]:
        names = {c[d] for d in e["contains"].query("src == @herb").dst}
        assert cpd in names, (herb, cpd)


def test_warfarin_is_cyp2c9_substrate_and_ketoconazole_style_negatives_exist():
    n, e = tables()
    s = set(map(tuple, e["substrate_of"][["src", "dst"]].values))
    assert ("drug:warfarin", "gene:CYP2C9") in s and ("drug:simvastatin", "gene:CYP3A4") in s
    assert len(e["non_substrate_of"]) > 10


def test_every_drug_pair_ddi_is_between_distinct_scoped_drugs():
    n, e = tables()
    ids = set(n["Drug"]["id"])
    assert all(a in ids and b in ids and a != b for a, b in zip(e["ddi"].src, e["ddi"].dst))
