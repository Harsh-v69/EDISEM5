import pandas as pd
import pytest

from ayurveda_kg.resolve import herb_table, resolve_compounds, resolve_drugs

SCOPE = {"herbs": [{"imppat_name": "Tinospora sinensis", "aliases": ["Tinospora cordifolia", "Giloy"], "common": "guduchi"},
                   {"imppat_name": "Curcuma longa", "aliases": ["Turmeric"], "common": "turmeric"}],
         "drugs": [{"name": "warfarin", "cls": "anticoagulant"}, {"name": "aspirin", "cls": "antiplatelet"}]}


def test_herb_table_ids_and_alias_lookup():
    h = herb_table(SCOPE)
    assert list(h.columns) == ["id", "name", "aliases", "common"]
    assert h.id.tolist() == ["herb:Tinospora sinensis", "herb:Curcuma longa"]
    assert "Tinospora cordifolia" in h.loc[0, "aliases"]


def test_resolve_compounds_merges_same_inchikey_and_keeps_all_phy_ids():
    details = [
        {"phy_id": "P1", "name": "Curcumin", "inchikey": "AAA-BBB-N", "smiles": "C", "cid": "1", "chembl_id": "CHEMBL1", "synonyms": ["x"]},
        {"phy_id": "P2", "name": "curcumin", "inchikey": "AAA-BBB-N", "smiles": "C", "cid": "1", "chembl_id": "CHEMBL1", "synonyms": ["y"]},
        {"phy_id": "P3", "name": "Other", "inchikey": "CCC-DDD-N", "smiles": "CC", "cid": "", "chembl_id": "", "synonyms": []},
        {"phy_id": "P4", "name": "NoKey", "inchikey": "", "smiles": "", "cid": "", "chembl_id": "", "synonyms": []},
    ]
    c, phy2cid = resolve_compounds(details)
    assert len(c) == 3
    assert phy2cid["P1"] == phy2cid["P2"] == "cpd:AAA-BBB-N"
    assert phy2cid["P4"] == "cpd:P4"          # no InChIKey: fall back to the IMPPAT id, never dropped
    row = c.set_index("id").loc["cpd:AAA-BBB-N"]
    assert sorted(row.phy_ids) == ["P1", "P2"] and row.n_merged == 2


def test_resolve_drugs_reports_unresolved_instead_of_dropping():
    d, missing = resolve_drugs(SCOPE, {"warfarin": "CHEMBL1464", "aspirin": None})
    assert d.id.tolist() == ["drug:warfarin", "drug:aspirin"]
    assert d.set_index("name").loc["warfarin", "chembl_id"] == "CHEMBL1464"
    assert missing == ["aspirin"]
