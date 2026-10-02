import json

from ayurveda_kg.ingest.chembl import pick_molecule, parse_metabolism

SEARCH = {"molecules": [
    {"molecule_chembl_id": "CHEMBL1200879", "pref_name": "WARFARIN SODIUM", "max_phase": 4,
     "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1464"}, "molecule_synonyms": [{"molecule_synonym": "Warfarin sodium"}]},
    {"molecule_chembl_id": "CHEMBL1464", "pref_name": "WARFARIN", "max_phase": 4,
     "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1464"}, "molecule_synonyms": [{"molecule_synonym": "Warfarin"}, {"molecule_synonym": "Coumadin"}]},
    {"molecule_chembl_id": "CHEMBL251074", "pref_name": None, "max_phase": None,
     "molecule_hierarchy": None, "molecule_synonyms": []},
]}


def test_pick_molecule_prefers_parent_exact_match():
    assert pick_molecule(SEARCH, "warfarin") == "CHEMBL1464"


def test_pick_molecule_matches_via_synonym():
    s = {"molecules": [{"molecule_chembl_id": "CHEMBL472", "pref_name": "GLYBURIDE", "max_phase": 4,
                        "molecule_hierarchy": {"parent_chembl_id": "CHEMBL472"},
                        "molecule_synonyms": [{"molecule_synonym": "Glibenclamide"}, {"molecule_synonym": "Glyburide"}]}]}
    assert pick_molecule(s, "glibenclamide") == "CHEMBL472"


def test_pick_molecule_no_match_returns_none():
    assert pick_molecule({"molecules": [SEARCH["molecules"][2]]}, "warfarin") is None
    assert pick_molecule({"molecules": []}, "warfarin") is None


def test_parse_metabolism_keeps_needed_fields():
    page = {"metabolisms": [{"drug_chembl_id": "CHEMBL1200879", "substrate_chembl_id": "CHEMBL1464",
                             "enzyme_name": "CYP2C9", "enzyme_chembl_id": "CHEMBL3397", "organism": None,
                             "substrate_name": "WARFARIN", "metabolite_name": "(S)-6-HYDROXYWARFARIN", "extra": 1}]}
    rows = parse_metabolism(page)
    assert rows == [{"drug_chembl_id": "CHEMBL1200879", "substrate_chembl_id": "CHEMBL1464", "enzyme_name": "CYP2C9",
                     "enzyme_chembl_id": "CHEMBL3397", "substrate_name": "WARFARIN", "metabolite_name": "(S)-6-HYDROXYWARFARIN"}]
    assert parse_metabolism({}) == []


def test_pick_molecule_returns_parent_id_not_salt_or_duplicate_entry():
    s = {"molecules": [
        {"molecule_chembl_id": "CHEMBL1355736", "pref_name": "THEOPHYLLINE", "max_phase": 4,
         "molecule_hierarchy": {"parent_chembl_id": "CHEMBL190"}, "molecule_synonyms": []},
        {"molecule_chembl_id": "CHEMBL190", "pref_name": "THEOPHYLLINE ANHYDROUS", "max_phase": 4,
         "molecule_hierarchy": {"parent_chembl_id": "CHEMBL190"}, "molecule_synonyms": []}]}
    assert pick_molecule(s, "theophylline") == "CHEMBL190"
