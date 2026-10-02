import pandas as pd

from ayurveda_kg.build import (compound_target_edges, ddi_edges, dgidb_edges, herb_edges, metabolism_edges, target_nodes)

DRUGS = pd.DataFrame([{"id": "drug:warfarin", "name": "warfarin", "cls": "a", "chembl_id": "CHEMBL1464"},
                      {"id": "drug:aspirin", "name": "aspirin", "cls": "b", "chembl_id": "CHEMBL25"}])


def test_herb_edges_aggregate_parts_and_skip_uncrawled_compounds():
    rows = [{"plant": "H", "part": "", "phy_id": "P1", "phy_name": "a", "reference": "r1"},
            {"plant": "H", "part": "leaf", "phy_id": "P1", "phy_name": "a", "reference": "r2"},
            {"plant": "H", "part": "root", "phy_id": "P1", "phy_name": "a", "reference": "r3"},
            {"plant": "H", "part": "", "phy_id": "P9", "phy_name": "z", "reference": "r"}]   # P9 never crawled
    e = herb_edges(rows, {"H": "herb:H"}, {"P1": "cpd:X"})
    assert len(e) == 1 and e.iloc[0].parts == "leaf; root" and e.iloc[0].n_refs == 3


def test_compound_target_edges_modulates_cyp_and_pgp():
    targets = {"P1": {"targets": [{"gene": "CYP3A4", "source": "ChEMBL"}, {"gene": "PTGS2", "source": "STITCH"}],
                      "adme": {"CYP3A4 inhibitor": "Yes", "CYP2D6 inhibitor": "No", "P-glycoprotein substrate": "Yes"}},
               "P2": {"targets": [{"gene": "CYP3A4", "source": "STITCH"}], "adme": {}}}
    mod, cyp, pgp = compound_target_edges(targets, {"P1": "cpd:A", "P2": "cpd:A"})   # P1 and P2 merged into one compound
    assert sorted(zip(mod.src, mod.dst)) == [("cpd:A", "gene:CYP3A4"), ("cpd:A", "gene:PTGS2")]
    assert mod.set_index("dst").loc["gene:CYP3A4", "sources"] == "ChEMBL; STITCH"
    assert cyp.dst.tolist() == ["gene:CYP3A4"]
    assert pgp.dst.tolist() == ["gene:ABCB1"]


def test_target_nodes_union_and_cyp_flag():
    e1 = pd.DataFrame([{"src": "c", "dst": "gene:CYP3A4"}])
    e2 = pd.DataFrame([{"src": "d", "dst": "gene:PTGS2"}, {"src": "d", "dst": "gene:CYP3A4"}])
    t = target_nodes([e1, e2]).set_index("id")
    assert sorted(t.index) == ["gene:CYP3A4", "gene:PTGS2"]
    assert bool(t.loc["gene:CYP3A4", "is_cyp"]) and not bool(t.loc["gene:PTGS2", "is_cyp"])


def test_dgidb_edges_match_by_chembl_concept_or_name_and_aggregate():
    df = pd.DataFrame([
        {"gene_name": "VKORC1", "drug_concept_id": "chembl:CHEMBL1464", "drug_name": "WARFARIN SODIUM", "interaction_type": "inhibitor", "interaction_source_db_name": "A", "interaction_score": "0.5"},
        {"gene_name": "VKORC1", "drug_concept_id": "ncit:C1", "drug_name": "WARFARIN", "interaction_type": None, "interaction_source_db_name": "B", "interaction_score": "0.9"},
        {"gene_name": "PTGS1", "drug_concept_id": "ncit:C2", "drug_name": "ASPIRIN", "interaction_type": "inhibitor", "interaction_source_db_name": "A", "interaction_score": None},
        {"gene_name": "X", "drug_concept_id": "ncit:C3", "drug_name": "OTHER", "interaction_type": None, "interaction_source_db_name": "A", "interaction_score": None}])
    e = dgidb_edges(df, DRUGS).set_index(["src", "dst"])
    assert set(e.index) == {("drug:warfarin", "gene:VKORC1"), ("drug:aspirin", "gene:PTGS1")}
    w = e.loc[("drug:warfarin", "gene:VKORC1")]
    assert w.sources == "A; B" and w.interaction_types == "inhibitor" and w.score == 0.9


def test_metabolism_edges_map_salt_ids_to_parent_and_skip_out_of_scope():
    rows = [{"substrate_chembl_id": "CHEMBL1200879", "enzyme_name": "CYP2C9", "drug_chembl_id": "CHEMBL1200879"},   # salt of warfarin
            {"substrate_chembl_id": "CHEMBL1464", "enzyme_name": "cyp2c9", "drug_chembl_id": "CHEMBL1464"},         # dup, lower case
            {"substrate_chembl_id": "CHEMBL999", "enzyme_name": "CYP3A4", "drug_chembl_id": "CHEMBL999"},           # not in scope
            {"substrate_chembl_id": "CHEMBL25", "enzyme_name": None, "drug_chembl_id": "CHEMBL25"}]                 # no enzyme
    e = metabolism_edges(rows, {"CHEMBL1200879": "CHEMBL1464", "CHEMBL1464": "CHEMBL1464", "CHEMBL25": "CHEMBL25"}, DRUGS)
    assert e[["src", "dst"]].values.tolist() == [["drug:warfarin", "gene:CYP2C9"]]


def test_ddi_edges_unordered_pairs_worst_level_scope_only():
    df = pd.DataFrame([{"Drug_A": "Warfarin", "Drug_B": "Aspirin", "Level": "Moderate"},
                       {"Drug_A": "Aspirin", "Drug_B": "Warfarin", "Level": "Major"},
                       {"Drug_A": "Warfarin", "Drug_B": "Iron", "Level": "Major"}])
    e = ddi_edges(df, DRUGS)
    assert len(e) == 1 and e.iloc[0].level == "Major"


# ---- drug aliases, enzyme-name filtering, TDC substrate source ----
from ayurveda_kg.build import merge_edge_sources, tdc_edges
from ayurveda_kg.resolve import alias_map, is_gene_symbol


def test_is_gene_symbol_rejects_enzyme_families():
    assert all(is_gene_symbol(x) for x in ["CYP3A4", "CYP2C9", "UGT1A9", "NAT2", "ABCB1", "CYP2C19"])
    assert not any(is_gene_symbol(x) for x in ["ESTERASES", "UGT", "UDP-GLUCURONOSYLTRANSFERASES", "", "cyp 3a4"])


def test_metabolism_edges_drop_non_gene_enzyme_names():
    rows = [{"substrate_chembl_id": "CHEMBL25", "enzyme_name": "ESTERASES"}, {"substrate_chembl_id": "CHEMBL25", "enzyme_name": "UGT"},
            {"substrate_chembl_id": "CHEMBL25", "enzyme_name": "CYP2C9"}]
    e = metabolism_edges(rows, {"CHEMBL25": "CHEMBL25"}, DRUGS)
    assert e.dst.tolist() == ["gene:CYP2C9"]


def test_alias_map_uses_aliases_and_drops_ambiguous_ones():
    d = pd.DataFrame([{"id": "drug:glibenclamide", "name": "glibenclamide", "aliases": ["Glyburide", "Diabeta"]},
                      {"id": "drug:a", "name": "a", "aliases": ["Shared"]}, {"id": "drug:b", "name": "b", "aliases": ["shared"]}])
    m = alias_map(d)
    assert m["GLYBURIDE"] == "drug:glibenclamide" and m["GLIBENCLAMIDE"] == "drug:glibenclamide"
    assert "SHARED" not in m                      # would silently attach to the wrong drug


def test_dgidb_and_ddi_match_through_aliases():
    drugs = pd.DataFrame([{"id": "drug:glibenclamide", "name": "glibenclamide", "chembl_id": "", "aliases": ["glyburide"]},
                          {"id": "drug:aspirin", "name": "aspirin", "chembl_id": "", "aliases": ["acetylsalicylic acid"]}])
    dg = pd.DataFrame([{"gene_name": "ABCC8", "drug_concept_id": "x:1", "drug_name": "GLYBURIDE", "interaction_type": None,
                        "interaction_source_db_name": "A", "interaction_score": None}])
    assert dgidb_edges(dg, drugs).src.tolist() == ["drug:glibenclamide"]
    dd = pd.DataFrame([{"Drug_A": "Acetylsalicylic acid", "Drug_B": "Glyburide", "Level": "Moderate"}])
    assert len(ddi_edges(dd, drugs)) == 1


def test_tdc_edges_split_substrates_and_verified_non_substrates():
    drugs = pd.DataFrame([{"id": "drug:amlodipine", "name": "amlodipine", "aliases": []},
                          {"id": "drug:metformin", "name": "metformin", "aliases": []}])
    frames = {"CYP3A4": pd.DataFrame({"Drug_ID": ["amlodipine", "metformin", "unrelated"], "Y": [1, 0, 1]})}
    pos, neg = tdc_edges(frames, drugs)
    assert pos[["src", "dst"]].values.tolist() == [["drug:amlodipine", "gene:CYP3A4"]]
    assert neg[["src", "dst"]].values.tolist() == [["drug:metformin", "gene:CYP3A4"]]


def test_merge_edge_sources_unions_sources_per_pair():
    a = pd.DataFrame([{"src": "d", "dst": "g", "source": "ChEMBL metabolism"}])
    b = pd.DataFrame([{"src": "d", "dst": "g", "source": "TDC Carbon-Mangels"}, {"src": "d", "dst": "h", "source": "TDC Carbon-Mangels"}])
    m = merge_edge_sources(a, b).set_index("dst")
    assert m.loc["g", "source"] == "ChEMBL metabolism; TDC Carbon-Mangels" and len(m) == 2


# ---- has_adme flag (labels need to tell "predicted No" from "never predicted") ----
from ayurveda_kg.build import with_adme_flag


def test_with_adme_flag_true_only_when_any_merged_phy_id_has_predictions():
    comp = pd.DataFrame([{"id": "cpd:A", "phy_ids": ["P1", "P2"]}, {"id": "cpd:B", "phy_ids": ["P3"]},
                         {"id": "cpd:C", "phy_ids": ["P4"]}])
    targets = {"P1": {"targets": [], "adme": {}}, "P2": {"targets": [], "adme": {"CYP3A4 inhibitor": "No"}},
               "P3": {"targets": [{"gene": "X"}], "adme": {}}}      # P4 never crawled
    out = with_adme_flag(comp, targets).set_index("id")["has_adme"]
    assert out.to_dict() == {"cpd:A": True, "cpd:B": False, "cpd:C": False}
