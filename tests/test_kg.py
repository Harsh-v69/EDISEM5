import pandas as pd

from ayurveda_kg.kg import SCHEMA, build_kg, load_tables, save_tables
from ayurveda_kg.validate import check_edges, check_nodes, check_scope, validate_all


def tiny():
    nodes = {
        "Herb": pd.DataFrame([{"id": "herb:H", "name": "H"}]),
        "Compound": pd.DataFrame([{"id": "cpd:C", "name": "C"}]),
        "Target": pd.DataFrame([{"id": "gene:CYP3A4", "symbol": "CYP3A4"}]),
        "Drug": pd.DataFrame([{"id": "drug:d", "name": "d"}]),
    }
    edges = {
        "contains": pd.DataFrame([{"src": "herb:H", "dst": "cpd:C"}]),
        "predicted_cyp_inhibitor": pd.DataFrame([{"src": "cpd:C", "dst": "gene:CYP3A4", "source": "SwissADME"}]),
        "substrate_of": pd.DataFrame([{"src": "drug:d", "dst": "gene:CYP3A4"}]),
    }
    return nodes, edges


def test_build_kg_counts_and_types():
    g = build_kg(*tiny())
    assert g.number_of_nodes() == 4 and g.number_of_edges() == 3
    assert g.nodes["herb:H"]["type"] == "Herb"
    assert {d["type"] for _, _, d in g.edges(data=True)} == {"contains", "predicted_cyp_inhibitor", "substrate_of"}


def test_save_load_roundtrip(tmp_path):
    nodes, edges = tiny()
    save_tables(nodes, edges, tmp_path)
    n2, e2 = load_tables(tmp_path)
    assert n2["Target"].symbol.tolist() == ["CYP3A4"]
    assert e2["substrate_of"].src.tolist() == ["drug:d"]
    assert build_kg(n2, e2).number_of_edges() == 3


def test_clean_graph_has_no_problems():
    assert validate_all(*tiny()) == []


def test_validator_catches_dangling_type_mismatch_duplicate_and_unknown_edge_type():
    nodes, edges = tiny()
    edges["contains"] = pd.concat([edges["contains"], pd.DataFrame([{"src": "herb:H", "dst": "cpd:MISSING"}])])
    edges["substrate_of"] = pd.DataFrame([{"src": "herb:H", "dst": "gene:CYP3A4"}])   # Herb cannot be a substrate_of source
    edges["mystery"] = pd.DataFrame([{"src": "drug:d", "dst": "gene:CYP3A4"}])
    nodes["Drug"] = pd.concat([nodes["Drug"], nodes["Drug"]])                           # duplicate node id
    problems = "\n".join(check_edges(nodes, edges) + check_nodes(nodes))
    assert "dangling" in problems and "cpd:MISSING" in problems
    assert "type mismatch" in problems
    assert "unknown edge type" in problems
    assert "duplicate node id" in problems


def test_check_scope_flags_missing_herb_and_drug_and_empty_herb():
    nodes, edges = tiny()
    scope = {"herbs": [{"imppat_name": "H"}, {"imppat_name": "Ghost"}], "drugs": [{"name": "d"}, {"name": "x"}]}
    p = "\n".join(check_scope(nodes, edges, scope))
    assert "herb missing" in p and "Ghost" in p and "drug missing" in p and "x" in p
    nodes["Herb"] = pd.concat([nodes["Herb"], pd.DataFrame([{"id": "herb:Empty", "name": "Empty"}])])
    scope["herbs"] = [{"imppat_name": "H"}, {"imppat_name": "Empty"}]
    scope["drugs"] = [{"name": "d"}]
    assert "no compounds" in "\n".join(check_scope(nodes, edges, scope))


def test_schema_types_are_known():
    allowed = {"Herb", "Compound", "Target", "Drug"}
    assert all(s in allowed and d in allowed for s, d in SCHEMA.values())
