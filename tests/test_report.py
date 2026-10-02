import pandas as pd

from ayurveda_kg.report import make_report


def test_report_lists_counts_and_flags_gaps():
    nodes = {"Herb": pd.DataFrame([{"id": "herb:H", "name": "H"}, {"id": "herb:E", "name": "E"}]),
             "Compound": pd.DataFrame([{"id": "cpd:C", "name": "C"}]),
             "Drug": pd.DataFrame([{"id": "drug:a", "name": "a"}, {"id": "drug:b", "name": "b"}]),
             "Target": pd.DataFrame([{"id": "gene:CYP3A4", "symbol": "CYP3A4"}])}
    edges = {"contains": pd.DataFrame([{"src": "herb:H", "dst": "cpd:C"}]),
             "substrate_of": pd.DataFrame([{"src": "drug:a", "dst": "gene:CYP3A4"}]),
             "targets": pd.DataFrame(columns=["src", "dst"])}
    md = make_report(nodes, edges, {"compounds_crawled": 1, "compounds_expected": 4, "problems": []})
    assert "| Herb | 2 |" in md and "1 of 4" in md
    assert "No validation problems" in md
    assert "Herbs with no crawled compounds: E" in md
    assert "Drugs with no substrate_of edge: b" in md
    assert "Drugs with no targets edge: a, b" in md
