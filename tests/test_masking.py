import pandas as pd
import pytest

from ayurveda_kg.labels import silver_labels
from ayurveda_kg.masking import LABEL_SOURCE_EDGE_TYPES, assert_no_label_leak, mask_label_edges


def toy():
    mk = lambda rows: pd.DataFrame(rows, columns=["src", "dst"])
    edges = {
        "contains": mk([("herb:H", "cpd:c1")]),
        "modulates": mk([("cpd:c1", "gene:CYP3A4"), ("cpd:c1", "gene:PTGS2"), ("cpd:c1", "gene:ABCB1")]),
        "predicted_cyp_inhibitor": mk([("cpd:c1", "gene:CYP3A4")]),
        "predicted_pgp_substrate": mk([("cpd:c1", "gene:ABCB1")]),
        "targets": mk([("drug:a", "gene:CYP2D6"), ("drug:a", "gene:VKORC1")]),
        "substrate_of": mk([("drug:a", "gene:CYP3A4")]),
        "non_substrate_of": mk([("drug:a", "gene:CYP2C9")]),
        "ddi": mk([("drug:a", "drug:b")]),
    }
    nodes = {"Compound": pd.DataFrame({"id": ["cpd:c1"], "has_adme": [True]}), "Drug": pd.DataFrame({"id": ["drug:a", "drug:b"]})}
    return nodes, edges


def test_unmasked_graph_leaks_and_masking_removes_every_label_path():
    nodes, edges = toy()
    with pytest.raises(AssertionError):
        assert_no_label_leak(edges)
    masked = mask_label_edges(edges)
    assert_no_label_leak(masked)                                   # must not raise
    assert all(t not in masked or len(masked[t]) == 0 for t in LABEL_SOURCE_EDGE_TYPES)
    assert masked["modulates"].dst.tolist() == ["gene:PTGS2", "gene:ABCB1"]   # non-CYP targets survive
    assert masked["targets"].dst.tolist() == ["gene:VKORC1"]
    assert len(masked["ddi"]) == 1 and len(masked["contains"]) == 1            # unrelated structure untouched


def test_labels_cannot_be_rederived_from_masked_graph():
    nodes, edges = toy()
    assert (silver_labels(nodes, edges).label == 1).any()          # sanity: the unmasked graph does give a positive
    masked = mask_label_edges(edges)
    for t in LABEL_SOURCE_EDGE_TYPES:                              # silver_labels expects all three tables to exist
        masked.setdefault(t, pd.DataFrame(columns=["src", "dst"]))
    assert not (silver_labels(nodes, masked).label == 1).any()


def test_mask_does_not_mutate_input():
    nodes, edges = toy()
    n_before = {k: len(v) for k, v in edges.items()}
    mask_label_edges(edges)
    assert {k: len(v) for k, v in edges.items()} == n_before
