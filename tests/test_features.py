import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.features import build_features, leaky_label_features, morgan_fingerprints
from ayurveda_kg.masking import mask_label_edges


def toy():
    mk = lambda rows, cols=("src", "dst"): pd.DataFrame(rows, columns=list(cols))
    nodes = {"Compound": pd.DataFrame({"id": ["cpd:1", "cpd:2"], "smiles": ["CCO", "c1ccccc1"], "has_adme": [True, True]}),
             "Drug": pd.DataFrame({"id": ["drug:a", "drug:b"], "name": ["a", "b"], "cls": ["statin", "nsaid"]})}
    edges = {
        "modulates": mk([("cpd:1", "gene:PTGS2"), ("cpd:1", "gene:CYP3A4"), ("cpd:2", "gene:ESR1")]),
        "predicted_cyp_inhibitor": mk([("cpd:1", "gene:CYP3A4")]),
        "predicted_pgp_substrate": mk([("cpd:2", "gene:ABCB1")]),
        "targets": mk([("drug:a", "gene:HMGCR"), ("drug:a", "gene:CYP2D6"), ("drug:b", "gene:PTGS2")]),
        "substrate_of": mk([("drug:a", "gene:CYP3A4")]), "non_substrate_of": mk([("drug:b", "gene:CYP3A4")]),
        "ddi": mk([("drug:a", "drug:b", "Major")], cols=("src", "dst", "level")),
        "contains": mk([("herb:H", "cpd:1")]),
    }
    return nodes, edges


def test_morgan_fingerprints_shape_dtype_and_invalid_smiles():
    pytest.importorskip("rdkit")
    fp, bad = morgan_fingerprints(["CCO", "c1ccccc1", "not a smiles", ""], nbits=64)
    assert fp.shape == (4, 64) and fp.dtype == np.uint8
    assert fp[0].sum() > 0 and fp[1].sum() > 0 and not fp[2].any() and not fp[3].any()
    assert bad == 2
    assert not np.array_equal(fp[0], fp[1])                       # different molecules, different fingerprints


def test_features_use_only_masked_graph_and_have_aligned_ids():
    pytest.importorskip("rdkit")
    nodes, edges = toy()
    f = build_features(nodes, mask_label_edges(edges), nbits=32, n_cpd_genes=8, n_drug_genes=8)
    assert f.compound_ids == ["cpd:1", "cpd:2"] and f.drug_ids == ["drug:a", "drug:b"]
    assert f.Xc.shape[0] == 2 and f.Xd.shape[0] == 2
    assert not any("CYP" in n for n in f.compound_cols + f.drug_cols)      # masked CYP targets never become features
    assert "cpd_gene:PTGS2" in f.compound_cols and "drug_gene:HMGCR" in f.drug_cols
    assert f.Xc[0, f.compound_cols.index("cpd_gene:PTGS2")] == 1 and f.Xc[1, f.compound_cols.index("cpd_gene:PTGS2")] == 0
    assert f.Xc[1, f.compound_cols.index("pgp_substrate")] == 1
    assert f.Xd[0, f.drug_cols.index("cls:statin")] == 1 and f.Xd[1, f.drug_cols.index("cls:statin")] == 0
    assert f.Xd[0, f.drug_cols.index("ddi_major")] == 1 and f.Xd[1, f.drug_cols.index("ddi_major")] == 1


def test_leaky_features_expose_the_label_inputs_exactly():
    nodes, edges = toy()
    lc, ld, cols = leaky_label_features(nodes, edges)
    assert lc.shape == (2, 5) and ld.shape == (2, 10)
    assert lc[0].tolist() == [0, 0, 0, 0, 1] and lc[1].tolist() == [0, 0, 0, 0, 0]        # CYP order: 1A2,2C9,2C19,2D6,3A4
    assert ld[0, :5].tolist() == [0, 0, 0, 0, 1]            # drug a is a CYP3A4 substrate
    assert ld[1, 5:].tolist() == [0, 0, 0, 0, 1]            # drug b is a verified CYP3A4 non-substrate
