import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ayurveda_kg.evaluate import metrics
from ayurveda_kg.features import Features
from ayurveda_kg.gnn import HeteroSAGE, build_graph, gnn_fit_predict


def toy(n_c=120, n_d=6, seed=0):
    rng = np.random.default_rng(seed)
    Xc = (rng.random((n_c, 8)) < 0.5).astype(np.float32)
    Xd = (rng.random((n_d, 4)) < 0.5).astype(np.float32)
    cids, dids = [f"c{i}" for i in range(n_c)], [f"d{j}" for j in range(n_d)]
    fe = Features(cids, dids, Xc, Xd, [f"x{i}" for i in range(8)], [f"y{i}" for i in range(4)], 0)
    genes = [f"gene:G{k}" for k in range(5)]
    edges = {
        "modulates": pd.DataFrame({"src": [cids[i] for i in range(n_c)], "dst": [genes[i % 5] for i in range(n_c)]}),
        "targets": pd.DataFrame({"src": dids, "dst": [genes[j % 5] for j in range(n_d)]}),
        "contains": pd.DataFrame({"src": [f"herb:H{i % 4}" for i in range(n_c)], "dst": cids}),
        "ddi": pd.DataFrame({"src": ["d0"], "dst": ["d1"], "level": ["Major"]}),
    }
    lab = pd.DataFrame([{"compound": cids[i], "drug": dids[j], "label": int(Xc[i, 0] * Xd[j, 0]), "enzymes": ""}
                        for i in range(n_c) for j in range(n_d)])
    return fe, edges, lab


def test_build_graph_excludes_nodes_not_kept_and_their_edges():
    fe, edges, _ = toy()
    keep = {f"c{i}" for i in range(50)}
    g = build_graph(fe, edges, keep_compounds=keep, keep_drugs={"d0", "d1", "d2"})
    assert g.data["compound"].x.shape[0] == 50 and g.data["drug"].x.shape[0] == 3
    assert "c60" not in g.cmap and "d5" not in g.dmap
    assert g.data[("compound", "modulates", "gene")].edge_index.shape[1] == 50           # edges of dropped compounds are gone
    assert g.data[("drug", "targets", "gene")].edge_index.shape[1] == 3
    assert g.data[("drug", "ddi", "drug")].edge_index.shape[1] == 2                      # stored in both directions
    full = build_graph(fe, edges)
    assert full.data["compound"].x.shape[0] == 120 and full.data[("compound", "modulates", "gene")].edge_index.shape[1] == 120


def test_model_embeds_every_node_type_and_scores_pairs():
    fe, edges, _ = toy()
    g = build_graph(fe, edges)
    m = HeteroSAGE(8, 4, g.data["gene"].x.shape[0], hidden=16)
    x = m.embed(g.data)
    assert set(x) == {"compound", "drug", "gene", "herb"} and x["compound"].shape == (120, 16)
    s = m.score(x, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    assert s.shape == (3,)


def cold_split(lab):
    tr, te = lab[lab.compound.isin([f"c{i}" for i in range(80)])], lab[~lab.compound.isin([f"c{i}" for i in range(80)])]
    return tr, te, {"train_compounds": set(tr.compound), "train_drugs": set(tr.drug)}


def test_gnn_learns_a_feature_rule_on_unseen_compounds():
    fe, edges, lab = toy()
    tr, te, fold = cold_split(lab)
    p = gnn_fit_predict(fe, edges, hidden=32, epochs=150, seed=0, device="cpu")(tr, te, fold)
    assert len(p) == len(te) and ((p >= 0) & (p <= 1)).all()
    assert metrics(te.label.values, p)["auroc"] > 0.9


def test_gnn_is_deterministic_for_a_seed_on_cpu():
    fe, edges, lab = toy()
    tr, te, fold = cold_split(lab)
    a = gnn_fit_predict(fe, edges, hidden=16, epochs=20, seed=2, device="cpu")(tr, te, fold)
    b = gnn_fit_predict(fe, edges, hidden=16, epochs=20, seed=2, device="cpu")(tr, te, fold)
    assert np.allclose(a, b, atol=1e-6)
