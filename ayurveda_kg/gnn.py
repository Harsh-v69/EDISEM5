"""Heterogeneous GraphSAGE link predictor for (compound, drug) silver labels, trained on the MASKED graph.
Inductive: compound/drug/herb nodes carry features, not ID embeddings, so unseen nodes can be scored. In every fold the
training graph contains no held-out node at all; the full graph is only used at inference."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

EDGE_TYPES = [("compound", "modulates", "gene"), ("gene", "rev_modulates", "compound"),
              ("herb", "contains", "compound"), ("compound", "rev_contains", "herb"),
              ("drug", "targets", "gene"), ("gene", "rev_targets", "drug"), ("drug", "ddi", "drug")]


@dataclass
class Graph:
    data: HeteroData
    cmap: dict
    dmap: dict


def build_graph(fe, edges, keep_compounds=None, keep_drugs=None, device="cpu") -> Graph:
    """HeteroData restricted to the kept compounds/drugs. `edges` must be masked. Genes and herbs are global id spaces."""
    cids = [c for c in fe.compound_ids if keep_compounds is None or c in keep_compounds]
    dids = [d for d in fe.drug_ids if keep_drugs is None or d in keep_drugs]
    cmap, dmap = {c: i for i, c in enumerate(cids)}, {d: i for i, d in enumerate(dids)}
    fc, fd = {c: i for i, c in enumerate(fe.compound_ids)}, {d: i for i, d in enumerate(fe.drug_ids)}
    genes = sorted(set(edges["modulates"]["dst"]) | set(edges["targets"]["dst"]))
    herbs = sorted(set(edges["contains"]["src"]))
    gmap, hmap = {g: i for i, g in enumerate(genes)}, {h: i for i, h in enumerate(herbs)}

    def ei(df, smap, dmap_):
        rows = [(smap[s], dmap_[d]) for s, d in zip(df["src"], df["dst"]) if s in smap and d in dmap_]
        return torch.tensor(rows, dtype=torch.long).t().contiguous() if rows else torch.zeros((2, 0), dtype=torch.long)

    d = HeteroData()
    d["compound"].x = torch.tensor(fe.Xc[[fc[c] for c in cids]], dtype=torch.float32)
    d["drug"].x = torch.tensor(fe.Xd[[fd[x] for x in dids]], dtype=torch.float32)
    d["gene"].x = torch.arange(len(genes))
    d["herb"].x = torch.ones((len(herbs), 1))
    for (s, r, t), (df, sm, tm) in {("compound", "modulates", "gene"): (edges["modulates"], cmap, gmap),
                                    ("herb", "contains", "compound"): (edges["contains"], hmap, cmap),
                                    ("drug", "targets", "gene"): (edges["targets"], dmap, gmap),
                                    ("drug", "ddi", "drug"): (edges["ddi"], dmap, dmap)}.items():
        e = ei(df, sm, tm)
        d[(s, r, t)].edge_index = e
        if r == "ddi":
            d[(s, r, t)].edge_index = torch.cat([e, e.flip(0)], dim=1)
        else:
            d[(t, "rev_" + r, s)].edge_index = e.flip(0)
    return Graph(d.to(device), cmap, dmap)


class HeteroSAGE(nn.Module):
    def __init__(self, d_c, d_d, n_genes, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lin = nn.ModuleDict({"compound": nn.Linear(d_c, hidden), "drug": nn.Linear(d_d, hidden), "herb": nn.Linear(1, hidden)})
        self.gene = nn.Embedding(n_genes, hidden)
        self.convs = nn.ModuleList([HeteroConv({et: SAGEConv((-1, -1), hidden) for et in EDGE_TYPES}, aggr="sum") for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def embed(self, data):
        x = {"compound": self.lin["compound"](data["compound"].x), "drug": self.lin["drug"](data["drug"].x),
             "herb": self.lin["herb"](data["herb"].x), "gene": self.gene(data["gene"].x)}
        x = {k: torch.relu(v) for k, v in x.items()}
        for conv in self.convs:
            out = conv(x, data.edge_index_dict)
            x = {k: self.drop(torch.relu(out.get(k, 0) + v)) for k, v in x.items()}      # residual keeps nodes with no incoming edges
        return x

    def score(self, x, ci, di):
        hc, hd = x["compound"][ci], x["drug"][di]
        return self.head(torch.cat([hc, hd, hc * hd], dim=1)).squeeze(-1)


def _idx(g: Graph, pairs, device):
    return (torch.tensor(pairs["compound"].map(g.cmap).to_numpy(), dtype=torch.long, device=device),
            torch.tensor(pairs["drug"].map(g.dmap).to_numpy(), dtype=torch.long, device=device))


def gnn_fit_predict(fe, edges, hidden=64, layers=2, epochs=100, lr=2e-3, weight_decay=1e-4, dropout=0.2,
                    seed=0, patience=15, val_frac=0.1, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def f(train, test, fold):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        comps = sorted(set(train["compound"]))
        val_c = set(rng.choice(comps, max(1, int(len(comps) * val_frac)), replace=False))
        tr, va = train[~train["compound"].isin(val_c)], train[train["compound"].isin(val_c)]
        g_tr = build_graph(fe, edges, set(fold["train_compounds"]) - val_c, set(fold["train_drugs"]), device)   # no val/test nodes
        g_va = build_graph(fe, edges, set(fold["train_compounds"]), set(fold["train_drugs"]), device)           # val compounds appear cold
        g_all = build_graph(fe, edges, None, None, device)
        n_genes = g_all.data["gene"].x.shape[0]
        model = HeteroSAGE(fe.Xc.shape[1], fe.Xd.shape[1], n_genes, hidden, layers, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        ci, di = _idx(g_tr, tr, device)
        y = torch.tensor(tr["label"].to_numpy(), dtype=torch.float32, device=device)
        vci, vdi = _idx(g_va, va, device)
        best, best_state, bad = -1.0, None, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model.score(model.embed(g_tr.data), ci, di), y)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model.score(model.embed(g_va.data), vci, vdi)).cpu().numpy()
            yv = va["label"].to_numpy()
            auc = roc_auc_score(yv, pv) if len(set(yv.tolist())) == 2 else -loss.item()
            if auc > best:
                best, bad, best_state = auc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        f.last_val = best                                                   # validation AUROC of the chosen epoch (for config selection)
        model.load_state_dict(best_state)
        model.eval()
        tci, tdi = _idx(g_all, test, device)
        with torch.no_grad():
            return torch.sigmoid(model.score(model.embed(g_all.data), tci, tdi)).cpu().numpy()
    f.last_val = None
    return f
