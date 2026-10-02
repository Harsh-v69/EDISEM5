"""Heterogeneous knowledge graph: node/edge tables on disk (parquet), NetworkX MultiDiGraph in memory."""
from pathlib import Path

import networkx as nx
import pandas as pd

# edge type -> (source node type, destination node type). The single source of truth for the schema.
SCHEMA = {
    "contains": ("Herb", "Compound"),
    "modulates": ("Compound", "Target"),
    "predicted_cyp_inhibitor": ("Compound", "Target"),
    "predicted_pgp_substrate": ("Compound", "Target"),
    "targets": ("Drug", "Target"),
    "substrate_of": ("Drug", "Target"),
    "non_substrate_of": ("Drug", "Target"),  # verified negatives (TDC); used as real negatives for silver labels
    "ddi": ("Drug", "Drug"),
}


def build_kg(nodes: dict, edges: dict) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for t, df in nodes.items():
        for r in df.to_dict("records"):
            g.add_node(r.pop("id"), type=t, **r)
    for et, df in edges.items():
        for r in df.to_dict("records"):
            g.add_edge(r.pop("src"), r.pop("dst"), type=et, **r)
    return g


def save_tables(nodes, edges, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for t, df in nodes.items():
        df.to_parquet(out / f"nodes_{t}.parquet", index=False)
    for et, df in edges.items():
        df.to_parquet(out / f"edges_{et}.parquet", index=False)


def load_tables(in_dir):
    d = Path(in_dir)
    nodes = {p.stem[6:]: pd.read_parquet(p) for p in sorted(d.glob("nodes_*.parquet"))}
    edges = {p.stem[6:]: pd.read_parquet(p) for p in sorted(d.glob("edges_*.parquet"))}
    return nodes, edges
