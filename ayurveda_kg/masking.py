"""Leakage firewall: silver labels are computed from CYP edges that live in the KG, so those edges are removed
before any model sees the graph. Otherwise a GNN would just rediscover the labelling rule."""
import pandas as pd

from ayurveda_kg.labels import CYP5

# whole edge types that define or reveal the labels
LABEL_SOURCE_EDGE_TYPES = ("predicted_cyp_inhibitor", "substrate_of", "non_substrate_of")
# other types are masked only for edges that touch one of the five label enzymes
CYP_TOUCHING_EDGE_TYPES = ("modulates", "targets")


def _touches_cyp(df, enzymes):
    return df["dst"].isin({f"gene:{g}" for g in enzymes})


def mask_label_edges(edges: dict, enzymes=CYP5) -> dict:
    out = {}
    for t, df in edges.items():
        if t in LABEL_SOURCE_EDGE_TYPES:
            out[t] = df.iloc[0:0]
        elif t in CYP_TOUCHING_EDGE_TYPES:
            out[t] = df[~_touches_cyp(df, enzymes)].reset_index(drop=True)
        else:
            out[t] = df
    return out


def assert_no_label_leak(edges: dict, enzymes=CYP5):
    for t in LABEL_SOURCE_EDGE_TYPES:
        assert t not in edges or len(edges[t]) == 0, f"label-source edge type still present: {t}"
    for t in CYP_TOUCHING_EDGE_TYPES:
        if t in edges:
            assert not _touches_cyp(edges[t], enzymes).any(), f"{t} edges into label enzymes still present"
