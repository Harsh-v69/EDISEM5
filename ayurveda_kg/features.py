"""Model inputs built ONLY from the masked graph (plus compound chemistry). The leaky variant exists solely for the leakage ablation."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ayurveda_kg.labels import CYP5

DDI_LEVELS = ("Major", "Moderate", "Minor")


def morgan_fingerprints(smiles, nbits=1024, radius=2):
    """Binary Morgan fingerprints. Invalid/empty SMILES give an all-zero row and are counted, never dropped."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    out, bad = np.zeros((len(smiles), nbits), dtype=np.uint8), 0
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() else None
        if m is None:
            bad += 1
        else:
            out[i] = gen.GetFingerprintAsNumPy(m)
    return out, bad


def _multi_hot(ids, edges, n_top):
    """Multi-hot over the n_top most frequent destination genes. Returns (matrix, gene list)."""
    counts = edges.groupby("dst")["src"].nunique().sort_values(ascending=False, kind="stable")
    genes = list(counts.index[:n_top])
    pos, row = {g: j for j, g in enumerate(genes)}, {i: r for r, i in enumerate(ids)}
    X = np.zeros((len(ids), len(genes)), dtype=np.float32)
    for s, d in zip(edges["src"], edges["dst"]):
        if d in pos and s in row:
            X[row[s], pos[d]] = 1
    return X, genes


@dataclass
class Features:
    compound_ids: list
    drug_ids: list
    Xc: np.ndarray
    Xd: np.ndarray
    compound_cols: list
    drug_cols: list
    n_bad_smiles: int


def build_features(nodes, edges, nbits=1024, n_cpd_genes=256, n_drug_genes=256) -> Features:
    """edges must already be masked (see masking.mask_label_edges)."""
    cids, dids = list(nodes["Compound"]["id"]), list(nodes["Drug"]["id"])
    fp, bad = morgan_fingerprints(list(nodes["Compound"]["smiles"]), nbits=nbits)
    mod, genes = _multi_hot(cids, edges["modulates"], n_cpd_genes)
    pgp = np.zeros((len(cids), 1), dtype=np.float32)
    row = {c: i for i, c in enumerate(cids)}
    for s in edges["predicted_pgp_substrate"]["src"]:
        pgp[row[s], 0] = 1
    Xc = np.hstack([fp.astype(np.float32), mod, pgp])
    ccols = [f"fp{i}" for i in range(nbits)] + [f"cpd_gene:{g[5:]}" for g in genes] + ["pgp_substrate"]

    tgt, dgenes = _multi_hot(dids, edges["targets"], n_drug_genes)
    classes = sorted(nodes["Drug"]["cls"].unique())
    cls = np.array([[1.0 if c == x else 0.0 for c in classes] for x in nodes["Drug"]["cls"]], dtype=np.float32)
    drow, ddi = {d: i for i, d in enumerate(dids)}, np.zeros((len(dids), len(DDI_LEVELS) + 1), dtype=np.float32)
    for s, d, lvl in zip(edges["ddi"]["src"], edges["ddi"]["dst"], edges["ddi"]["level"]):
        for x in (s, d):
            if lvl in DDI_LEVELS:
                ddi[drow[x], DDI_LEVELS.index(lvl)] += 1
            ddi[drow[x], -1] += 1
    Xd = np.hstack([tgt, cls, ddi])
    dcols = [f"drug_gene:{g[5:]}" for g in dgenes] + [f"cls:{c}" for c in classes] + [f"ddi_{l.lower()}" for l in DDI_LEVELS] + ["ddi_total"]
    return Features(cids, dids, Xc, Xd, ccols, dcols, bad)


def leaky_label_features(nodes, edges):
    """UNMASKED label inputs: compound CYP-inhibition (5) and drug substrate (5) + verified non-substrate (5) flags.
    Using these makes the silver label (almost) exactly computable; only for the leakage ablation."""
    cids, dids = list(nodes["Compound"]["id"]), list(nodes["Drug"]["id"])
    gi = {f"gene:{g}": j for j, g in enumerate(CYP5)}

    def flags(ids, df, width, offset=0, out=None):
        out = np.zeros((len(ids), width), dtype=np.float32) if out is None else out
        row = {x: r for r, x in enumerate(ids)}
        for s, d in zip(df["src"], df["dst"]):
            if d in gi and s in row:
                out[row[s], offset + gi[d]] = 1
        return out

    lc = flags(cids, edges["predicted_cyp_inhibitor"], 5)
    ld = flags(dids, edges["substrate_of"], 10)
    ld = flags(dids, edges["non_substrate_of"], 10, offset=5, out=ld)
    cols = [f"inh_{g}" for g in CYP5] + [f"sub_{g}" for g in CYP5] + [f"nonsub_{g}" for g in CYP5]
    return lc, ld, cols
