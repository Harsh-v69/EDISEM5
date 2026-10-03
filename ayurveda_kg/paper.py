"""Paper drafts are rendered from templates (docs/paper/*.md.tmpl) so every number comes from the project's own result files and can never be
mistyped. Rendering REFUSES to proceed if any placeholder has no value. Usage: python -m ayurveda_kg.paper"""
import json
import re
from pathlib import Path

import pandas as pd

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)(?:\|([A-Za-z0-9]+))?\s*\}\}")
FILTERS = {"pct": lambda v: f"{v * 100:.0f}%", "pct1": lambda v: f"{v * 100:.1f}%", "f2": lambda v: f"{v:.2f}", "f3": lambda v: f"{v:.3f}",
           "int": lambda v: f"{int(round(v)):,}"}


def placeholders(template: str) -> set:
    return {m.group(1) for m in PLACEHOLDER.finditer(template)}


def render(template: str, numbers: dict) -> str:
    missing = sorted(k for k in placeholders(template) if numbers.get(k) is None)
    if missing:
        raise KeyError("no value for placeholder(s): " + ", ".join(missing))
    for _, f in PLACEHOLDER.findall(template):
        if f and f not in FILTERS:
            raise ValueError(f"unknown filter: {f}")

    def sub(m):
        v, f = numbers[m.group(1)], m.group(2)
        return FILTERS[f](v) if f else str(v)
    return PLACEHOLDER.sub(sub, template)


def kg_facts(nodes: dict, edges: dict) -> dict:
    f = {"kg_herbs": len(nodes["Herb"]), "kg_compounds": len(nodes["Compound"]), "kg_drugs": len(nodes["Drug"]), "kg_targets": len(nodes["Target"])}
    f.update({f"kg_edges_{t}": len(df) for t, df in edges.items()})
    return f


def label_facts(labels: pd.DataFrame, gold: pd.DataFrame) -> dict:
    c = labels["label"].value_counts()
    return {"lab_pairs": len(labels), "lab_pos": int(c.get(1, 0)), "lab_neg": int(c.get(0, 0)), "lab_unl": int(c.get(-1, 0)),
            "lab_pos_compounds": int(labels.loc[labels["label"] == 1, "compound"].nunique()),
            "gold_n": len(gold), "gold_neg": int((gold["label"] == 0).sum()), "gold_pk_pos": int(((gold["label"] == 1) & (gold["mechanism"] == "PK")).sum())}


def phase4_facts(res: pd.DataFrame, grouped: pd.DataFrame) -> dict:
    from ayurveda_kg.phase4 import summarize
    f = {}
    for r in summarize(res).itertuples():
        k = f"p4_{r.split}_{r.model}"
        f[f"{k}_auroc"], f[f"{k}_auroc_std"], f[f"{k}_auroc_min"], f[f"{k}_auroc_max"], f[f"{k}_auprc"] = (
            float(r.auroc_mean), float(r.auroc_std), float(r.auroc_min), float(r.auroc_max), float(r.auprc_mean))
    for r in grouped.itertuples():
        f[f"p4_{r.split}_{r.model}_within_drug"], f[f"p4_{r.split}_{r.model}_within_compound"] = float(r.within_drug_auroc), float(r.within_compound_auroc)
    for sp in ("cold_compound", "cold_herb", "cold_drug", "random_pair"):
        if f"p4_{sp}_rf_leaky_auroc" in f and f"p4_{sp}_rf_auroc" in f:
            f[f"p4_leak_{sp}"] = f[f"p4_{sp}_rf_leaky_auroc"] - f[f"p4_{sp}_rf_auroc"]
        if f"p4_{sp}_gnn_auroc" in f and f"p4_{sp}_mlp_auroc" in f:
            f[f"p4_gnn_minus_mlp_{sp}"] = f[f"p4_{sp}_gnn_auroc"] - f[f"p4_{sp}_mlp_auroc"]
    return f


def gather_numbers() -> dict:
    """Every number the drafts use, from the real stored results (needs the local processed data)."""
    from ayurveda_kg import curated
    from ayurveda_kg.kg import load_tables
    from ayurveda_kg.phase4 import OUT as P4
    from ayurveda_kg.phase4_report import grouped_table, load_preds
    nodes, edges = load_tables("data/processed/kg")
    f = kg_facts(nodes, edges)
    f.update(label_facts(pd.read_parquet("data/processed/labels/silver.parquet"), curated.load_gold()))
    f.update(phase4_facts(pd.read_csv(P4 / "results.csv"), grouped_table(load_preds())))
    for tag, path in (("p5", "data/processed/phase5/facts.json"), ("p6", "data/processed/rag/facts.json")):
        f.update({f"{tag}_{k}": v for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()})
    return f


def main(out_dir="docs/paper"):
    nums = gather_numbers()
    for t in sorted(Path(out_dir).glob("*.md.tmpl")):
        out = t.with_suffix("")                                     # paper1_draft.md.tmpl -> paper1_draft.md
        out.write_text(render(t.read_text(encoding="utf-8"), nums), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
