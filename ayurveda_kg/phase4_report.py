"""Assemble docs/phase4_results.md from data/processed/phase4 (results.csv + out-of-fold predictions for every split/model)."""
import re
from pathlib import Path

import pandas as pd

from ayurveda_kg import curated
from ayurveda_kg.evaluate import grouped_auroc
from ayurveda_kg.kg import load_tables
from ayurveda_kg.labels import gold_vs_silver, herb_drug_scores
from ayurveda_kg.phase4 import OUT, gold_model_scores, make_phase4_report

PRED = re.compile(r"(?P<split>cold_compound|cold_herb|cold_drug|random_pair)_(?P<model>[a-z_]+)_s(?P<seed>\d+)_f(?P<fold>\d+)\.parquet")


def load_preds(pred_dir=OUT / "preds") -> pd.DataFrame:
    """All out-of-fold predictions; seeds are averaged per (split, model, fold, compound, drug)."""
    frames = []
    for f in sorted(Path(pred_dir).glob("*.parquet")):
        m = PRED.fullmatch(f.name)
        frames.append(pd.read_parquet(f).assign(split=m["split"], model=m["model"], fold=int(m["fold"])))
    d = pd.concat(frames, ignore_index=True)
    return d.groupby(["split", "model", "fold", "compound", "drug"], as_index=False).agg(label=("label", "first"), p=("p", "mean"))


def grouped_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, model), g in preds.groupby(["split", "model"]):
        g = g.assign(by_drug=g["fold"].astype(str) + "|" + g["drug"], by_cpd=g["fold"].astype(str) + "|" + g["compound"])
        wd, nd = grouped_auroc(g, "by_drug")
        wc, nc = grouped_auroc(g, "by_cpd")
        rows.append({"split": split, "model": model, "within_drug_auroc": wd, "n_drugs": nd, "within_compound_auroc": wc, "n_compounds": nc})
    return pd.DataFrame(rows)


def build(report_path="docs/phase4_results.md"):
    res = pd.read_csv(OUT / "results.csv")
    _, edges = load_tables("data/processed/kg")
    labels = pd.read_parquet("data/processed/labels/silver.parquet")
    silver = gold_vs_silver(curated.load_gold(), herb_drug_scores(labels, edges["contains"]))
    preds = load_preds()
    parts = []
    for model in ("rf", "gnn"):
        p = preds[(preds["split"] == "cold_herb") & (preds["model"] == model)]
        if len(p):
            parts.append(gold_model_scores(p[["compound", "drug", "p"]], edges["contains"], silver).assign(model=model))
    gold = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    Path(report_path).write_text(make_phase4_report(res, gold, grouped_table(preds)), encoding="utf-8")
    return res, gold


if __name__ == "__main__":
    build()
    print("wrote docs/phase4_results.md")
