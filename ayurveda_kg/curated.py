"""Hand-curated, cited data: the gold HDI set (evaluation only, never training) and a substrate supplement for known data gaps."""
import pandas as pd

from ayurveda_kg.resolve import alias_map

MECHANISMS = {"PK", "PD", "none", "unclear"}
EVIDENCE = {"human_crossover_pk", "human_patient_pk", "case_report", "review"}
CONFIDENCE = {"high", "medium", "low"}


def load_gold(path="data/gold/gold_hdi.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype={"pmid": str}).fillna("")


def validate_gold(df, scope) -> list[str]:
    herbs = {h["imppat_name"] for h in scope["herbs"]}
    drugs = {d["name"] for d in scope["drugs"]}
    out = []
    for i, r in df.iterrows():
        tag = f"row {i} ({r['herb']} + {r['drug']})"
        checks = [
            (r["herb"] not in herbs, "unknown herb"), (r["drug"] not in drugs, "unknown drug"),
            (r["label"] not in (0, 1), "label must be 0 or 1"), (r["mechanism"] not in MECHANISMS, "bad mechanism"),
            (r["evidence"] not in EVIDENCE, "bad evidence"), (r["confidence"] not in CONFIDENCE, "bad confidence"),
            (not str(r["pmid"]).isdigit(), "pmid must be numeric"),
            (not str(r["verification"]).strip(), "missing verification"), (not str(r["citation"]).strip(), "missing citation"),
            (r["label"] == 0 and r["mechanism"] != "none", "label 0 requires mechanism none"),
            (r["label"] == 1 and r["mechanism"] == "none", "label 1 cannot have mechanism none"),
        ]
        out += [f"{tag}: {msg}" for bad, msg in checks if bad]
    dup = df[df.duplicated(["herb", "drug", "pmid"], keep=False)]
    out += [f"duplicate herb/drug/pmid: {r.herb} + {r.drug} + {r.pmid}" for r in dup.drop_duplicates(["herb", "drug", "pmid"]).itertuples()]
    return out


def load_substrate_supplement(path, drugs) -> pd.DataFrame:
    """Cited drug->enzyme substrate facts that no open bulk source carried. An unknown drug is an error, never silently dropped."""
    amap = alias_map(drugs)
    rows = []
    for r in pd.read_csv(path).itertuples():
        d = amap.get(r.drug.strip().upper())
        if d is None:
            raise ValueError(f"substrate supplement names a drug not in scope: {r.drug}")
        rows.append({"src": d, "dst": f"gene:{r.enzyme.strip().upper()}", "source": "curated literature"})
    return pd.DataFrame(rows, columns=["src", "dst", "source"])
