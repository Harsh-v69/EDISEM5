"""Evaluation for the GraphRAG system. Three layers, each honest about what it proves:
(1) synthetic retrieval questions (gold = the passage the question was written from; lexically easier than real questions),
(2) KG-grounded multi-hop questions with programmatically computed gold answers (graph advantage is by construction),
(3) an expert question template for a domain advisor (NOT claimed until an expert fills it)."""
import csv
import json
import random
import re
from pathlib import Path

import pandas as pd

from ayurveda_kg.rag.metrics import mrr, recall_at_k


def _strip_citations(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", " ", text)


def kg_questions(risk: pd.DataFrame, scope: dict, pairs, top_k=3, include_top_herbs=False) -> list[dict]:
    """Questions whose gold answers come straight from the project's herb-drug risk table."""
    common = {h["imppat_name"]: (h.get("common") or h["imppat_name"]) for h in scope["herbs"]}
    qs = []
    for herb in sorted(risk["herb"].unique()):
        top = risk[risk["herb"] == herb].sort_values(["risk_rf", "drug"], ascending=[False, True]).head(top_k)["drug"].tolist()
        qs.append({"id": f"KG-top-{herb.replace(' ', '_')}", "type": "top_drugs", "herb": herb, "gold": top,
                   "question": f"According to the project's research risk scores, which {top_k} drugs have the highest predicted interaction risk with {common[herb]}?"})
    for i, (herb, drug) in enumerate(pairs):
        row = risk[(risk["herb"] == herb) & (risk["drug"] == drug)]
        if len(row):
            qs.append({"id": f"KG-pair-{i}", "type": "pair_score", "herb": herb, "drug": drug, "gold": float(row["risk_rf"].iloc[0]),
                       "question": f"What is the research risk score for {common[herb]} with {drug}?"})
    if include_top_herbs:
        for drug in sorted(risk["drug"].unique()):
            top = risk[risk["drug"] == drug].sort_values(["risk_rf", "herb"], ascending=[False, True]).head(top_k)["herb"].tolist()
            qs.append({"id": f"KG-herbs-{drug}", "type": "top_herbs", "drug": drug, "gold": top,
                       "question": f"According to the project's research risk scores, which {top_k} herbs have the highest predicted interaction risk with {drug}?"})
    return qs


def score_top_drugs(answer: str, gold: list[str]) -> float:
    a = _strip_citations(answer).lower()
    hit = [bool(re.search(r"(?<![a-z])" + re.escape(g.lower()) + r"(?![a-z])", a)) for g in gold]
    return sum(hit) / len(gold) if gold else float("nan")


def score_pair_score(answer: str, gold: float, tol=0.03) -> bool:
    nums = [float(x) for x in re.findall(r"\d*\.\d+", _strip_citations(answer))]
    return any(abs(x - gold) <= tol for x in nums)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def too_verbatim(question: str, passage: str, n=6) -> bool:
    """True if the question contains a run of n consecutive words copied from the passage."""
    q, p = _tokens(question), " ".join(_tokens(passage))
    return any(" ".join(q[i:i + n]) in p for i in range(len(q) - n + 1))


def parse_question(raw: str):
    try:
        q = json.loads(raw).get("question")
    except (TypeError, ValueError, AttributeError):
        return None
    return q.strip() if isinstance(q, str) and len(q.split()) >= 6 else None


def make_retrieval_questions(chunks: list[dict], client, n=60, seed=0) -> list[dict]:
    """Ask the local LLM to write one question per sampled passage; drop short or near-verbatim ones."""
    rng = random.Random(seed)
    pool = list(chunks)
    rng.shuffle(pool)
    out = []
    for c in pool:
        if len(out) >= n:
            break
        prompt = ("Write ONE question that the passage below answers, as a student of Ayurveda might ask it. "
                  "Do not copy phrases from the passage. Answer with JSON only: {\"question\": \"...\"}\n\nPASSAGE:\n" + c["text"])
        try:
            q = parse_question(client(prompt))
        except Exception:
            q = None
        if q and not too_verbatim(q, c["text"]):
            out.append({"id": f"RQ-{len(out):03d}", "question": q, "gold": [c["id"]]})
    return out


def triple_questions(rows, chunks: dict, scope: dict, n=60) -> list[dict]:
    """Retrieval questions that test the vocabulary gap: they name the herb by its English common name while the supporting passage
    (found by grounded triple extraction) does not use that name or the Latin name. Gold = the passages supporting the triple."""
    common = {f"herb:{h['imppat_name']}": ((h.get("common") or h["imppat_name"]).lower(), h["imppat_name"].lower()) for h in scope["herbs"]}
    support = {}
    for r in rows:
        text = chunks[r["chunk"]]["text"].lower() if r["chunk"] in chunks else ""
        for t in r.get("accepted", []):
            if t["r"] != "treats" or t["s"] not in common or not t["o"].startswith("condition:"):
                continue
            eng, latin = common[t["s"]]
            if re.search(r"(?<![a-z])" + re.escape(eng) + r"(?![a-z])", text) or latin in text:
                continue                                        # the passage already uses the English/Latin name: no vocabulary gap
            support.setdefault((t["s"], t["o"]), []).append(r["chunk"])
    keys = sorted(support, key=lambda k: (-len(support[k]), k))[:n]
    return [{"id": f"TQ-{i:03d}", "herb": h, "condition": c,
             "question": f"What does the classical text say about {common[h][0]} for {c.split(':', 1)[1]}?",
             "gold": list(dict.fromkeys(support[(h, c)]))} for i, (h, c) in enumerate(keys)]


def retrieval_eval(questions: list[dict], rankers: dict, ks=(5, 10)) -> pd.DataFrame:
    """rankers: {config name: fn(question text) -> ranked chunk ids}. Gold = the source passage."""
    gold = {q["id"]: set(q["gold"]) for q in questions}
    rows = []
    for name, fn in rankers.items():
        ranked = {q["id"]: list(fn(q["question"])) for q in questions}
        rows.append({"config": name, **{f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}, "mrr": mrr(ranked, gold), "n": len(questions)})
    return pd.DataFrame(rows)


def write_expert_template(path, questions: list[dict]):
    """CSV for a domain advisor to fill in. Nothing in it is evidence until an expert verifies it."""
    cols = ["question_id", "question", "expected_answer", "gold_source_ids", "status", "rater", "notes"]
    with open(Path(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for q in questions:
            w.writerow({"question_id": q["id"], "question": q["question"], "expected_answer": "", "gold_source_ids": "",
                        "status": "DRAFT - needs expert verification", "rater": "", "notes": ""})
