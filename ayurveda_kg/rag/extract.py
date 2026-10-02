"""Text-graph construction. (1) Co-occurrence edges from the lexicon linker (cheap, noisy). (2) LLM triple extraction where a triple is
accepted ONLY if its entities literally appear in the source passage and resolve to lexicon entities (grounding check).
Precision of accepted triples is an expert-review item; the automatic check only removes hallucinated entities."""
import json
import re
from itertools import combinations
from pathlib import Path

from ayurveda_kg.rag.lexicon import Lexicon, _fold

RELATIONS = ("treats", "pacifies", "aggravates")
# verbs the model actually uses, mapped transparently (the raw verb is kept on every accepted triple)
REDUCE = {"treats", "cures", "relieves", "allays", "alleviates", "removes", "destroys", "checks", "heals", "pacifies", "subdues", "soothes", "reduces"}
INCREASE = {"aggravates", "increases", "excites", "provokes", "vitiates", "exacerbates"}


def build_prompt(passage: str, focus_terms=None) -> str:
    focus = ("- Use ONLY these names as subjects (they are the herbs of interest found in this passage): "
             + "; ".join(focus_terms) + ".\n") if focus_terms else ""
    return ("You read a passage from a classical Ayurvedic text (English translation, OCR text).\n"
            "Extract (subject, relation, object) triples the passage states explicitly.\n"
            f"Allowed relations: {', '.join(RELATIONS)}.\n"
            "- subject: a herb or drug name; object: a disease/condition or a dosha (wind/vata, bile/pitta, phlegm/kapha).\n"
            + focus +
            "- Copy subject and object exactly as written in the passage. Do not invent, translate or infer names.\n"
            "- If nothing is stated explicitly, return an empty list.\n"
            'Answer with JSON only: {"triples": [{"subject": "...", "relation": "...", "object": "..."}]}\n\n'
            f"PASSAGE:\n{passage}")


def _in_passage(surface: str, passage: str) -> bool:
    s = re.sub(r"\s+", " ", _fold(surface)).strip()
    if not s:
        return False
    pat = r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, s.split(" "))) + r"(?![a-z0-9])"
    return re.search(pat, _fold(passage)) is not None


def parse_triples(raw: str, passage: str, lexicon: Lexicon):
    """Returns (accepted, rejected). Each rejection carries a reason."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return [], [{"reason": "malformed output"}]
    items = data.get("triples") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return [], [{"reason": "no triples list"}]
    accepted, rejected, seen = [], [], set()
    ents = lambda surface: lexicon.entities(surface)
    for t in items:
        if not isinstance(t, dict) or not all(isinstance(t.get(k), str) for k in ("subject", "relation", "object")):
            rejected.append({"triple": t, "reason": "malformed triple"})
            continue
        s, r, o = t["subject"].strip(), t["relation"].strip().lower(), t["object"].strip()
        if r not in REDUCE and r not in INCREASE:
            rejected.append({"triple": t, "reason": "relation not allowed"})
        elif not _in_passage(s, passage):
            rejected.append({"triple": t, "reason": "subject not in passage"})
        elif not _in_passage(o, passage):
            rejected.append({"triple": t, "reason": "object not in passage"})
        else:
            sid = next((e for e in ents(s) if e.startswith(("herb:", "drug:"))), None)
            oid = next((e for e in ents(o) if e.startswith(("condition:", "dosha:"))), None)
            if sid is None:
                rejected.append({"triple": t, "reason": "subject not a known entity"})
            elif oid is None:
                rejected.append({"triple": t, "reason": "object not a known entity"})
            else:
                final = "aggravates" if r in INCREASE else ("treats" if oid.startswith("condition:") else "pacifies")
                if (sid, final, oid) not in seen:
                    seen.add((sid, final, oid))
                    accepted.append({"s": sid, "r": final, "o": oid, "subject": s, "object": o, "relation_raw": r})
    return accepted, rejected


def cooccurrence_edges(chunk_entities: dict, min_count=2) -> list[dict]:
    """herb-condition and herb-dosha pairs seen together in the same chunk; count = number of distinct chunks (provenance kept)."""
    pairs = {}
    for cid, ents in chunk_entities.items():
        herbs = [e for e in dict.fromkeys(ents) if e.startswith("herb:")]
        others = [e for e in dict.fromkeys(ents) if e.startswith(("condition:", "dosha:"))]
        for h in herbs:
            for o in others:
                pairs.setdefault((h, o), []).append(cid)
    return [{"a": a, "b": b, "type": "herb-" + b.split(":")[0], "count": len(c), "chunks": c}
            for (a, b), c in sorted(pairs.items()) if len(c) >= min_count]


def run_extraction(chunks, lexicon, client, out_path, limit=None, focus_fn=None) -> dict:
    """Resumable: chunks already present in out_path (including logged errors) are skipped. A client error is logged, never fatal."""
    out = Path(out_path)
    done = {json.loads(l)["chunk"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    new = 0
    with out.open("a", encoding="utf-8") as f:
        for c in chunks:
            if c["id"] in done:
                continue
            if limit is not None and new >= limit:
                break
            try:
                acc, rej = parse_triples(client(build_prompt(c["text"], focus_fn(c) if focus_fn else None)), c["text"], lexicon)
                row = {"chunk": c["id"], "accepted": acc, "rejected": rej, "error": ""}
            except Exception as e:                                       # a model hiccup must not kill a long background run
                row = {"chunk": c["id"], "accepted": [], "rejected": [], "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            new += 1
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    return {"done": len(rows), "errors": sum(1 for r in rows if r["error"]), "accepted": sum(len(r["accepted"]) for r in rows),
            "rejected": sum(len(r["rejected"]) for r in rows)}


def ollama_client(model="qwen3:8b", host="http://127.0.0.1:11434", timeout=600, num_predict=500):
    """Thin local Ollama chat client returning the model's text. Temperature 0, JSON mode, thinking off."""
    import requests

    def call(prompt: str) -> str:
        r = requests.post(f"{host}/api/chat", timeout=timeout, json={
            "model": model, "stream": False, "think": False, "format": "json",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 4096}})
        r.raise_for_status()
        return r.json()["message"]["content"]
    return call
