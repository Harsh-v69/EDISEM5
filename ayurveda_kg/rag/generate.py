"""Citation-constrained answer generation and automatic faithfulness checks. The disclaimer is appended by code, never left to the model.
The checks are PROXIES (citation validity, lexical support): they catch invented citations and unsupported sentences, but do not replace
an expert judging correctness."""
import math
import re

NOT_ENOUGH = "The retrieved sources do not contain enough information to answer this question."
DISCLAIMER = "Research use only: this is not medical advice or a diagnosis; consult a qualified clinician."
STOP = set("this that with from have been were will would could should their there which these those into than then them they also such "
           "when where what while about after before because between both each other some only over very more most like used uses using".split())


def build_answer_prompt(question: str, items: list[dict], kind_order: str = None, strict_rules: bool = True) -> str:
    """kind_order e.g. "KGP" puts knowledge-graph facts first, then text-graph facts, then passages (ids are unchanged).
    strict_rules=False drops the two rules added after the first evaluation run (co-mention caution, safety-question rule); it exists only
    so the before/after comparison in the report can be reproduced."""
    if kind_order:
        items = sorted(items, key=lambda i: kind_order.index(i["kind"]) if i["kind"] in kind_order else len(kind_order))
    src = "\n".join(f"[{i['id']}] ({i['source']}) {i['text']}" for i in items) if items else "(no sources were retrieved)"
    extra = ("- A source that says two things are 'co-mentioned' only means they appear together in a passage; do not infer that one treats, balances or affects the other.\n"
             "- For interaction or safety questions: if a [K] source gives a research risk score, report that score and say it is a research hypothesis; never declare a combination safe or unsafe.\n"
             ) if strict_rules else ""
    return ("You answer questions about Ayurveda using ONLY the numbered sources below.\n"
            "Rules:\n"
            "- Use only information stated in the sources. Do not add outside knowledge.\n"
            "- After every claim, cite the source id in square brackets, for example [P1] or [K1].\n"
            f"- If the sources do not contain the answer, reply exactly: {NOT_ENOUGH}\n"
            + extra +
            "- Be concise (at most 120 words). Do not give medical advice or dosing.\n\n"
            f"SOURCES:\n{src}\n\nQUESTION: {question}\nANSWER:")


def extract_citations(text: str) -> list[str]:
    """Ids cited in square brackets, in order, deduplicated. Lists ([P2, K1]) and ranges ([G1-G4], [G1-4]) are expanded."""
    out = []
    for grp in re.findall(r"\[([A-Z]\d+(?:\s*[-,]\s*[A-Z]?\d+)*)\]", text):
        for tok in grp.split(","):
            tok = tok.strip()
            m = re.fullmatch(r"([A-Z])(\d+)\s*-\s*[A-Z]?(\d+)", tok)
            if m:
                a, b = int(m.group(2)), int(m.group(3))
                out += [f"{m.group(1)}{n}" for n in range(a, b + 1)] if a <= b else [f"{m.group(1)}{a}"]
            else:
                out.append(tok)
    return list(dict.fromkeys(out))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def check_citations(text: str, items: list[dict]) -> dict:
    ids = {i["id"] for i in items}
    cited = extract_citations(text)
    valid, invalid = [c for c in cited if c in ids], [c for c in cited if c not in ids]
    sents = _sentences(text)
    return {"valid": valid, "invalid": invalid, "n_sentences": len(sents), "n_uncited": sum(1 for s in sents if not extract_citations(s)),
            "citation_validity": len(valid) / len(cited) if cited else float("nan")}


def _content_words(s: str) -> set:
    s = re.sub(r"\[[^\]]*\]", " ", s.lower())
    return {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", s) if w not in STOP}


def sentence_support(sentence: str, texts: list[str]) -> float:
    """Fraction of the sentence's content words found in the given source texts. NaN when the sentence has no content words."""
    cw = _content_words(sentence)
    if not cw:
        return float("nan")
    src = set().union(*[_content_words(t) for t in texts]) if texts else set()
    return len(cw & src) / len(cw)


CLAIM = re.compile(r"\b(balanc\w*|treat\w*|cure\w*|heal\w*|relie\w*|pacif\w*|role in|benefici\w*|therap\w*|remed\w*|used (?:for|to|in))\b", re.I)


NEG = re.compile(r"\b(not|no|never|neither|nor|without|cannot)\b|n't", re.I)      # a sentence that denies the relation is a hedge, not an over-reading


def overclaim_flags(text: str, items: list[dict]) -> dict:
    """Sentences that cite ONLY co-occurrence facts: how many, and how many of them nevertheless state a therapeutic role (an over-reading)."""
    by_id = {i["id"]: i for i in items}
    n_co = n_over = 0
    for s in _sentences(text):
        cites = [c for c in extract_citations(s) if c in by_id]
        if cites and all(by_id[c]["kind"] == "G" and "co-mentioned" in by_id[c]["text"] for c in cites):
            n_co += 1
            body = re.sub(r"\[[^\]]*\]", " ", s)
            n_over += bool(CLAIM.search(body) and not NEG.search(body))
    return {"n_cocite": n_co, "n_overclaim": n_over}


def answer(question: str, items: list[dict], client, kind_order: str = None, strict_rules: bool = True) -> dict:
    """client(prompt) -> text. Never raises: a model failure is reported in `error`."""
    try:
        model_text, error = client(build_answer_prompt(question, items, kind_order, strict_rules)).strip(), ""
    except Exception as e:
        model_text, error = "", f"{type(e).__name__}: {e}"
    by_id = {i["id"]: i["text"] for i in items}
    per = []
    if model_text and NOT_ENOUGH not in model_text:
        for s in _sentences(model_text):
            cites = [c for c in extract_citations(s) if c in by_id]
            v = sentence_support(s, [by_id[c] for c in cites] if cites else list(by_id.values()))
            if not math.isnan(v):
                per.append(v)
    shown = model_text if model_text else "(the model was unavailable)"
    return {"text": f"{shown}\n\n{DISCLAIMER}", "model_text": model_text, "checks": check_citations(model_text, items),
            "support": {"rate": sum(v >= 0.5 for v in per) / len(per) if per else float("nan"), "n": len(per), "mean": sum(per) / len(per) if per else float("nan")},
            "refused": NOT_ENOUGH in model_text, "overclaim": overclaim_flags(model_text, items), "error": error}


def ollama_text_client(model="qwen3:8b", host="http://127.0.0.1:11434", timeout=900, num_predict=350):
    """Local Ollama chat returning plain text. Temperature 0, thinking off."""
    import requests

    def call(prompt: str) -> str:
        r = requests.post(f"{host}/api/chat", timeout=timeout, json={
            "model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 4096}})
        r.raise_for_status()
        return r.json()["message"]["content"]
    return call
