"""Corpus cleaning and chunking for the classical-text GraphRAG. Input is raw archive.org OCR (public-domain translations)."""
import re

HEADER = re.compile(r"[A-Z][A-Z\s.\-]*")                 # all-caps running headers like 'CHARAKA-SAMHITA.'
OK_TOKEN = re.compile(r"[A-Za-z0-9.,;:()'\-—\"’“”\[\]?!]+")


def _bad_token(tok: str) -> bool:
    if not OK_TOKEN.fullmatch(tok):
        return True                                       # stray symbols such as ^ * »
    if re.search(r"[a-z][A-Z]", tok):
        return True                                       # internal capital after a lowercase letter: OCR noise
    return len(tok) >= 3 and not re.search(r"[aeiouyAEIOUY]", tok)


def _garbage(line: str) -> bool:
    toks = line.split()
    return bool(toks) and sum(_bad_token(t) for t in toks) / len(toks) >= 0.3


def clean_text(raw: str) -> str:
    """Collapse spaces, drop running headers / bare page numbers / OCR garbage lines, re-join line-break hyphenation.
    Blank lines are kept as paragraph separators."""
    lines = [re.sub(r"\s+", " ", l).strip() for l in raw.replace("\r", "").split("\n")]
    out = []
    for l in lines:
        if not l:
            out.append("")
        elif re.fullmatch(r"\d{1,4}", l):
            continue                                       # bare page number
        elif HEADER.fullmatch(l) and len(l.split()) <= 4:
            continue                                       # running header (short, all caps)
        elif _garbage(l):
            continue
        else:
            out.append(l)
    merged = []
    for l in out:
        if merged and merged[-1] and re.search(r"[A-Za-z¬]-?$", merged[-1]) and (merged[-1].endswith("¬") or merged[-1].endswith("-")) \
                and l and l[0].islower():
            merged[-1] = merged[-1].rstrip("¬-") + l   # 'carda¬' + 'moms' -> 'cardamoms'
        else:
            merged.append(l)
    return "\n".join(merged)


def paragraphs(clean: str, min_words=5, short_words=12) -> list[str]:
    """Blank-line separated paragraphs, lines re-flowed. Consecutive short paragraphs (footnotes such as 'Long pepper.') are merged
    so plant identifications are kept instead of dropped."""
    raw = [" ".join(p.split("\n")).strip() for p in re.split(r"\n\s*\n", clean)]
    merged = []
    for p in raw:
        if not p:
            continue
        if merged and len(p.split()) < short_words and len(merged[-1].split()) < short_words:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return [p for p in merged if len(p.split()) >= min_words]


def chunk_paragraphs(paras, target_words=170, max_words=260, overlap_words=40) -> list[dict]:
    """Greedy paragraph packing. A chunk is flushed once it reaches target_words or the next paragraph would exceed max_words;
    the trailing paragraph is carried into the next chunk as overlap when short enough. A paragraph longer than max_words is split."""
    chunks, cur, words = [], [], 0

    def flush():
        nonlocal cur, words
        if cur:
            chunks.append({"text": "\n".join(p for _, p in cur), "n_words": words, "first_para": cur[0][0], "last_para": cur[-1][0]})
            tail = cur[-1]
            cur, words = ([tail], len(tail[1].split())) if overlap_words and len(tail[1].split()) <= overlap_words else ([], 0)

    for i, p in enumerate(paras):
        w = len(p.split())
        if w > max_words:                                  # huge paragraph: own windows, nothing lost
            flush()
            cur, words = [], 0
            toks = p.split()
            for s in range(0, len(toks), max_words):
                piece = toks[s:s + max_words]
                chunks.append({"text": " ".join(piece), "n_words": len(piece), "first_para": i, "last_para": i})
            continue
        if cur and words + w > max_words:
            flush()
        cur.append((i, p))
        words += w
        if words >= target_words:
            flush()
    if cur and (not chunks or cur[-1][0] > chunks[-1]["last_para"]):    # leftover that is not only overlap already emitted
        chunks.append({"text": "\n".join(p for _, p in cur), "n_words": words, "first_para": cur[0][0], "last_para": cur[-1][0]})
    return chunks


def make_chunks(raw_text: str, key: str, label: str, min_chunk_words=15, **kw) -> list[dict]:
    chunks = [c for c in chunk_paragraphs(paragraphs(clean_text(raw_text)), **kw) if c["n_words"] >= min_chunk_words]
    return [{"id": f"{key}-{i:05d}", "source": label, **c} for i, c in enumerate(chunks)]
