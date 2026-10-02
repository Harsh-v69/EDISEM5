"""Run grounded LLM triple extraction on a deterministic sample of herb+condition passages (resumable, background)."""
import sys

from ayurveda_kg.rag.build_index import OUT, load_chunks
from ayurveda_kg.rag.extract import ollama_client, run_extraction
from ayurveda_kg.rag.lexicon import load_lexicon

N_SAMPLE = 480  # covers all herb+condition candidate passages (476)


def candidates(chunks, lex):
    return [c for c in chunks if (e := lex.entities(c["text"])) and any(x.startswith("herb:") for x in e)
            and any(x.startswith("condition:") for x in e)]


def sample(cands, n=N_SAMPLE):
    """Evenly strided, deterministic sample over the id-sorted candidates (covers all texts, no randomness)."""
    cands = sorted(cands, key=lambda c: c["id"])
    step = max(1, len(cands) // n)
    return cands[::step][:n]


if __name__ == "__main__":
    lex, chunks = load_lexicon(), load_chunks()
    todo = sample(candidates(chunks, lex))
    print(f"{len(todo)} chunks to extract", flush=True)
    focus = lambda c: sorted({m["surface"] for m in lex.link(c["text"]) if m["type"] == "Herb"})     # scoped herbs actually in the passage
    print(run_extraction(todo, lex, ollama_client(), OUT / "triples.jsonl", focus_fn=focus), flush=True)
