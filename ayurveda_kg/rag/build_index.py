"""Build data/processed/rag/{chunks.jsonl,index.npz} from the cached public-domain texts."""
import json
import time
from pathlib import Path

from ayurveda_kg.rag.corpus import make_chunks
from ayurveda_kg.rag.index import DenseIndex, sentence_embedder

SOURCES = {"charaka_kaviratna": ("charaka", "Charaka Samhita (Kaviratna trans.)"),
           "sushruta_v1": ("sushruta1", "Sushruta Samhita vol. 1 (Bhishagratna trans.)"),
           "sushruta_v2": ("sushruta2", "Sushruta Samhita vol. 2 (Bhishagratna trans.)"),
           "sushruta_v3": ("sushruta3", "Sushruta Samhita vol. 3 (Bhishagratna trans.)")}
OUT = Path("data/processed/rag")


def load_chunks(path=OUT / "chunks.jsonl") -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def main():
    chunks = []
    for f, (key, label) in SOURCES.items():
        chunks += make_chunks(Path(f"data/raw/texts/{f}.txt").read_text(encoding="utf-8", errors="ignore"), key, label)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    t = time.time()
    DenseIndex.build(chunks, sentence_embedder()).save(OUT / "index.npz")
    print(f"{len(chunks)} chunks indexed in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
