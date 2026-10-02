"""Dense retrieval: brute-force cosine over a unit-normalised NumPy matrix. At ~7k chunks x 384 dims this needs no FAISS/Chroma.
# ponytail: brute force is fine to ~1M chunks; switch to FAISS only if the corpus grows far beyond the two texts."""
from pathlib import Path

import numpy as np


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1.0, n)                      # zero vectors stay zero (score 0), never NaN


class DenseIndex:
    def __init__(self, ids, matrix):
        self.ids = list(ids)
        self.matrix = _unit(matrix)

    @classmethod
    def build(cls, chunks, embed_fn, batch_size=256):
        texts = [c["text"] for c in chunks]
        parts = [embed_fn(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return cls([c["id"] for c in chunks], np.vstack(parts))

    def search(self, query_vec, k=10) -> list[tuple[str, float]]:
        scores = self.matrix @ _unit(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, path):
        np.savez_compressed(Path(path), ids=np.array(self.ids), matrix=self.matrix)

    @classmethod
    def load(cls, path):
        z = np.load(Path(path), allow_pickle=False)
        return cls(z["ids"].tolist(), z["matrix"])


def sentence_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
    """Real embedder (downloads the model on first use). Chunks over the model's 256-token limit are truncated."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return lambda texts: model.encode(list(texts), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
