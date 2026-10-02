import hashlib

import numpy as np
import pytest

from ayurveda_kg.rag.index import DenseIndex
from ayurveda_kg.rag.metrics import mrr, recall_at_k


def fake_embed(texts, dim=128):
    """Deterministic hashing bag-of-words embedder: stands in for the real model so tests need no download."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return out


CHUNKS = [{"id": f"c{i}", "text": t} for i, t in enumerate([
    "pippali long pepper is used for cough and asthma",
    "haritaki is a purgative used in disorders of the intestinal canal",
    "turmeric haridra is applied to wounds and used in skin disease",
    "the wine asava is taken after the meal as an after drink",
    "ginger shunthi relieves indigestion and nausea",
])]


def test_search_returns_best_match_first_with_scores_in_descending_order_and_normalised():
    idx = DenseIndex.build(CHUNKS, fake_embed)
    hits = idx.search(fake_embed(["which herb treats cough and asthma pippali"])[0], k=3)
    assert hits[0][0] == "c0"
    assert [h[1] for h in hits] == sorted([h[1] for h in hits], reverse=True)
    assert all(-1e-6 <= h[1] <= 1 + 1e-6 for h in hits)                    # cosine similarity of unit vectors


def test_every_chunk_retrieves_itself_and_k_larger_than_corpus_is_safe():
    idx = DenseIndex.build(CHUNKS, fake_embed)
    for c in CHUNKS:
        assert idx.search(fake_embed([c["text"]])[0], k=1)[0][0] == c["id"]
    assert len(idx.search(fake_embed(["anything"])[0], k=99)) == len(CHUNKS)


def test_save_and_load_roundtrip_preserves_ids_and_results(tmp_path):
    idx = DenseIndex.build(CHUNKS, fake_embed)
    idx.save(tmp_path / "idx.npz")
    back = DenseIndex.load(tmp_path / "idx.npz")
    q = fake_embed(["ginger nausea"])[0]
    assert back.ids == idx.ids and back.search(q, 3) == idx.search(q, 3)


def test_zero_vector_query_does_not_crash_or_produce_nan():
    idx = DenseIndex.build(CHUNKS, fake_embed)
    hits = idx.search(np.zeros(128, dtype=np.float32), k=2)
    assert len(hits) == 2 and all(np.isfinite(h[1]) for h in hits)


def test_recall_at_k_and_mrr():
    ranked = {"q1": ["a", "b", "c"], "q2": ["x", "gold2", "y"], "q3": ["m", "n", "o"]}
    gold = {"q1": {"a"}, "q2": {"gold2"}, "q3": {"zzz"}}
    assert recall_at_k(ranked, gold, 1) == pytest.approx(1 / 3) and recall_at_k(ranked, gold, 3) == pytest.approx(2 / 3)
    assert mrr(ranked, gold) == pytest.approx((1 + 0.5 + 0) / 3)
    assert recall_at_k({}, {}, 5) != recall_at_k({}, {}, 5)                  # empty input is undefined (NaN), not a fake 0 or 1
