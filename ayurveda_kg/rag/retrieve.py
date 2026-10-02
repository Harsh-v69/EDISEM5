"""Hybrid retrieval: dense passages (optionally alias-expanded) + text-graph facts + knowledge-graph facts, fused with reciprocal-rank fusion.
Switches (use_alias / use_graph / use_kg) make every ablation a config change; all three off is plain RAG."""
from ayurveda_kg.rag.lexicon import normalise


def rrf(rankings: list[list[str]], k=60) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion: score(item) = sum over lists of 1 / (k + rank). Returns items by descending score (stable on ties)."""
    score = {}
    for ranking in rankings:
        for r, item in enumerate(ranking, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + r)
    return sorted(score.items(), key=lambda kv: -kv[1])


def expand_query(question: str, lexicon, max_aliases=4) -> str:
    """Append Sanskrit/alias forms of herbs named in the question so a query about 'turmeric' also finds passages that say 'Haridra'."""
    ents = [e for e in lexicon.entities(question) if e.startswith("herb:")]
    if not ents:
        return question
    q_norm = normalise(question)
    extra = []
    for e in ents:
        forms = sorted((s for s, ent in lexicon.map.items() if ent["id"] == e and s not in q_norm), key=lambda s: (len(s.split()), s))
        extra += forms[:max_aliases]
    return question + (" " + " ".join(dict.fromkeys(extra)) if extra else "")


class HybridRetriever:
    def __init__(self, index, embed_fn, chunks: dict, lexicon, text_graph, kg_facts, use_alias=True, use_graph=True, use_kg=True, k_dense=20):
        self.index, self.embed, self.chunks, self.lex = index, embed_fn, chunks, lexicon
        self.graph, self.kg = text_graph, kg_facts
        self.use_alias, self.use_graph, self.use_kg, self.k_dense = use_alias, use_graph, use_kg, k_dense

    def _graph_items(self, entities, max_graph):
        edges, seen = [], set()
        for ent in entities:
            for e in self.graph.neighbors(ent):
                key = (e["a"], e["b"], e["relation"])
                if key not in seen:
                    seen.add(key)
                    edges.append(e)
        edges = sorted(edges, key=lambda e: (-e["weight"], e["relation"] == "co-mentioned"))[:max_graph]
        name = lambda x: x.split(":", 1)[1]
        items = []
        for i, e in enumerate(edges, 1):
            if e["relation"] == "co-mentioned":
                text = (f"{name(e['a'])} is co-mentioned with {name(e['b'])} in {e['weight']} passage(s) of the classical texts "
                        "(co-occurrence only: this does not state that either affects the other).")
            else:
                text = f"{name(e['a'])} {e['relation']} {name(e['b'])} (stated in {e['weight']} passage(s) of the classical texts)."
            items.append({"id": f"G{i}", "kind": "G", "text": text, "chunks": e["chunks"][:3], "source": "text graph"})
        return items, [c for e in edges for c in e["chunks"]]

    def retrieve(self, question: str, k_passages=5, max_graph=4, max_kg=3) -> list[dict]:
        q_text = expand_query(question, self.lex) if self.use_alias else question
        dense = [i for i, _ in self.index.search(self.embed([q_text])[0], self.k_dense)]
        entities = self.lex.entities(question)
        rankings, g_items = [dense], []
        if self.use_graph:
            g_items, prov = self._graph_items(entities, max_graph)
            if prov:
                rankings.append(list(dict.fromkeys(prov)))
        top = [c for c, _ in rrf(rankings)[:k_passages]]
        items = [{"id": f"P{i}", "kind": "P", "chunk": c, "text": self.chunks[c]["text"], "source": self.chunks[c]["source"]}
                 for i, c in enumerate(top, 1)]
        items += g_items
        if self.use_kg:
            for i, f in enumerate(self.kg.facts(entities)[:max_kg], 1):
                items.append({"id": f"K{i}", "kind": "K", "text": f["text"], "source": f["source"]})
        return items
