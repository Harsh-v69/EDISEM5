"""Text graph: herb-condition / herb-dosha edges from the classical texts, each with provenance (the passages that support it).
Sources: lexicon co-occurrence (weak, noisy) and grounded LLM triples (explicit relations)."""
import json
from pathlib import Path


class TextGraph:
    def __init__(self, edges: list[dict]):
        self.edges = edges
        self.adj = {}
        for e in edges:
            self.adj.setdefault(e["a"], []).append(e)
            self.adj.setdefault(e["b"], []).append(e)

    @classmethod
    def from_sources(cls, cooccurrence, triples):
        """cooccurrence: rows from extract.cooccurrence_edges. triples: extraction rows {chunk, accepted:[{s,r,o}]}."""
        merged = {}
        for r in cooccurrence:
            e = merged.setdefault((r["a"], r["b"], "co-mentioned"), {"a": r["a"], "b": r["b"], "relation": "co-mentioned", "chunks": []})
            e["chunks"] += [c for c in r["chunks"] if c not in e["chunks"]]
        for row in triples:
            for t in row.get("accepted", []):
                e = merged.setdefault((t["s"], t["o"], t["r"]), {"a": t["s"], "b": t["o"], "relation": t["r"], "chunks": []})
                if row["chunk"] not in e["chunks"]:
                    e["chunks"].append(row["chunk"])
        edges = [{**e, "weight": len(e["chunks"])} for e in merged.values()]
        return cls(sorted(edges, key=lambda e: (-e["weight"], e["a"], e["b"], e["relation"])))

    def neighbors(self, entity_id: str) -> list[dict]:
        """Edges touching the entity, strongest first; explicit LLM relations rank above plain co-mentions at equal weight."""
        return sorted(self.adj.get(entity_id, []), key=lambda e: (-e["weight"], e["relation"] == "co-mentioned", e["a"], e["b"]))

    def save(self, path):
        Path(path).write_text(json.dumps(self.edges, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
