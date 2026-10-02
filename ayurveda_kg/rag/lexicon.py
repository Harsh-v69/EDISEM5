"""Alias-aware entity linker: maps surface forms (English, Sanskrit, OCR-era spellings) to knowledge-graph entity ids.
Herbs and drugs come from config/scope.yaml; doshas and conditions from config/lexicon.yaml (needs domain-advisor review)."""
import re
import unicodedata
from pathlib import Path

import yaml


def _fold_char(ch: str) -> str:
    """Length-preserving fold: base letter of a diacritic char, lowercase, hyphen/underscore as space (spans stay valid)."""
    d = unicodedata.normalize("NFKD", ch)
    c = d[0].lower() if d else ch
    return " " if c in "-_" else c


def _fold(text: str) -> str:
    return "".join(_fold_char(c) for c in text)


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", _fold(s)).strip()


class Lexicon:
    def __init__(self, surface_to_entity: dict):
        self.map = surface_to_entity
        forms = sorted(self.map, key=len, reverse=True)                      # longest match first
        pat = "|".join(r"\s+".join(map(re.escape, f.split(" "))) for f in forms)
        self.rx = re.compile(r"(?<![a-z0-9])(?:" + pat + r")(?![a-z0-9])") if forms else None

    @classmethod
    def from_config(cls, scope: dict, extra: dict):
        claims = {}                                                           # surface -> set of entity ids
        ents = {}

        def add(ent_id, etype, name, forms):
            ents[ent_id] = {"id": ent_id, "type": etype, "name": name}
            for f in forms:
                s = normalise(f)
                if len(s) >= 3:
                    claims.setdefault(s, set()).add(ent_id)

        extra_h = extra.get("extra_herb_aliases", {})
        for h in scope["herbs"]:
            add(f"herb:{h['imppat_name']}", "Herb", h["imppat_name"],
                [h["imppat_name"], *h.get("aliases", []), h.get("common", ""), *extra_h.get(h["imppat_name"], [])])
        for d in scope["drugs"]:
            add(f"drug:{d['name']}", "Drug", d["name"], [d["name"]])
        for k, forms in extra.get("doshas", {}).items():
            add(f"dosha:{k}", "Dosha", k, [k, *forms])
        for k, forms in extra.get("conditions", {}).items():
            add(f"condition:{k}", "Condition", k, [k, *forms])
        # a surface form claimed by two different entities would attach evidence to the wrong one: drop it
        unique = {s: next(iter(v)) for s, v in claims.items() if len(v) == 1}
        return cls({s: ents[e] for s, e in unique.items()})

    def link(self, text: str) -> list[dict]:
        if self.rx is None:
            return []
        folded = _fold(text)
        out = []
        for m in self.rx.finditer(folded):
            ent = self.map[re.sub(r"\s+", " ", m.group(0))]
            out.append({**ent, "surface": text[m.start():m.end()], "start": m.start(), "end": m.end()})
        return out

    def entities(self, text: str) -> list[str]:
        """Unique entity ids in order of first mention."""
        return list(dict.fromkeys(m["id"] for m in self.link(text)))


def load_lexicon(scope_path="config/scope.yaml", lexicon_path="config/lexicon.yaml") -> Lexicon:
    scope = yaml.safe_load(Path(scope_path).read_text(encoding="utf-8"))
    extra = yaml.safe_load(Path(lexicon_path).read_text(encoding="utf-8"))
    return Lexicon.from_config(scope, extra)
