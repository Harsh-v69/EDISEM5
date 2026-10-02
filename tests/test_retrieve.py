import hashlib

import numpy as np
import pandas as pd
import pytest

from ayurveda_kg.rag.index import DenseIndex
from ayurveda_kg.rag.kgfacts import KGFacts
from ayurveda_kg.rag.lexicon import Lexicon
from ayurveda_kg.rag.retrieve import HybridRetriever, expand_query, rrf
from ayurveda_kg.rag.textgraph import TextGraph


def fake_embed(texts, dim=256):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").replace(",", " ").split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return out


SCOPE = {"herbs": [{"imppat_name": "Curcuma longa", "aliases": ["Turmeric", "Haridra"], "common": "turmeric"},
                   {"imppat_name": "Piper nigrum", "aliases": ["Black pepper", "Maricha"], "common": "black pepper"}],
         "drugs": [{"name": "phenytoin", "cls": "anticonvulsant"}]}
EXTRA = {"extra_herb_aliases": {}, "doshas": {"kapha": ["kapha", "phlegm"]}, "conditions": {"cough": ["cough"], "skin disease": ["skin disease"]}}
LEX = Lexicon.from_config(SCOPE, EXTRA)
CHUNKS = [{"id": "c0", "source": "Charaka", "text": "Haridra paste cures skin disease and wounds when applied."},
          {"id": "c1", "source": "Charaka", "text": "A decoction of Maricha relieves cough and phlegm in the chest."},
          {"id": "c2", "source": "Sushruta", "text": "The wine is taken after meals as an after drink for digestion."},
          {"id": "c3", "source": "Sushruta", "text": "Purgatives are used in disorders of the intestinal canal."},
          {"id": "c4", "source": "Sushruta", "text": "Skin disease is treated by pastes and washing the part."}]


def graph():
    return TextGraph.from_sources(
        cooccurrence=[{"a": "herb:Curcuma longa", "b": "condition:skin disease", "type": "herb-condition", "count": 3, "chunks": ["c0", "c4", "c2"]}],
        triples=[{"chunk": "c1", "accepted": [{"s": "herb:Piper nigrum", "r": "treats", "o": "condition:cough", "subject": "Maricha", "object": "cough"}]}])


RISK = pd.DataFrame([{"herb": "Piper nigrum", "drug": "phenytoin", "risk_rf": 0.40, "risk_silver": 0.48, "n_compounds": 100},
                     {"herb": "Curcuma longa", "drug": "phenytoin", "risk_rf": 0.20, "risk_silver": 0.10, "n_compounds": 80}])


def make(**kw):
    idx = DenseIndex.build(CHUNKS, fake_embed)
    return HybridRetriever(idx, fake_embed, {c["id"]: c for c in CHUNKS}, LEX, graph(), KGFacts(RISK), **kw)


def test_rrf_rewards_items_ranked_high_in_several_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [x for x, _ in fused][:2] == ["a", "b"] or [x for x, _ in fused][:2] == ["b", "a"]
    assert {x for x, _ in fused} == {"a", "b", "c", "d"}
    assert dict(fused)["a"] > dict(fused)["c"] and dict(fused)["b"] > dict(fused)["d"]


def test_expand_query_adds_sanskrit_aliases_of_linked_herbs_only():
    q = expand_query("How is turmeric used for wounds?", LEX)
    assert "haridra" in q.lower() and q.startswith("How is turmeric used for wounds?")
    assert expand_query("What is the wine taken after meals?", LEX) == "What is the wine taken after meals?"


def test_plain_mode_is_dense_only_and_returns_passage_items():
    r = make(use_alias=False, use_graph=False, use_kg=False)
    items = r.retrieve("A decoction of Maricha for cough", k_passages=3)
    assert all(i["kind"] == "P" for i in items) and items[0]["chunk"] == "c1"
    assert [i["id"] for i in items] == ["P1", "P2", "P3"] and len({i["id"] for i in items}) == 3


def test_alias_expansion_closes_the_vocabulary_gap():
    chunks = [{"id": "d0", "source": "S", "text": "The wine is taken after meals as a drink."},
              {"id": "d1", "source": "S", "text": "Haridra is the best remedy for the skin."}]            # the passage says Haridra, never turmeric
    idx = DenseIndex.build(chunks, fake_embed)
    def top(use_alias):
        r = HybridRetriever(idx, fake_embed, {c["id"]: c for c in chunks}, LEX, graph(), KGFacts(RISK),
                            use_alias=use_alias, use_graph=False, use_kg=False)
        return r.retrieve("Where is turmeric found?", k_passages=1)[0]["chunk"]
    assert top(False) == "d0" and top(True) == "d1"


def test_graph_items_and_provenance_boost_surface_a_passage_dense_ranks_low():
    r = make(use_alias=False, use_graph=True, use_kg=False)
    items = r.retrieve("Which remedies are linked with skin disease?", k_passages=3)
    g = [i for i in items if i["kind"] == "G"]
    assert g and "Curcuma longa" in g[0]["text"] and g[0]["id"] == "G1" and "c0" in g[0]["chunks"]
    assert "c0" in [i["chunk"] for i in items if i["kind"] == "P"]          # provenance chunk of the graph edge is retrieved as a passage


def test_kg_facts_are_added_for_herb_drug_pairs_with_the_caveat_and_unique_ids():
    r = make(use_alias=False, use_graph=False, use_kg=True)
    items = r.retrieve("Is black pepper safe with phenytoin?", k_passages=2)
    k = [i for i in items if i["kind"] == "K"]
    assert k and "Piper nigrum" in k[0]["text"] and "phenytoin" in k[0]["text"] and "0.40" in k[0]["text"]
    assert "not medical advice" in k[0]["text"].lower()
    assert len({i["id"] for i in items}) == len(items) and k[0]["id"].startswith("K")


def test_kg_facts_for_a_single_herb_list_the_highest_risk_drugs_and_unknown_entities_give_nothing():
    f = KGFacts(RISK)
    assert "phenytoin" in f.facts(["herb:Piper nigrum"])[0]["text"]
    assert f.facts(["dosha:kapha"]) == [] and f.facts([]) == []


def test_co_mention_graph_facts_state_that_they_are_co_occurrence_only():
    r = make(use_alias=False, use_graph=True, use_kg=False)
    g = [i for i in r.retrieve("Which remedies are linked with skin disease?", k_passages=2) if i["kind"] == "G"]
    assert g and "co-mentioned" in g[0]["text"] and "co-occurrence only" in g[0]["text"]
