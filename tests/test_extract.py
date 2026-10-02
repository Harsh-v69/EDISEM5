import json

import pytest

from ayurveda_kg.rag.extract import RELATIONS, build_prompt, cooccurrence_edges, parse_triples, run_extraction
from ayurveda_kg.rag.lexicon import Lexicon

SCOPE = {"herbs": [{"imppat_name": "Curcuma longa", "aliases": ["Haridra", "Turmeric"], "common": "turmeric"},
                   {"imppat_name": "Piper nigrum", "aliases": ["Maricha", "Black pepper"], "common": "black pepper"}], "drugs": []}
EXTRA = {"extra_herb_aliases": {}, "doshas": {"kapha": ["kapha", "phlegm"]}, "conditions": {"cough": ["cough"], "skin disease": ["skin disease", "kushtha"]}}
LEX = Lexicon.from_config(SCOPE, EXTRA)
PASSAGE = "Haridra applied as a paste cures skin disease. Maricha pacifies phlegm and relieves cough."


def raw(*triples):
    return json.dumps({"triples": [{"subject": s, "relation": r, "object": o} for s, r, o in triples]})


def test_prompt_contains_the_passage_the_allowed_relations_and_the_grounding_rule():
    p = build_prompt(PASSAGE)
    assert PASSAGE in p and all(r in p for r in RELATIONS) and "exactly as written" in p


def test_grounded_linked_triples_are_accepted_with_entity_ids():
    acc, rej = parse_triples(raw(("Haridra", "treats", "skin disease"), ("Maricha", "pacifies", "phlegm")), PASSAGE, LEX)
    assert [(t["s"], t["r"], t["o"]) for t in acc] == [("herb:Curcuma longa", "treats", "condition:skin disease"),
                                                       ("herb:Piper nigrum", "pacifies", "dosha:kapha")]
    assert rej == []


def test_hallucinated_entity_not_in_the_passage_is_rejected_even_if_the_lexicon_knows_it():
    acc, rej = parse_triples(raw(("Turmeric", "treats", "cough")), PASSAGE, LEX)           # 'Turmeric' never appears in PASSAGE
    assert acc == [] and rej[0]["reason"] == "subject not in passage"


def test_bad_relation_unknown_entity_and_malformed_output_are_rejected_with_reasons():
    acc, rej = parse_triples(raw(("Haridra", "promotes", "cough")), PASSAGE, LEX)
    assert acc == [] and rej[0]["reason"] == "relation not allowed"
    acc, rej = parse_triples(raw(("Maricha", "treats", "relieves")), PASSAGE, LEX)         # 'relieves' is in the passage but is not a lexicon entity
    assert acc == [] and rej[0]["reason"] == "object not a known entity"
    for bad in ["not json at all", "{}", json.dumps({"triples": "nope"}), json.dumps({"triples": [{"subject": "x"}]})]:
        acc, rej = parse_triples(bad, PASSAGE, LEX)
        assert acc == []


def test_duplicates_are_removed_and_relation_case_is_normalised():
    acc, _ = parse_triples(raw(("Haridra", "Treats", "skin disease"), ("haridra", "treats", "Skin Disease")), PASSAGE, LEX)
    assert len(acc) == 1 and acc[0]["r"] == "treats"


def test_cooccurrence_edges_count_chunks_apply_min_count_and_keep_provenance():
    ents = {"c1": ["herb:A", "condition:x", "dosha:vata"], "c2": ["herb:A", "condition:x"], "c3": ["herb:B", "condition:x"], "c4": ["condition:x", "dosha:vata"]}
    e = {(r["a"], r["b"]): r for r in cooccurrence_edges(ents, min_count=2)}
    assert set(e) == {("herb:A", "condition:x")}                                           # only pairs seen in >= 2 chunks; herb-dosha count is 1
    assert e[("herb:A", "condition:x")]["count"] == 2 and e[("herb:A", "condition:x")]["chunks"] == ["c1", "c2"]
    assert {(r["a"], r["b"]) for r in cooccurrence_edges(ents, min_count=1)} >= {("herb:A", "dosha:vata"), ("herb:B", "condition:x")}


def test_run_extraction_is_resumable_retries_errored_chunks_and_never_crashes_on_a_client_error(tmp_path):
    chunks = [{"id": "c1", "text": PASSAGE}, {"id": "c2", "text": PASSAGE}, {"id": "c3", "text": PASSAGE}]
    calls = []

    def client(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("model hiccup")                       # e.g. the local model server was down for a moment
        return raw(("Haridra", "treats", "skin disease"))

    out = tmp_path / "ex.jsonl"
    stats = run_extraction(chunks, LEX, client, out, limit=2)
    assert stats["done"] == 2 and stats["errors"] == 1 and len(calls) == 2           # c1 ok, c2 logged as an error, nothing crashed
    stats = run_extraction(chunks, LEX, client, out, limit=None)                      # resume: c1 is skipped, c2 is RETRIED, c3 is new
    assert len(calls) == 4 and stats["errors"] == 0 and stats["done"] == 3
    final = {}
    for l in out.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        final[r["chunk"]] = r                                                           # last row per chunk wins
    assert set(final) == {"c1", "c2", "c3"} and not any(r["error"] for r in final.values())
    assert stats["accepted"] == 3                                                       # counted once per chunk, not once per attempt


def test_reduce_verbs_map_to_treats_for_conditions_and_pacifies_for_doshas_with_the_raw_verb_kept():
    p = "Haridra allays cough and Maricha subdues phlegm and Haridra aggravates phlegm."
    acc, rej = parse_triples(raw(("Haridra", "allays", "cough"), ("Maricha", "subdues", "phlegm"), ("Haridra", "aggravates", "phlegm")), p, LEX)
    assert [(t["r"], t["o"], t["relation_raw"]) for t in acc] == [("treats", "condition:cough", "allays"), ("pacifies", "dosha:kapha", "subdues"),
                                                                    ("aggravates", "dosha:kapha", "aggravates")]
    assert rej == []


def test_unmapped_verbs_stay_rejected():
    for verb in ["is", "promotes", "contains", "causes"]:
        acc, rej = parse_triples(raw(("Haridra", verb, "cough")), PASSAGE + " cough", LEX)
        assert acc == [] and rej[0]["reason"] == "relation not allowed", verb


def test_focus_terms_are_listed_in_the_prompt_as_the_only_allowed_subjects():
    p = build_prompt(PASSAGE, focus_terms=["Haridra", "Maricha"])
    assert "Haridra" in p.split("PASSAGE:")[0] and "only" in p.lower() and "Maricha" in p.split("PASSAGE:")[0]
    assert "only" not in build_prompt(PASSAGE).split("PASSAGE:")[0].lower().replace("json only", "")


def test_run_extraction_passes_focus_terms_per_chunk_to_the_prompt(tmp_path):
    seen = []
    run_extraction([{"id": "c1", "text": PASSAGE}], LEX, lambda pr: (seen.append(pr), raw())[1], tmp_path / "e.jsonl",
                   focus_fn=lambda c: ["Haridra"])
    assert "Haridra" in seen[0].split("PASSAGE:")[0]
