import pytest

from ayurveda_kg.rag.generate import (DISCLAIMER, NOT_ENOUGH, answer, build_answer_prompt, check_citations, extract_citations, sentence_support)

ITEMS = [{"id": "P1", "kind": "P", "text": "Haridra paste cures skin disease and wounds when applied.", "source": "Charaka"},
         {"id": "P2", "kind": "P", "text": "The wine is taken after meals as an after drink.", "source": "Sushruta"},
         {"id": "K1", "kind": "K", "text": "Research risk score for Piper nigrum with phenytoin: predicted interaction probability 0.40.", "source": "KG"}]


def test_prompt_lists_every_item_with_its_id_and_the_citation_and_refusal_rules():
    p = build_answer_prompt("What treats skin disease?", ITEMS)
    for it in ITEMS:
        assert f"[{it['id']}]" in p and it["text"] in p
    assert "What treats skin disease?" in p and NOT_ENOUGH in p and "only" in p.lower() and "[P1]" in p


def test_prompt_with_no_items_tells_the_model_there_is_no_context_and_still_allows_a_refusal():
    p = build_answer_prompt("anything", [])
    assert "no sources" in p.lower() and NOT_ENOUGH in p


def test_extract_citations_finds_ids_in_order_and_dedupes():
    assert extract_citations("Haridra cures skin disease [P1]. It is applied [P1][K1] and [G2, P2].") == ["P1", "K1", "G2", "P2"]
    assert extract_citations("no citations here") == []


def test_check_citations_separates_valid_from_invented_ids_and_flags_uncited_sentences():
    r = check_citations("Haridra cures skin disease [P1]. The wine helps digestion [P9]. It is a nice day.", ITEMS)
    assert r["valid"] == ["P1"] and r["invalid"] == ["P9"]
    assert r["n_sentences"] == 3 and r["n_uncited"] == 1
    assert r["citation_validity"] == pytest.approx(0.5)
    assert check_citations("nothing cited", ITEMS)["citation_validity"] != check_citations("nothing cited", ITEMS)["citation_validity"]   # NaN, not a fake 1.0


def test_sentence_support_is_high_for_grounded_sentences_and_low_for_invented_ones():
    grounded = sentence_support("Haridra paste cures skin disease.", [ITEMS[0]["text"]])
    invented = sentence_support("Ashwagandha builds muscle and boosts testosterone quickly.", [ITEMS[0]["text"]])
    assert grounded >= 0.8 and invented <= 0.2
    assert sentence_support("It is.", [ITEMS[0]["text"]]) != sentence_support("It is.", [ITEMS[0]["text"]])      # no content words -> NaN


def test_answer_calls_the_model_once_appends_the_disclaimer_and_reports_checks():
    calls = []
    out = answer("What treats skin disease?", ITEMS, lambda prompt: (calls.append(prompt), "Haridra paste cures skin disease [P1].")[1])
    assert len(calls) == 1 and out["text"].endswith(DISCLAIMER) and "Haridra paste cures skin disease [P1]." in out["text"]
    assert out["checks"]["valid"] == ["P1"] and out["checks"]["invalid"] == [] and out["support"]["rate"] >= 0.8


def test_the_disclaimer_is_added_even_when_the_model_refuses_or_fails():
    assert answer("q", ITEMS, lambda p: NOT_ENOUGH)["text"].endswith(DISCLAIMER)
    def boom(p):
        raise RuntimeError("down")
    out = answer("q", ITEMS, boom)
    assert out["error"] and out["text"].endswith(DISCLAIMER)


def test_kind_order_puts_kg_facts_before_graph_facts_before_passages_without_changing_ids():
    items = [{"id": "P1", "kind": "P", "text": "passage text", "source": "S"}, {"id": "G1", "kind": "G", "text": "graph fact", "source": "g"},
             {"id": "K1", "kind": "K", "text": "kg fact", "source": "k"}]
    default = build_answer_prompt("q", items).split("SOURCES:")[1]          # the rules text above also mentions [P1]/[K1] as examples
    kfirst = build_answer_prompt("q", items, kind_order="KGP").split("SOURCES:")[1]
    assert default.index("[P1]") < default.index("[G1]") < default.index("[K1]")           # unchanged default: retrieval order
    assert kfirst.index("[K1]") < kfirst.index("[G1]") < kfirst.index("[P1]")
    assert "[K1]" in kfirst and "kg fact" in kfirst


# ---- over-reading of co-occurrence facts, and the safety-question rule ----
from ayurveda_kg.rag.generate import overclaim_flags

COITEMS = [{"id": "G1", "kind": "G", "text": "turmeric is co-mentioned with pitta in 17 passages (co-occurrence only: this does not state that either affects the other).", "source": "g"},
           {"id": "G2", "kind": "G", "text": "Piper nigrum treats cough (stated in 3 passage(s) of the classical texts).", "source": "g"},
           {"id": "P1", "kind": "P", "text": "Haridra paste cures skin disease.", "source": "S"}]


def test_prompt_forbids_inferring_a_relation_from_co_mention_and_asks_for_the_score_on_safety_questions():
    p = build_answer_prompt("Is black pepper safe with phenytoin?", COITEMS)
    assert "co-mentioned" in p and "do not infer" in p.lower()
    assert "research risk score" in p and "never declare" in p.lower()


def test_overclaim_flags_sentences_that_cite_only_co_mention_facts_but_state_a_therapeutic_role():
    text = ("Turmeric is used for skin disease [P1]. Turmeric balances pitta [G1]. Turmeric appears alongside pitta in the texts [G1]. "
            "Piper nigrum treats cough [G2]. It is a nice day [G1][P1].")
    r = overclaim_flags(text, COITEMS)
    assert r["n_cocite"] == 2 and r["n_overclaim"] == 1                      # only 'balances pitta [G1]' over-reads; the explicit 'treats' edge G2 and mixed citations are fine
    assert overclaim_flags("No citations here.", COITEMS) == {"n_cocite": 0, "n_overclaim": 0}


def test_answer_reports_the_overclaim_counts():
    out = answer("q", COITEMS, lambda p: "Turmeric balances pitta [G1].")
    assert out["overclaim"] == {"n_cocite": 1, "n_overclaim": 1}


def test_extract_citations_expands_ranges_and_handles_mixed_lists():
    assert extract_citations("Fine [G1-G4].") == ["G1", "G2", "G3", "G4"]
    assert extract_citations("Fine [G1-4] and [P2, K1-K2].") == ["G1", "G2", "G3", "G4", "P2", "K1", "K2"]
    assert extract_citations("Single [P1] and again [P1].") == ["P1"]
    assert extract_citations("Reverse range [G4-G2] is ignored safely.") == ["G4"]


def test_strict_rules_can_be_switched_off_for_the_before_after_comparison():
    on = build_answer_prompt("q", COITEMS)
    off = build_answer_prompt("q", COITEMS, strict_rules=False)
    assert "co-mentioned' only means" in on and "research risk score" in on
    assert "co-mentioned' only means" not in off and "never declare" not in off.lower()
    assert NOT_ENOUGH in off and "[P1]" in off                                      # the core rules stay


def test_overclaim_check_ignores_negated_or_hedged_sentences_that_deny_the_relation():
    items = [{"id": "G1", "kind": "G", "text": "neem is co-mentioned with skin disease in 48 passages (co-occurrence only).", "source": "g"}]
    hedged = ["They do not state that neem affects or treats skin disease [G1].",
              "Neem is co-mentioned with skin disease, though co-occurrence does not imply treatment [G1].",
              "No direct effect of neem on skin disease is stated [G1].",
              "The texts never say neem cures skin disease [G1]."]
    for s in hedged:
        r = overclaim_flags(s, items)
        assert r == {"n_cocite": 1, "n_overclaim": 0}, s                  # counted as a co-occurrence sentence, but NOT as an over-reading
    assert overclaim_flags("Neem treats skin disease [G1].", items) == {"n_cocite": 1, "n_overclaim": 1}
    assert overclaim_flags("Neem is an effective remedy for skin disease [G1].", items) == {"n_cocite": 1, "n_overclaim": 1}
