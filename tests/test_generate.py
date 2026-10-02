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
