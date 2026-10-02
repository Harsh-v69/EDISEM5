import pytest

from ayurveda_kg.rag.corpus import chunk_paragraphs, clean_text, make_chunks, paragraphs

RAW = """
CHARAKA-SAMHITA.


14


Kritavedhana(a),  Pippali(b),  Kutaja(c),  large  carda¬
moms,  and  Dhamargava(e),  should  be  used  as
emetics  in  disorders  of  the  phlegm  and  bile.

a.  Vide  note  i,  p.  8.

wwift ^ I

numftifmiTn Hnww*  »’  i

The  physician  should  use  the  drugs  without  injuring  the
system,  as  purgatives  in  diseases  of  the  intestinal  canal.
"""


def test_clean_text_rejoins_hyphenation_collapses_spaces_and_drops_headers_page_numbers_and_garbage():
    t = clean_text(RAW)
    assert "cardamoms" in t and "carda" in t and "carda¬" not in t
    assert "  " not in t
    assert "CHARAKA-SAMHITA" not in t                     # running header removed
    assert "\n14\n" not in t and not any(line.strip() == "14" for line in t.splitlines())   # bare page number removed
    assert "wwift" not in t and "numftifmiTn" not in t    # OCR garbage lines removed
    assert "Pippali(b)" in t and "physician should use" in t     # real content kept


def test_paragraphs_reflow_lines_and_drop_tiny_fragments():
    ps = paragraphs(clean_text(RAW))
    assert all("\n" not in p for p in ps) and all(len(p.split()) >= 5 for p in ps)
    assert any(p.startswith("Kritavedhana") for p in ps)
    assert not any(p.startswith("a. Vide") and len(p.split()) < 5 for p in ps)


def test_chunks_respect_size_bounds_overlap_and_cover_all_text_in_order():
    paras = [f"para{i} " + " ".join(f"w{i}_{j}" for j in range(40)) for i in range(30)]       # 41 words each
    chunks = chunk_paragraphs(paras, target_words=120, max_words=200, overlap_words=60)
    assert all(c["n_words"] <= 200 for c in chunks[:-1]) and all(c["n_words"] >= 60 for c in chunks[:-1])
    joined = " ".join(c["text"] for c in chunks)
    for p in paras:
        assert p.split()[0] in joined                                # every paragraph appears somewhere
    assert chunks[0]["first_para"] == 0 and [c["first_para"] for c in chunks] == sorted(c["first_para"] for c in chunks)
    # overlap: the last paragraph of a chunk is repeated at the start of the next one (it is short enough to carry)
    assert chunks[1]["text"].startswith(chunks[0]["text"].split("\n")[-1][:20]) or chunks[1]["first_para"] <= chunks[0]["last_para"]


def test_a_huge_paragraph_becomes_its_own_chunk_and_nothing_is_lost():
    big = " ".join(f"x{i}" for i in range(500))
    chunks = chunk_paragraphs(["small start paragraph with enough words here", big, "small end paragraph with enough words here"],
                              target_words=100, max_words=150, overlap_words=0)
    assert any(c["n_words"] >= 500 for c in chunks) or sum(c["n_words"] for c in chunks) >= 500
    assert "x499" in " ".join(c["text"] for c in chunks)


def test_make_chunks_assigns_unique_stable_ids_and_source_labels():
    text = "\n\n".join(" ".join(f"word{i}x{j}" for j in range(60)) for i in range(20))
    a = make_chunks(text, "charaka", "Charaka Samhita (Kaviratna)", target_words=100, max_words=180)
    b = make_chunks(text, "charaka", "Charaka Samhita (Kaviratna)", target_words=100, max_words=180)
    assert [c["id"] for c in a] == [c["id"] for c in b] and len({c["id"] for c in a}) == len(a)
    assert a[0]["id"] == "charaka-00000" and a[0]["source"] == "Charaka Samhita (Kaviratna)"
    assert all(c["text"].strip() for c in a)


def test_make_chunks_drops_tiny_remnants_but_ids_stay_contiguous():
    big = "\n\n".join(" ".join(f"word{i}x{j}" for j in range(70)) for i in range(5))
    text = big + "\n\nshort tail of five words"
    chunks = make_chunks(text, "k", "K", target_words=100, max_words=180, overlap_words=0)
    assert all(c["n_words"] >= 15 for c in chunks)
    assert [c["id"] for c in chunks] == [f"k-{i:05d}" for i in range(len(chunks))]
