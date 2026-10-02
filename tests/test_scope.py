from ayurveda_kg.scope import load_scope


def test_scope_loads_with_expected_shape():
    s = load_scope()
    assert len(s["herbs"]) >= 15 and len(s["drugs"]) >= 25
    assert all(h["imppat_name"] for h in s["herbs"])
    assert all(d["name"] and d["cls"] for d in s["drugs"])


def test_no_duplicate_names_or_aliases():
    s = load_scope()
    herb_keys = [h["imppat_name"].lower() for h in s["herbs"]]
    assert len(herb_keys) == len(set(herb_keys))
    aliases = [a.lower() for h in s["herbs"] for a in h.get("aliases", [])]
    assert len(aliases) == len(set(aliases))
    assert not set(aliases) & set(herb_keys)
    drugs = [d["name"].lower() for d in s["drugs"]]
    assert len(drugs) == len(set(drugs))


def test_tinospora_synonym_is_declared():
    s = load_scope()
    h = next(h for h in s["herbs"] if h["imppat_name"] == "Tinospora sinensis")
    assert "Tinospora cordifolia" in h["aliases"]
