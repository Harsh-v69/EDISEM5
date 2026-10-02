import pytest

from ayurveda_kg.rag.lexicon import Lexicon, load_lexicon, normalise

SCOPE = {"herbs": [{"imppat_name": "Curcuma longa", "aliases": ["Turmeric", "Haridra"], "common": "turmeric"},
                   {"imppat_name": "Piper nigrum", "aliases": ["Black pepper", "Maricha"], "common": "black pepper"},
                   {"imppat_name": "Zingiber officinale", "aliases": ["Ginger", "Shunthi"], "common": "ginger"}],
         "drugs": [{"name": "warfarin", "cls": "anticoagulant"}, {"name": "phenytoin", "cls": "anticonvulsant"}]}
EXTRA = {"extra_herb_aliases": {"Zingiber officinale": ["Sunthi", "dry ginger"]},
         "doshas": {"vata": ["vata", "wind"], "pitta": ["pitta", "bile"], "kapha": ["kapha", "phlegm"]},
         "conditions": {"cough": ["cough", "kasa"], "skin disease": ["skin disease", "kushtha"]}}


def lex():
    return Lexicon.from_config(SCOPE, EXTRA)


def ids(text):
    return [m["id"] for m in lex().link(text)]


def test_normalise_strips_diacritics_case_and_hyphen_variants():
    assert normalise("Haridrā ") == "haridra" and normalise("S'ringa-vera") == "s'ringa vera"
    assert normalise("  DRY   Ginger ") == "dry ginger"


def test_links_herb_by_canonical_alias_and_sanskrit_name_case_insensitively():
    assert ids("Turmeric is applied") == ["herb:Curcuma longa"]
    assert ids("the paste of HARIDRA and Maricha") == ["herb:Curcuma longa", "herb:Piper nigrum"]
    assert ids("black pepper") == ["herb:Piper nigrum"] and ids("Curcuma longa") == ["herb:Curcuma longa"]


def test_extra_aliases_doshas_conditions_and_drugs_link_with_types():
    m = {x["id"]: x for x in lex().link("Sunthi relieves cough born of phlegm; warfarin interacts")}
    assert m["herb:Zingiber officinale"]["type"] == "Herb" and m["condition:cough"]["type"] == "Condition"
    assert m["dosha:kapha"]["type"] == "Dosha" and m["drug:warfarin"]["type"] == "Drug"


def test_longest_match_wins_and_matches_respect_word_boundaries():
    assert ids("dry ginger is used") == ["herb:Zingiber officinale"] and len(lex().link("dry ginger")) == 1     # not two matches
    assert ids("gingerbread and windows") == []                                                                  # 'ginger' in 'gingerbread', 'wind' in 'windows'
    assert ids("skin disease of the wind") == ["condition:skin disease", "dosha:vata"]


def test_mentions_report_surface_form_and_span_and_repeats_are_kept_in_order():
    ms = lex().link("Haridra and then haridra again")
    assert [m["surface"] for m in ms] == ["Haridra", "haridra"] and ms[0]["start"] < ms[1]["start"]
    assert lex().entities("Haridra and haridra and cough") == ["herb:Curcuma longa", "condition:cough"]          # unique, in order of first mention


def test_ambiguous_surface_form_claimed_by_two_entities_is_dropped():
    scope = {"herbs": [{"imppat_name": "A herb", "aliases": ["Shared name"]}, {"imppat_name": "B herb", "aliases": ["shared name"]}], "drugs": []}
    l = Lexicon.from_config(scope, {"extra_herb_aliases": {}, "doshas": {}, "conditions": {}})
    assert l.link("a shared name here") == []


def test_real_config_loads_and_links_the_expected_herbs():
    l = load_lexicon()
    got = l.entities("Pippali and Haridra with Maricha for kasa from kapha; Amalaki; Guduchi")
    for must in ["herb:Curcuma longa", "herb:Piper nigrum", "herb:Phyllanthus emblica", "herb:Tinospora sinensis", "condition:cough", "dosha:kapha"]:
        assert must in got, must
