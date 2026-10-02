import pandas as pd
import pytest

from ayurveda_kg.curated import load_gold, load_substrate_supplement, validate_gold
from ayurveda_kg.scope import load_scope

SCOPE = {"herbs": [{"imppat_name": "Piper nigrum"}, {"imppat_name": "Zingiber officinale"}],
         "drugs": [{"name": "warfarin"}, {"name": "propranolol"}]}
ROW = dict(herb="Piper nigrum", drug="propranolol", label=1, mechanism="PK", evidence="human_crossover_pk", confidence="high",
           pmid="1815977", citation="Bano 1991", finding="x", verification="abstract read")


def frame(*overrides):
    return pd.DataFrame([{**ROW, **o} for o in (overrides or [{}])])


def probs(*overrides):
    return "\n".join(validate_gold(frame(*overrides), SCOPE))


def test_valid_row_has_no_problems():
    assert validate_gold(frame(), SCOPE) == []


@pytest.mark.parametrize("override,expect", [
    ({"herb": "Ghost herb"}, "unknown herb"),
    ({"drug": "ghostdrug"}, "unknown drug"),
    ({"label": 2}, "label must be 0 or 1"),
    ({"mechanism": "magic"}, "bad mechanism"),
    ({"evidence": "vibes"}, "bad evidence"),
    ({"confidence": "certain"}, "bad confidence"),
    ({"pmid": "abc"}, "pmid must be numeric"),
    ({"verification": ""}, "missing verification"),
    ({"citation": " "}, "missing citation"),
    ({"label": 0, "mechanism": "PK"}, "label 0 requires mechanism none"),
    ({"label": 1, "mechanism": "none"}, "label 1 cannot have mechanism none"),
])
def test_each_kind_of_bad_row_is_caught(override, expect):
    assert expect in probs(override)


def test_duplicate_herb_drug_pmid_is_caught():
    assert "duplicate" in probs({}, {})


def test_real_gold_file_is_valid_and_has_both_labels_and_pk_subset():
    g = load_gold()
    assert validate_gold(g, load_scope()) == []
    assert set(g.label) == {0, 1} and len(g) >= 10
    assert ((g.mechanism == "PK") & (g.label == 1)).sum() >= 5     # enough PK positives for a meaningful recall check


def test_supplement_maps_drugs_and_refuses_unknown_drug(tmp_path):
    drugs = pd.DataFrame({"id": ["drug:apixaban"], "name": ["apixaban"], "aliases": [[]]})
    p = tmp_path / "s.csv"
    p.write_text("drug,enzyme,source_type,citation,url,verification\napixaban,CYP3A4,fda_label,c,u,v\n")
    e = load_substrate_supplement(p, drugs)
    assert e[["src", "dst", "source"]].values.tolist() == [["drug:apixaban", "gene:CYP3A4", "curated literature"]]
    p.write_text("drug,enzyme,source_type,citation,url,verification\nnotadrug,CYP3A4,fda_label,c,u,v\n")
    with pytest.raises(ValueError, match="notadrug"):
        load_substrate_supplement(p, drugs)
