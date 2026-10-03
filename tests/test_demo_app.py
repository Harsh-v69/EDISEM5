"""Headless smoke test of the Streamlit app on the REAL local data (skipped where the IMPPAT-derived data is absent)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.skipif(not Path("data/processed/phase5/formulations.parquet").exists() or not Path("data/processed/phase5/herb_drug_risk.parquet").exists(),
                                reason="local IMPPAT-derived data not present (not redistributed)")
APP = str(Path(__file__).resolve().parents[1] / "ayurveda_kg" / "demo" / "app.py")      # AppTest resolves relative paths against the test file


def test_app_renders_the_risk_lookup_without_errors_and_shows_the_disclaimer():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("not medical advice" in w.value.lower() for w in at.warning)
    assert len(at.tabs) == 3 and len(at.metric) >= 4
    assert at.metric[0].label == "Research risk score" and 0.0 <= float(at.metric[0].value) <= 1.0


def test_changing_herb_and_drug_updates_the_score_and_shows_the_published_study_for_a_gold_pair():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="herb").set_value("Piper nigrum")
    at.selectbox(key="drug").set_value("phenytoin")
    at.run()
    assert not at.exception, at.exception
    assert any("PubMed 16767797" in i.value for i in at.info)                      # Pattanaik 2006 shown as evidence for this pair
    at.selectbox(key="herb").set_value("Zingiber officinale")
    at.selectbox(key="drug").set_value("warfarin")
    at.run()
    assert any("no interaction found" in i.value for i in at.info)                  # the human-trial negative is shown as such


def test_formulation_tab_returns_a_suggestion_table_for_the_default_selection():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 2                                                    # shares table + risk before/after table
