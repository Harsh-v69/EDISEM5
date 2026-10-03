"""Headless smoke test of the Streamlit app on the REAL local data (skipped where the IMPPAT-derived data is absent)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from ayurveda_kg.demo.ui import EM, EN

pytestmark = pytest.mark.skipif(not Path("data/processed/phase5/formulations.parquet").exists() or not Path("data/processed/phase5/herb_drug_risk.parquet").exists(),
                                reason="local IMPPAT-derived data not present (not redistributed)")
APP = str(Path(__file__).resolve().parents[1] / "ayurveda_kg" / "demo" / "app.py")      # AppTest resolves relative paths against the test file


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def test_app_renders_the_pair_check_without_errors_and_shows_the_disclaimer():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    text = page_text(at)
    assert "Not medical advice" in text and "Research risk score" in text and "What drives the score" in text
    assert len(at.tabs) == 3 and [t.label for t in at.tabs][0] == "Check a pair"


def test_changing_herb_and_drug_shows_the_published_study_for_a_gold_pair_and_the_negative_as_such():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="herb").set_value("Piper nigrum")
    at.selectbox(key="drug").set_value("phenytoin")
    at.run()
    assert not at.exception, at.exception
    assert "PubMed 16767797" in page_text(at) and "Interaction reported" in page_text(at)
    at.selectbox(key="herb").set_value("Zingiber officinale")
    at.selectbox(key="drug").set_value("warfarin")
    at.run()
    assert "No interaction found" in page_text(at)                                  # the human-trial negative is shown as such


def test_formulation_tab_returns_a_suggestion_table_for_the_default_selection():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    text = page_text(at)
    assert "Suggested shares" in text and 'class="table"' in text and "Predicted risk for this formulation" in text


def test_no_em_or_en_dash_is_visible_anywhere_in_the_rendered_page():
    at = AppTest.from_file(APP, default_timeout=120).run()
    shown = page_text(at) + " ".join(str(e.value) for e in at.get("selectbox")) + " ".join(str(o) for s in at.selectbox for o in s.options)
    assert EM not in shown and EN not in shown
