from pathlib import Path

import pandas as pd
import pytest

from ayurveda_kg.demo.ui import PALETTE, chip, cite_html, clean_dashes, contrast_ratio, esc, load_css, path_html, shares_table_html, stat_html

EM, EN = "—", "–"


def test_clean_dashes_removes_em_and_en_dashes_but_keeps_hyphens():
    assert EM not in clean_dashes(f"word {EM} word") and EN not in clean_dashes(f"2018{EN}2026")
    assert clean_dashes(f"a {EM} b") == "a, b" and clean_dashes(f"2018{EN}2026") == "2018-2026" and clean_dashes("well-known") == "well-known"
    assert clean_dashes(f"Charaka{EM}Samhita") == "Charaka-Samhita"


def test_esc_escapes_html_and_dashes():
    out = esc(f'<script>alert("x")</script> a {EM} b')
    assert "<script>" not in out and "&lt;script&gt;" in out and EM not in out


def test_chip_and_stat_escape_their_content_and_carry_the_kind_class():
    assert 'class="chip chip-accent"' in chip("<b>hit</b>", "accent") and "<b>" not in chip("<b>hit</b>", "accent")
    s = stat_html("Mechanistic score", "0.48", "silver")
    assert "Mechanistic score" in s and "0.48" in s and "silver" in s and 'class="stat"' in s


def test_cite_html_turns_citation_tokens_into_chips_after_escaping_and_splits_paragraphs():
    out = cite_html("Haridra cures skin disease [P1] <b>x</b>. See [G1-G4] and [P2, K1].\n\nSecond paragraph.")
    assert out.count('class="cite"') == 3 and ">P1<" in out and ">G1-G4<" in out and ">P2, K1<" in out
    assert "<b>" not in out and "&lt;b&gt;" in out and out.count("<p>") == 2
    assert "[P1]" not in out


def test_path_html_shows_all_four_entities_and_both_sources_escaped():
    out = path_html({"herb": "Piper nigrum", "compound": "Guineensine <i>", "enzyme": "CYP2C9", "drug": "phenytoin",
                     "inhibitor_source": "SwissADME (via IMPPAT)", "substrate_source": "TDC Carbon-Mangels"})
    for must in ["Piper nigrum", "Guineensine", "CYP2C9", "phenytoin", "SwissADME", "TDC Carbon-Mangels"]:
        assert must in out
    assert "<i>" not in out and "&lt;i&gt;" in out


def test_shares_table_html_lists_herbs_with_signed_percent_changes():
    df = pd.DataFrame({"herb": ["Zingiber officinale", "Terminalia chebula"], "baseline_share": [0.034, 0.034], "suggested_share": [0.0272, 0.0476], "change": [-0.2, 0.4]})
    out = shares_table_html(df)
    assert "Zingiber officinale" in out and "3.4" in out and "-20%" in out and "+40%" in out


@pytest.mark.parametrize("fg,bg,minimum", [("text", "bg", 7.0), ("text", "surface", 7.0), ("muted", "bg", 4.5), ("muted", "surface", 4.5),
                                           ("accent_text", "accent_soft", 4.5), ("accent_text", "surface", 4.5), ("on_accent", "accent", 4.5),
                                           ("chip_text", "chip_bg", 4.5), ("link", "surface", 4.5)])
def test_palette_meets_wcag_aa_contrast(fg, bg, minimum):
    assert contrast_ratio(PALETTE[fg], PALETTE[bg]) >= minimum, (fg, bg, contrast_ratio(PALETTE[fg], PALETTE[bg]))


def test_contrast_ratio_known_values():
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


def test_palette_is_light_and_avoids_the_ai_purple_and_pure_black_white_defaults():
    assert PALETTE["bg"].lower() not in ("#ffffff", "#000000") and PALETTE["text"].lower() != "#000000"
    r, g, b = (int(PALETTE["accent"][i:i + 2], 16) for i in (1, 3, 5))
    assert g > r and g >= b - 10                                    # a green-teal accent, not a purple/violet one (red or blue dominant)
    assert sum(int(PALETTE["bg"][i:i + 2], 16) for i in (1, 3, 5)) > 735     # a light page background


def test_css_is_built_from_the_palette_variables_and_uses_the_chosen_fonts():
    css = load_css()
    for k in PALETTE:
        assert f"--{k.replace('_', '-')}:" in css
    assert "Geist" in css and "Inter" not in css and "prefers-color-scheme" not in css          # one light theme, no Inter default


def test_no_em_or_en_dash_anywhere_in_the_demo_sources():
    root = Path(__file__).resolve().parents[1]
    files = [root / "ayurveda_kg/demo/app.py", root / "ayurveda_kg/demo/ui.py", root / "ayurveda_kg/demo/service.py", root / "ayurveda_kg/demo/style.css",
             root / ".streamlit/config.toml"]
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert EM not in text and EN not in text, f.name
