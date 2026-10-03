"""Presentation helpers for the demo: one light palette, escaped HTML snippets, and the CSS loader.
Rules this module enforces: one theme, one accent, WCAG AA contrast, and no em or en dashes in anything shown to the user."""
import html
import re
from pathlib import Path

# One light palette. Accent is a muted green-teal; everything else is a warm-neutral grey. Checked for WCAG AA contrast in tests.
PALETTE = {
    "bg": "#F8FAF9", "surface": "#FFFFFF", "border": "#E1E7E4", "text": "#1B2621", "muted": "#56675F",
    "accent": "#2D7A66", "accent_text": "#1D5C4C", "accent_soft": "#E4F1EC", "on_accent": "#FFFFFF",
    "chip_bg": "#EDF1EF", "chip_text": "#3D4C45", "link": "#1D5C4C",
}
FONT_IMPORT = '@import url("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap");'
EM, EN = chr(0x2014), chr(0x2013)    # built from code points so these characters never appear literally in the source
CITE = re.compile(r"\[([A-Z]\d+(?:\s*[-,]\s*[A-Z]?\d+)*)\]")


def _lum(hex_color: str) -> float:
    c = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast_ratio(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def clean_dashes(text) -> str:
    """Display-only typography fix: no em or en dashes reach the screen (the stored data is left untouched)."""
    t = str(text)
    for d in (EM, EN):
        t = re.sub(r"(?<=\w)" + d + r"(?=\w)", "-", t)         # unspaced: a hyphen (Charaka-Samhita, 2018-2026)
        t = re.sub(r"\s*" + d + r"\s*", ", ", t)               # spaced or at an edge: a comma
    return t


def esc(text) -> str:
    return html.escape(clean_dashes(text), quote=True)


def chip(text, kind="neutral") -> str:
    return f'<span class="chip chip-{kind}">{esc(text)}</span>'


def stat_html(label, value, note=None) -> str:
    n = f'<div class="stat-note">{esc(note)}</div>' if note else ""
    return f'<div class="stat"><div class="stat-label">{esc(label)}</div><div class="stat-value">{esc(value)}</div>{n}</div>'


def cite_html(text) -> str:
    """Model or source text as paragraphs; citation tokens like [P1] or [G1-G4] become small chips. Escapes first, so content cannot inject markup."""
    paras = [p for p in re.split(r"\n\s*\n", clean_dashes(text).strip()) if p.strip()]
    return "".join("<p>" + CITE.sub(lambda m: f'<span class="cite">{m.group(1)}</span>', html.escape(p.replace("\n", " "), quote=False)) + "</p>" for p in paras)


def path_html(s: dict) -> str:
    """herb, compound, enzyme and drug as a left-to-right chain with the source of each link underneath."""
    node = lambda t, k="": f'<span class="node {k}">{esc(t)}</span>'
    edge = lambda t, src: f'<span class="edge"><span>{esc(t)} →</span><small>{esc(src)}</small></span>'
    return ('<div class="path">' + node(s["herb"]) + edge("contains", "IMPPAT") + node(s["compound"]) + edge("predicted inhibitor of", s["inhibitor_source"])
            + node(s["enzyme"], "enzyme") + edge("known substrate", s["substrate_source"]) + node(s["drug"]) + "</div>")


def shares_table_html(df) -> str:
    rows = []
    for r in df.itertuples():
        kind = "accent" if r.change < 0 else "neutral"
        rows.append(f"<tr><td>{esc(r.herb)}</td><td class='num'>{r.baseline_share * 100:.1f}</td><td class='num'>{r.suggested_share * 100:.1f}</td>"
                    f"<td class='num'>{chip(f'{r.change * 100:+.0f}%', kind)}</td></tr>")
    return ('<table class="table"><thead><tr><th>Herb</th><th class="num">Baseline %</th><th class="num">Suggested %</th><th class="num">Change</th></tr></thead>'
            "<tbody>" + "".join(rows) + "</tbody></table>")


def load_css() -> str:
    variables = ":root{" + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in PALETTE.items()) + "}"
    return FONT_IMPORT + variables + Path(__file__).with_name("style.css").read_text(encoding="utf-8")
