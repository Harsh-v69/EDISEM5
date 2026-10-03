"""Local demo: streamlit run ayurveda_kg/demo/app.py
Research use only. Reads IMPPAT-derived data (CC BY-NC-ND): run it locally, never host it publicly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # `streamlit run` does not add the repo root, and the data paths are relative to it

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ayurveda_kg.demo import service as S  # noqa: E402
from ayurveda_kg.demo import ui  # noqa: E402
from ayurveda_kg.rag.generate import DISCLAIMER, gemini_text_client, load_env, ollama_text_client, with_fallback  # noqa: E402

load_env()
st.set_page_config(page_title="Herb and drug research tool", layout="wide")
st.markdown(f"<style>{ui.load_css()}</style>", unsafe_allow_html=True)
html = lambda s: st.markdown(s, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the knowledge graph and risk tables")
def get_data():
    return S.DemoData.load()


@st.cache_resource(show_spinner="Loading the text index and models (first search only)")
def get_retriever():
    return S.build_retriever()


html('<h1 class="page-title">Herb and drug research tool</h1>'
     '<p class="page-sub">Check how a herb and a drug may interact, adjust a formulation, or search two classical texts. Every score links back to its evidence.</p>'
     '<div class="notice"><strong>Research use only.</strong> Not medical advice. Scores are hypotheses from predicted enzyme inhibition and have known false alarms. '
     'This tool shows IMPPAT-derived data (CC BY-NC-ND): run it on this computer and do not host it publicly.</div>')

d = get_data()
herbs, drugs = S.list_herbs(d), S.list_drugs(d)
common = dict(herbs)
tab1, tab2, tab3 = st.tabs(["Check a pair", "Adjust a formulation", "Search the texts"])

with tab1:
    c1, c2 = st.columns(2)
    herb_names = [h for h, _ in herbs]
    herb = c1.selectbox("Herb", herb_names, index=herb_names.index("Piper nigrum"), format_func=lambda h: f"{h} ({common[h]})", key="herb")
    drug = c2.selectbox("Drug", drugs, index=drugs.index("phenytoin"), key="drug")
    r = S.explain_pair(d, herb, drug)
    if "error" in r:
        st.error(r["error"])
    else:
        s = r["scores"]
        left, right = st.columns([1, 2], gap="medium")
        left.markdown(f'<div class="hero-score"><div class="label">Research risk score</div><div class="value">{s["risk_rf"]:.2f}</div>'
                      f'<div class="note">Average predicted probability across {s["n_compounds"]} compounds of this herb.</div></div>', unsafe_allow_html=True)
        right.markdown('<div class="stats">' + ui.stat_html("Mechanistic score", f"{s['risk_silver']:.2f}", "rule based, from enzyme data")
                       + ui.stat_html("Rank for this drug", f"{s['percentile_for_drug']:.0%}", "percentile among the 20 herbs")
                       + ui.stat_html("Compounds scored", s["n_compounds"], "from IMPPAT") + "</div>", unsafe_allow_html=True)

        html('<div class="section-title">Published evidence</div>')
        if r["gold"]:
            for g in r["gold"]:
                kind, verdict = ("accent", "Interaction reported") if g["label"] == 1 else ("neutral", "No interaction found")
                html(f'<div class="card evidence">{ui.chip(verdict, kind)}<div class="finding">{ui.esc(g["finding"])}</div>'
                     f'<div class="cite-line">{ui.esc(g["citation"])}. PubMed {ui.esc(g["pmid"])}. {ui.esc(g["mechanism"])}, {ui.esc(g["evidence"].replace("_", " "))}, '
                     f'confidence {ui.esc(g["confidence"])}. Curated by the project, pending expert review.</div></div>')
        else:
            html('<div class="card"><span class="muted">No published human study for this pair is in the curated set. The score above is the only evidence here.</span></div>')

        html('<div class="section-title">What drives the score</div>')
        body = ""
        for c in r["compounds"]:
            tag = {1: ui.chip("Mechanistic hit", "accent"), 0: ui.chip("No mechanistic hit"), -1: ui.chip("Undetermined"), None: ui.chip("Not labelled")}[c["silver_label"]]
            body += (f'<div class="compound"><div class="compound-head"><span class="compound-name">{ui.esc(c["name"])}</span>'
                     f'<span class="compound-p">probability {c["p"]:.2f} {tag}</span></div>' + "".join(ui.path_html(st_) for st_ in c["path_steps"]) + "</div>")
        html(f'<div class="card">{body}</div>')

        html(f'<div class="section-title">Highest scores for {ui.esc(drug)}</div>')
        rank = S.risk_ranking(d, drug, 8).rename(columns={"herb": "Herb", "risk_rf": "Research risk score"})
        st.bar_chart(rank, x="Herb", y="Research risk score", sort="-Research risk score", color=ui.PALETTE["accent"], horizontal=True, height=300)
        html(f'<p class="muted">{ui.esc(r["caveat"])}</p>')

with tab2:
    choices = S.formulation_choices(d)
    opt = choices[choices["optimisable"]].set_index("id")
    html(f'<p class="muted">{len(opt)} real formulations contain two or more of the 20 scoped herbs. IMPPAT gives no proportions, so the baseline is equal parts.</p>')
    a, b = st.columns([3, 2])
    fid = a.selectbox("Formulation", opt.index.tolist(), format_func=lambda i: f"{opt.loc[i, 'name']} ({opt.loc[i, 'n_in_scope']} scoped herbs)")
    pat = b.multiselect("Patient's drugs", drugs, default=["phenytoin"])
    with st.expander("Constraints"):
        k1, k2, k3 = st.columns(3)
        tau = k1.slider("Keep each therapeutic use at least this fraction of baseline", 0.5, 1.0, 0.8, 0.05)
        lo = k2.slider("Minimum share, times baseline", 0.0, 1.0, 0.5, 0.05)
        hi = k3.slider("Maximum share, times baseline", 1.0, 4.0, 2.0, 0.5)
        obj = st.radio("Objective", ["sum", "max"], format_func=lambda o: "Lower total risk" if o == "sum" else "Protect the worst drug", horizontal=True)
    if not pat:
        html('<div class="card"><span class="muted">Choose at least one drug to see a suggestion.</span></div>')
    else:
        res = S.suggest(d, fid, pat, tau, lo, hi, obj)
        if "error" in res:
            st.error(res["error"])
        elif not res["optimisable"]:
            html(f'<div class="card"><span class="muted">{ui.esc(res["message"])}</span></div>')
        else:
            html('<div class="section-title">Suggested shares</div>')
            html(f'<div class="card">{ui.shares_table_html(res["table"])}</div>')
            html(f'<p class="muted">{res["fixed_ingredients"]} other ingredients stay fixed and unscored. Lowest therapeutic coverage kept: {res["min_coverage_ratio"]:.2f}.</p>')
            html('<div class="section-title">Predicted risk for this formulation</div>')
            html('<div class="stats">' + "".join(ui.stat_html(x, f"{res['risk_after'][x]:.3f}", f"was {res['risk_before'][x]:.3f}") for x in pat) + "</div>")
            for f in res["flags"]:
                html(f'<div class="notice">{ui.esc(f)}</div>')
        html(f'<p class="muted" style="margin-top:1rem">{ui.esc(S.SUGGEST_CAVEAT)}</p>')

with tab3:
    q = st.text_input("Question", "What does the Charaka Samhita say about treating cough?")
    gemini = gemini_text_client()
    use_llm = st.checkbox("Write an answer with a language model" + (" (Gemini, falling back to local qwen3:8b)" if gemini else " (local qwen3:8b)"), value=bool(gemini))
    if gemini:
        html('<p class="muted">With this on, your question and the retrieved passages are sent to Google. Turn it off to stay fully local.</p>')
    go = st.button("Search")
    if not go:
        html('<p class="muted">Try: how is turmeric used, which remedies are used for skin disease, or what is said about ginger and digestion. '
             'Turmeric appears in the texts as Haridra, and the search knows that.</p>')
    elif q.strip():
        with st.spinner("Searching the texts"):
            client = with_fallback(gemini, ollama_text_client()) if use_llm else None
            out = S.ask(get_retriever(), q, client)
        if client and client.used is not None:
            html(f'<p class="muted">Answered by {"Gemini" if gemini and client.used == 0 else "local qwen3:8b"}.'
                 + (f' Gemini failed ({ui.esc("; ".join(client.errors))}), so the local model was used.' if client.errors else "") + "</p>")
        if out["answer"]:
            ck = out["answer"]["checks"]
            body = f'<div class="card answer">{ui.cite_html(out["answer"]["model_text"] or "The language model was unavailable.")}</div>'
            html(body)
            html(f'<p class="muted">{ui.esc(DISCLAIMER)} Citations found: {ui.esc(", ".join(ck["valid"]) or "none")}. Invalid: {ui.esc(", ".join(ck["invalid"]) or "none")}. '
                 'These automatic checks are proxies, not a judgement of correctness.</p>')
            if out["answer"]["error"]:
                st.error(f"The language model was unavailable: {out['answer']['error']}")
        else:
            html(f'<p class="muted">{ui.esc(out["note"])}</p>')
        src = "".join(f'<div class="source"><div class="meta">{ui.chip(i["id"], "accent")} {ui.esc(i["source"])}</div>{ui.esc(i["text"])}</div>' for i in out["items"])
        with st.expander(f"Sources ({len(out['items'])})", expanded=not out["answer"]):
            html(src)
