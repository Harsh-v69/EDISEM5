"""Local demo: streamlit run ayurveda_kg/demo/app.py
Research use only. Reads IMPPAT-derived data (CC BY-NC-ND): run it locally, never host it publicly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # `streamlit run` does not add the repo root, and the data paths are relative to it

import streamlit as st  # noqa: E402

from ayurveda_kg.demo import service as S  # noqa: E402
from ayurveda_kg.rag.generate import DISCLAIMER, ollama_text_client  # noqa: E402

st.set_page_config(page_title="Ayurveda herb-drug research tool", layout="wide")


@st.cache_resource(show_spinner="Loading the knowledge graph and risk tables...")
def get_data():
    return S.DemoData.load()


@st.cache_resource(show_spinner="Loading the text index and models (first use only)...")
def get_retriever():
    return S.build_retriever()


st.title("Ayurveda herb-drug research tool")
st.warning(f"{DISCLAIMER} Scores are research hypotheses from predicted CYP inhibition, with known false alarms. "
           "This app uses IMPPAT-derived data (CC BY-NC-ND): run it locally; do not host it publicly.")
d = get_data()
herbs, drugs = S.list_herbs(d), S.list_drugs(d)
tab1, tab2, tab3 = st.tabs(["Risk lookup", "Re-weight a formulation", "Ask the classical texts"])

with tab1:
    c1, c2 = st.columns(2)
    herb = c1.selectbox("Herb", [h for h, _ in herbs], format_func=lambda h: f"{h} ({dict(herbs)[h]})", key="herb")
    drug = c2.selectbox("Drug", drugs, key="drug")
    r = S.explain_pair(d, herb, drug)
    if "error" in r:
        st.error(r["error"])
    else:
        s = r["scores"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Research risk score", f"{s['risk_rf']:.2f}")
        m2.metric("Mechanistic (silver) score", f"{s['risk_silver']:.2f}")
        m3.metric("Rank among herbs for this drug", f"{s['percentile_for_drug']:.0%} percentile")
        m4.metric("Compounds scored", s["n_compounds"])
        st.caption(r["caveat"])
        st.subheader("Evidence from the literature")
        if r["gold"]:
            for g in r["gold"]:
                verdict = "interaction reported" if g["label"] == 1 else "no interaction found"
                st.info(f"**{verdict}** ({g['mechanism']}, {g['evidence']}, confidence {g['confidence']}): {g['finding']}  \n"
                        f"{g['citation']} · PubMed {g['pmid']}  \n_Gold set, pending expert review._")
        else:
            st.write("No published human study for this pair is in the project's curated set.")
        st.subheader("Compounds driving the score")
        for c in r["compounds"]:
            label = {1: "mechanistic hit", 0: "no mechanistic hit", -1: "undetermined", None: "not labelled"}[c["silver_label"]]
            st.markdown(f"**{c['name']}**: out-of-fold probability {c['p']:.2f} · {label}")
            for p in c["paths"]:
                st.code(p, language=None, wrap_lines=True)
        st.subheader(f"Herbs with the highest scores for {drug}")
        st.bar_chart(S.risk_ranking(d, drug, 8).set_index("herb")["risk_rf"])

with tab2:
    choices = S.formulation_choices(d)
    opt = choices[choices["optimisable"]]
    st.write(f"{len(opt)} real IMPPAT formulations contain two or more scoped herbs and can be re-weighted. IMPPAT gives no proportions, so the baseline is equal parts.")
    fid = st.selectbox("Formulation", opt["id"].tolist(), format_func=lambda i: f"{opt.set_index('id').loc[i, 'name']} ({opt.set_index('id').loc[i, 'n_in_scope']} scoped herbs)")
    pat = st.multiselect("Patient's drugs", drugs, default=[drugs[0]])
    a, b, c = st.columns(3)
    tau = a.slider("Keep each therapeutic use at least this fraction of baseline", 0.5, 1.0, 0.8, 0.05)
    lo = b.slider("Minimum share (x baseline)", 0.0, 1.0, 0.5, 0.05)
    hi = c.slider("Maximum share (x baseline)", 1.0, 4.0, 2.0, 0.5)
    obj = st.radio("Objective", ["sum", "max"], format_func=lambda o: "Lower total risk" if o == "sum" else "Protect the worst drug", horizontal=True)
    if pat:
        res = S.suggest(d, fid, pat, tau, lo, hi, obj)
        if "error" in res:
            st.error(res["error"])
        elif not res["optimisable"]:
            st.info(res["message"])
        else:
            t = res["table"].copy()
            t["baseline_share"], t["suggested_share"] = (t["baseline_share"] * 100).round(1), (t["suggested_share"] * 100).round(1)
            t["change"] = (t["change"] * 100).round(0)
            st.dataframe(t.rename(columns={"baseline_share": "baseline %", "suggested_share": "suggested %", "change": "change %"}), hide_index=True)
            st.write(f"{res['fixed_ingredients']} other ingredient(s) stay fixed and unscored. Lowest therapeutic-use coverage kept: {res['min_coverage_ratio']:.2f}.")
            st.dataframe({"drug": list(res["risk_before"]), "risk before": [round(v, 3) for v in res["risk_before"].values()],
                          "risk after": [round(v, 3) for v in res["risk_after"].values()]}, hide_index=True)
            for f in res["flags"]:
                st.warning(f)
        st.caption(S.SUGGEST_CAVEAT)

with tab3:
    q = st.text_input("Question", "What does the Charaka Samhita say about treating cough?")
    use_llm = st.checkbox("Answer with the local language model (Ollama, qwen3:8b)", value=False)
    if st.button("Search") and q.strip():
        out = S.ask(get_retriever(), q, ollama_text_client() if use_llm else None)
        if out["answer"]:
            st.markdown(out["answer"]["text"])
            ck = out["answer"]["checks"]
            st.caption(f"Citation check: valid {ck['valid']} · invalid {ck['invalid']}. Automatic checks are proxies, not a judgement of correctness.")
            if out["answer"]["error"]:
                st.error(f"The language model was unavailable: {out['answer']['error']}")
        else:
            st.info(out["note"])
        with st.expander("Retrieved sources", expanded=not out["answer"]):
            for i in out["items"]:
                st.markdown(f"**[{i['id']}]** _{i['source']}_  \n{i['text']}")
