"""FOMC RAG Lab — clean single-page Streamlit research workspace.

Run from the RAG_FOMC project root:
    python -m streamlit run app/streamlit_app.py

Only presentation/orchestration is changed. Existing src/ modules and benchmark
indexes/qrels are not modified.
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate import generate  # noqa: E402
from rag_pipeline import retrieve  # noqa: E402
from explain import build_explanation, markdown_report  # noqa: E402
from temporal_retrieval import detect_periods, retrieve_temporal  # noqa: E402
from coverage_guard import generate_with_coverage, build_sections  # noqa: E402
from evidence_audit import audit_evidence  # noqa: E402

APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from groq_generation import generate_groq  # noqa: E402

st.set_page_config(
    page_title="FOMC RAG Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

RETRIEVAL_LABELS = {
    "hybrid_rrf": "Hybrid RRF · BM25 + E5",
    "bm25": "BM25 · Lexical",
    "dense": "Dense E5 · Semantic",
}
RERANK_LABELS = {
    "colbert": "ColBERT · MaxSim",
    "cross_encoder": "Cross-Encoder",
    "bm25": "BM25 · Baseline",
    "none": "None",
}
EXAMPLES = [
    "What change did the May 2024 statement announce for Treasury redemptions starting in June?",
    "What did the FOMC decide about the federal funds rate in September 2022, and why?",
    "How did the FOMC's monetary policy stance change between March 2022 and July 2023, and what economic factors motivated these decisions?",
]

DEFAULTS = {
    "draft_question": "",
    "search_mode": "classic",
    "retrieval_method": "hybrid_rrf",
    "reranker_method": "colbert",
    "context_k": 6,
    "structured_generation": True,
    "use_llm": True,
    "model_name": "qwen2.5:3b",
}
for name, value in DEFAULTS.items():
    if name not in st.session_state:
        st.session_state[name] = value
if "question" not in st.session_state:
    st.session_state["question"] = st.session_state["draft_question"]


def photo_data_uri() -> str:
    """Embed the supplied image locally; no remote image hosting is needed."""
    path = ROOT / "app" / "assets" / "finance_background.webp"
    if not path.exists():
        return ""
    return "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


HERO_URL = photo_data_uri()

# Photo is embedded as a real image element (not a CSS background URL):
# this avoids Streamlit stripping large data: URLs from stylesheet rules.
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&family=Playfair+Display:ital,wght@1,500;1,600&display=swap');
:root {
  --ink:#242338; --muted:#747489; --indigo:#565B9C;
  --violet:#8581C4; --lilac:#F3F2FB; --line:#E8E7F0;
  --surface:#FFFFFF; --canvas:#FCFCFE;
}
html,body,[data-testid="stAppViewContainer"] {font-family:'DM Sans',sans-serif;color:var(--ink);}
[data-testid="stAppViewContainer"] {background:var(--canvas);}
header[data-testid="stHeader"] {height:3.5rem;background:rgba(252,252,254,.96);border-bottom:0;}
[data-testid="stSidebar"] {background:#FDFDFE;border-right:1px solid #F0EFF4;}
.block-container {max-width:1440px;padding:4.5rem 2.4rem 3.2rem;}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif;color:var(--ink);letter-spacing:-.045em;}
p {line-height:1.68;}
/* Soft, Cosmos-like controls: restrained accents, generous white space. */
[data-testid="stMetric"] {background:rgba(255,255,255,.72)!important;border:1px solid #ECEBF4!important;border-radius:20px!important;padding:17px 18px!important;box-shadow:0 4px 17px rgba(62,62,112,.035)!important;}
[data-testid="stMetricLabel"] {font-size:.78rem;color:#77758D;}
[data-testid="stMetricValue"] {color:#494C89;font-size:clamp(1.35rem,2vw,1.95rem);font-weight:700;letter-spacing:-.05em;}
[data-testid="stExpander"] {background:#fff!important;border:1px solid #EAE9F2!important;border-radius:18px!important;box-shadow:none!important;overflow:hidden;margin-bottom:10px;}
[data-testid="stExpander"] details {border:0!important;box-shadow:none!important;}
[data-testid="stExpander"] summary {padding:13px 18px!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:1.3rem;border-bottom:1px solid var(--line);}
button[data-baseweb="tab"] {border:0!important;background:transparent!important;color:#79778A!important;}
button[data-baseweb="tab"][aria-selected="true"] {color:#555A9C!important;font-weight:700;}
[data-testid="stDataFrame"] {border:1px solid var(--line)!important;border-radius:14px!important;overflow:hidden;box-shadow:none!important;}
[data-testid="stAlert"] {border-radius:14px;border-color:#E5E4F4!important;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {background:#fff;border:1px solid #E7E6EE;border-radius:16px;color:var(--ink);box-shadow:none;}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus {border-color:#9C9AD2;box-shadow:0 0 0 2px rgba(133,129,196,.10);}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {border-radius:13px;background:white;border:1px solid #EAE9F1;box-shadow:none;}
[data-testid="stSlider"] [role="slider"] {background:#7978BA;border-color:#7978BA;}
[data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button,[data-testid="stLinkButton"] a {border-radius:999px!important;box-shadow:none!important;font-family:'DM Sans',sans-serif;font-weight:600;}
[data-testid="stButton"] button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primary"] {background:#696BB4!important;border:1px solid #696BB4!important;color:white!important;}
[data-testid="stButton"] button[kind="primary"]:hover,[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {background:#565898!important;border-color:#565898!important;color:white!important;}
[data-testid="stButton"] button[kind="secondary"],[data-testid="stDownloadButton"] button {background:white;border:1px solid #E5E3EE;color:#34314A;}
.brand {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.35rem;letter-spacing:-.05em;margin:4px 0;}
.brand span {color:#7473B2;}
/* Photo fills a tall hero. Text never sits behind the fixed Streamlit header. */
.hero {position:relative;min-height:clamp(440px,39vw,560px);overflow:hidden;isolation:isolate;border:0;border-radius:26px;background:#F7F5FE;}
.hero-photo {position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center 49%;z-index:0;}
.hero:before {content:"";position:absolute;inset:0;z-index:1;background:linear-gradient(90deg,rgba(251,250,255,.97) 0%,rgba(251,250,255,.90) 29%,rgba(251,250,255,.45) 57%,rgba(251,250,255,.05) 100%);}
.hero-body {position:relative;z-index:2;padding:clamp(55px,6vw,92px) clamp(32px,5.5vw,76px);max-width:760px;}
.hero h1 {font-size:clamp(2.45rem,4.1vw,4.35rem);font-weight:800;line-height:1.11;letter-spacing:-.07em;margin:18px 0 24px;}
.hero h1 em {color:#7773B8;font-family:'Playfair Display',serif;font-weight:500;letter-spacing:-.045em;}
.hero p {font-size:1.04rem;color:#58576E;max-width:520px;line-height:1.8;}
.eyebrow {font-size:.66rem;letter-spacing:.16em;text-transform:uppercase;color:#7471A8;font-weight:700;}
.section-title {font-family:'Manrope',sans-serif;font-size:1.47rem;letter-spacing:-.04em;font-weight:760;margin:45px 0 17px;}
.micro-title {font-family:'Manrope',sans-serif;font-size:1.03rem;letter-spacing:-.03em;font-weight:700;margin:25px 0 15px;}
.rule {height:1px;background:var(--line);margin:37px 0;}
.feature-symbol {color:#8684BB;font-size:1.35rem;margin:14px 0 17px;}
.feature-title {font-family:'Manrope',sans-serif;font-size:1.02rem;font-weight:740;margin-bottom:10px;}
.feature-desc {font-size:.88rem;color:#77758B;max-width:300px;line-height:1.75;}
.answer-text {font-size:1rem;line-height:1.95;white-space:pre-wrap;overflow-wrap:anywhere;color:#302E43;}
.source-marker {font-size:.72rem;font-weight:700;color:#6A68A9;letter-spacing:.04em;}
.subtle,.sidebar-note {color:#77758B;font-size:.85rem;line-height:1.7;}
/* Search bar BELOW hero. No negative margins or overlap. */
[data-testid="stForm"] {position:relative;z-index:1;width:100%!important;max-width:1080px;margin:30px auto 28px!important;padding:12px 15px!important;background:white!important;border:1px solid #E7E6F0!important;border-radius:28px!important;box-shadow:0 9px 25px rgba(65,61,113,.055)!important;}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {align-items:center!important;gap:1rem!important;flex-wrap:nowrap;}
[data-testid="stForm"] [data-testid="column"] {min-width:0!important;}
[data-testid="stForm"] [data-testid="stTextInput"] {width:100%!important;min-width:0!important;}
[data-testid="stForm"] [data-testid="stTextInput"] input {width:100%!important;min-width:0!important;height:58px!important;border:0!important;background:transparent!important;box-shadow:none!important;padding:0 15px!important;font-size:1.02rem!important;}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {min-height:54px!important;min-width:142px!important;padding:0 22px!important;white-space:nowrap;}
@media(max-width:900px) {
  .block-container {padding:4.5rem 1rem 2.6rem;}
  .hero {min-height:440px;border-radius:22px;}
  .hero-body {padding:46px 27px;}
  .hero h1 {font-size:clamp(2.05rem,7vw,3.1rem);}
  .hero-photo {object-position:72% center;}
  .hero:before {background:linear-gradient(90deg,rgba(251,250,255,.96) 0%,rgba(251,250,255,.84) 48%,rgba(251,250,255,.12) 100%);}
  [data-testid="stForm"] {margin:22px auto 24px!important;padding:12px!important;border-radius:24px!important;}
  [data-testid="stForm"] [data-testid="stHorizontalBlock"] {flex-wrap:wrap!important;}
  [data-testid="stForm"] [data-testid="column"] {flex:1 1 100%!important;width:100%!important;}
  [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {width:100%!important;}
}


/* Remove the inner search input frame */
[data-testid="stForm"] [data-testid="stTextInput"] input,
[data-testid="stForm"] [data-testid="stTextInput"] input:focus {
    border: none !important;
    outline: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Remove the surrounding input wrapper frame */
[data-testid="stForm"] [data-testid="stTextInput"]
[data-baseweb="input"] {
    border: none !important;
    outline: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Remove the secondary input container frame */
[data-testid="stForm"] [data-testid="stTextInput"]
[data-baseweb="base-input"] {
    border: none !important;
    outline: none !important;
    background: transparent !important;
    box-shadow: none !important;
}




/* FINAL DESIGN REFINEMENTS: gentle indigo, airy space and answer-first hierarchy. */
:root {--ink:#242338;--muted:#747489;--indigo:#666AB0;--violet:#8581C4;--line:#E9E8F3;}
[data-testid="stAppViewContainer"] {background:#FCFCFE;}
.hero {border-radius:28px;box-shadow:0 14px 42px rgba(49,47,90,.055);min-height:clamp(440px,39vw,560px);margin-bottom:0!important;}
.hero-body {max-width:790px;}
.hero h1 {font-size:clamp(2.35rem,4vw,4.3rem);}
/* Make the entire search form one pill and eliminate BOTH nested input outlines. */
[data-testid="stForm"] {margin:38px auto 26px!important;max-width:1100px!important;padding:10px 13px!important;border:1px solid #E7E6F2!important;border-radius:999px!important;background:white!important;box-shadow:0 12px 35px rgba(55,53,108,.07)!important;}
[data-testid="stForm"] [data-testid="stTextInput"],
[data-testid="stForm"] [data-testid="stTextInput"] > div,
[data-testid="stForm"] [data-baseweb="input"],
[data-testid="stForm"] [data-baseweb="base-input"],
[data-testid="stForm"] [data-baseweb="input"]:focus-within,
[data-testid="stForm"] input,
[data-testid="stForm"] input:focus {border:0!important;outline:0!important;box-shadow:none!important;background:transparent!important;}
[data-testid="stForm"] input {font-size:1rem!important;min-height:56px!important;padding-left:18px!important;}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {gap:12px!important;align-items:center!important;}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {border-radius:999px!important;min-width:142px!important;min-height:54px!important;}
/* Main answer is intentionally prominent, with ONE beautiful container. */
.answer-shell {margin:20px 0 25px;padding:30px clamp(22px,3vw,38px);background:linear-gradient(135deg,#FFFFFF 0%,#FCFBFF 100%);border:1px solid #EAE8F4;border-radius:28px;box-shadow:0 14px 40px rgba(64,61,125,.055);}
.answer-eyebrow {font-size:.7rem;letter-spacing:.13em;font-weight:800;text-transform:uppercase;color:#696BAE;margin-bottom:12px;}
.answer-heading {font-family:'Manrope',sans-serif;color:#242338;font-size:1.5rem;font-weight:800;letter-spacing:-.04em;margin-bottom:21px;}
.answer-shell .answer-text {font-size:1rem;color:#302F44;line-height:1.94;white-space:pre-wrap;overflow-wrap:anywhere;}
.answer-footnote {color:#77778A;font-size:.8rem;line-height:1.65;margin-top:19px;padding-top:16px;border-top:1px solid #EFEEF6;}
.answer-followup {margin:12px 0 18px;color:#77778A;font-size:.82rem;}
/* Compact soft KPI frames like Cosmos, but not busy. */
[data-testid="stMetric"] {border:1px solid #EBEAF4!important;background:#FFFFFF!important;border-radius:22px!important;box-shadow:0 6px 19px rgba(64,61,125,.035)!important;padding:19px 20px!important;}
[data-testid="stMetricValue"] {color:#51558D!important;}
[data-testid="stExpander"] {border:1px solid #ECEAF5!important;border-radius:20px!important;background:#fff!important;box-shadow:none!important;}
[data-testid="stExpander"] summary {border-radius:20px!important;}
.feature-symbol {color:#8180B9;}
.feature-desc {line-height:1.8;}
.rule {margin:40px 0;}
@media(max-width:900px) {
  [data-testid="stForm"] {margin:24px auto 20px!important;border-radius:24px!important;}
  .answer-shell {padding:23px 19px;border-radius:22px;}
}
</style>""", unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def corpus_summary() -> dict:
    path = ROOT / "data" / "processed" / "chunk_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def qrels_summary() -> dict:
    path = ROOT / "data" / "evaluation" / "alignment_report.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (ValueError, OSError):
        return {}


def approved_source(url: str) -> str | None:
    parts = urlsplit(url or "")
    if (parts.scheme == "https" and parts.hostname == "www.federalreserve.gov"
            and parts.path.startswith("/monetarypolicy/files/")
            and parts.path.lower().endswith(".pdf")):
        return url
    return None


def dumps(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def set_example(example: str) -> None:
    st.session_state["question"] = example
    st.session_state["draft_question"] = example


def groq_key() -> str:
    """Read Streamlit's server-side secret without exposing it to users."""
    import os
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", ""))
    except (FileNotFoundError, OSError):
        return os.getenv("GROQ_API_KEY", "")


def launch() -> None:
    question = st.session_state.get("question", "").strip()
    if not question:
        st.error("Please enter a question before starting research.")
        return
    if st.session_state["use_llm"] and not groq_key() and not st.session_state["model_name"].strip():
        st.error("Please select a local Ollama model.")
        return
    st.session_state["draft_question"] = question
    try:
        started = perf_counter()
        timings = {}
        periods = detect_periods(question)
        mode = st.session_state["search_mode"]
        if mode == "automatic":
            mode = "temporal" if periods and len(periods) <= st.session_state["context_k"] else "classic"
        if mode == "temporal" and (not periods or len(periods) > st.session_state["context_k"]):
            st.warning("Temporal search is not suitable for this query/budget. Using classic search.")
            mode = "classic"
        with st.spinner("Retrieving and reranking FOMC evidence..."):
            if mode == "temporal":
                t0 = perf_counter()
                passages = retrieve_temporal(
                    question, per_date=2, pool_size=20,
                    max_passages=st.session_state["context_k"],
                )
                timings["temporal_retrieval_and_rerank_s"] = round(perf_counter() - t0, 3)
                if not passages:
                    raise RuntimeError("No indexed passages found for the selected periods.")
            else:
                passages = retrieve(
                    question,
                    method=st.session_state["retrieval_method"],
                    reranker=st.session_state["reranker_method"],
                    top_k=20,
                    context_k=st.session_state["context_k"],
                    timings=timings,
                )
        if st.session_state["use_llm"]:
            with st.spinner("Qwen is composing an evidence-grounded answer..."):
                if groq_key():
                    generation = generate_groq(
                        question, passages,
                        api_key=groq_key(),
                        structured=st.session_state["structured_generation"],
                        periods=periods if mode == "temporal" else [],
                    )
                elif st.session_state["structured_generation"]:
                    generation = generate_with_coverage(
                        question, passages,
                        model=st.session_state["model_name"].strip(),
                        periods=periods if mode == "temporal" else [],
                    )
                else:
                    generation = generate(question, passages, model=st.session_state["model_name"].strip())
        else:
            generation = {
                "answer": "Local generation is disabled. Explore the retrieved evidence.",
                "model": None, "abstained": False,
            }
        timings["generation_s"] = round(generation.get("generation_timing", {}).get("wall_s", 0), 3)
        timings["total_s"] = round(perf_counter() - started, 3)
        st.session_state["rag_result"] = {
            **generation,
            "question": question,
            "periods": periods,
            "mode": mode,
            "retrieval": "temporal_hybrid_rrf" if mode == "temporal" else st.session_state["retrieval_method"],
            "reranking": "colbert" if mode == "temporal" else st.session_state["reranker_method"],
            "structured": bool(st.session_state["use_llm"] and st.session_state["structured_generation"]),
            "passages": passages,
            "timings": timings,
        }
        st.session_state.pop("rag_explanation", None)
        st.rerun()
    except Exception as exc:
        st.error(f"The pipeline could not complete this query: {exc}")
        st.caption("Check the indexed dates, local models and Ollama service. Earlier results are preserved.")



# Single page — controls remain in the sidebar, analysis remains in one flow.
with st.sidebar:
    st.markdown('<div class="brand">FOMC <span>RAG Lab</span></div>', unsafe_allow_html=True)
    st.caption("Evidence-first research workspace")
    st.markdown("### Research settings")
    st.selectbox("Search mode", ["automatic", "classic", "temporal"], key="search_mode",
                 format_func={"automatic":"Automatic · By question", "classic":"Classic · Entire corpus", "temporal":"Temporal · Specific periods"}.get)
    st.selectbox("Retrieval strategy", list(RETRIEVAL_LABELS), key="retrieval_method", format_func=RETRIEVAL_LABELS.get)
    st.selectbox("Reranking method", list(RERANK_LABELS), key="reranker_method", format_func=RERANK_LABELS.get)
    if st.session_state['context_k'] not in (5, 6):
        st.session_state['context_k'] = 6
    st.selectbox("Passages finaux après reranking", [5, 6], key="context_k")
    st.caption("20 candidats reclassés, puis 5 ou 6 passages transmis au LLM.")
    st.markdown("### Language model")
    st.toggle("Enable Qwen response", key="use_llm")
    if groq_key():
        st.caption("Cloud generation: Groq / Qwen")
    else:
        st.caption("Local generation: Ollama (no Groq secret found)")
    if not groq_key():
        st.text_input("Ollama model", key="model_name", disabled=not st.session_state["use_llm"])
    st.toggle("Structured answer & evidence audit", key="structured_generation", disabled=not st.session_state["use_llm"])
    st.markdown('<div class="sidebar-note">Temporal mode uses Hybrid RRF + ColBERT for each detected period. Classic settings are preserved separately.</div>', unsafe_allow_html=True)
    st.markdown("### Corpus")
    corpus, qrels = corpus_summary(), qrels_summary()
    a,b = st.columns(2)
    a.metric("PDF documents", corpus.get("documents", "—"))
    b.metric("Chunks", corpus.get("chunks", "—"))
    c,d = st.columns(2)
    c.metric("Questions", qrels.get("questions", "—"))
    d.metric("Silver qrels", qrels.get("judgments", "—"))

# Hero: an original image, not a copyrighted watermarked stock photo.
st.markdown(
    '<div class="hero">'
    + (f'<img class="hero-photo" src="{HERO_URL}" alt="Abstract financial market visualization"/>' if HERO_URL else '')
    + '<div class="hero-body"><div class="eyebrow">Research · FOMC 2021—2026</div>'
      '<h1>Explore FOMC<br>decisions with <em>evidence</em></h1>'
      '<p>Search Federal Reserve statements and minutes, compare monetary policy decisions, '
      'and trace answers back to their documentary sources.</p></div></div>',
    unsafe_allow_html=True,
)
# The only question entry is a compact, Cosmos-inspired search bar.
# The form sits BELOW the hero; Enter also submits the question.
with st.form("research_question_form"):
    search_col, submit_col = st.columns([5.8, 1.3], gap="small", vertical_alignment="center")
    with search_col:
        st.text_input(
            "Ask a research question",
            key="question",
            label_visibility="collapsed",
            placeholder="Ask about FOMC policy, rates, inflation or market impact...",
        )
    with submit_col:
        submitted = st.form_submit_button("Search →", type="primary", use_container_width=True)
if submitted:
    launch()
with st.expander("Try an example question"):
    for i, example in enumerate(EXAMPLES):
        st.button(example, key=f"example_{i}", on_click=set_example, args=(example,), use_container_width=True)

run = st.session_state.get("rag_result")
if run:
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    # Answer FIRST: escape user/model content before inserting in HTML.
    safe_answer = html.escape(str(run.get("answer", "No answer generated.")))
    st.markdown(
        '<div class="answer-shell">'
        '<div class="answer-eyebrow">Research output · Answer & citations</div>'
        '<div class="answer-heading">Evidence-grounded response</div>'
        '<div class="answer-text">' + safe_answer + '</div>'
        '<div class="answer-footnote">Citation identifiers are checked structurally. '
        'Automatic verification of factual or semantic support is not performed.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if run.get("abstained"):
        st.warning("The model reported insufficient evidence.")
    if run.get("model"):
        st.caption("Generation model · " + str(run["model"]))
    st.download_button(
        "Export answer & metadata (.json)", dumps(run),
        file_name="fomc_rag_result.json", mime="application/json",
    )
    st.markdown('<div class="section-title">Your results</div>', unsafe_allow_html=True)
    st.caption("Question: " + run["question"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Passages selected", len(run["passages"]))
    m2.metric("Retrieval", {"temporal_hybrid_rrf":"Temporal RRF", "hybrid_rrf":"Hybrid RRF", "bm25":"BM25", "dense":"Dense E5"}.get(run["retrieval"], run["retrieval"]))
    m3.metric("Reranking", {"colbert":"ColBERT", "cross_encoder":"Cross-Encoder", "bm25":"BM25", "none":"None"}.get(run["reranking"], run["reranking"]))
    with st.expander("Pipeline performance", expanded=False):
        timing=run.get("timings",{})
        p1,p2,p3,p4=st.columns(4)
        if run.get("mode")=="temporal":
            p1.metric("Periods detected",len(run.get("periods", [])))
            p2.metric("Temporal RRF + ColBERT",f"{timing.get('temporal_retrieval_and_rerank_s',0):.2f} s")
            p3.metric("Passages selected",len(run["passages"]))
        else:
            p1.metric("Model setup",f"{timing.get('assets_s',0):.2f} s")
            p2.metric("Retrieval",f"{timing.get('retrieval_s',0):.2f} s")
            p3.metric("Reranking",f"{timing.get('reranking_s',0):.2f} s")
        p4.metric("LLM generation",f"{timing.get('generation_s',0):.2f} s" if run.get("model") else "Disabled")
        st.markdown(f"**Total time: {timing.get('total_s',0):.2f} seconds**")
        gen=run.get("generation_timing",{})
        if gen and run.get("model"):
            st.dataframe([{"Load (s)":gen.get("load_s"),"Prompt processing (s)":gen.get("prompt_eval_s"),"Generation (s)":gen.get("eval_s"),"Input tokens":gen.get("prompt_tokens"),"Output tokens":gen.get("generated_tokens")}],hide_index=True,use_container_width=True)
            st.caption("Generation times are reported when available.")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explore the research workflow</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    for col, symbol, heading, desc in zip(cols,["↗","≋","❞"],["Retrieval","Reranking","Citations"],[
        "Find relevant passages using BM25, E5 and Hybrid RRF.",
        "Prioritize supporting evidence with ColBERT or a cross-encoder.",
        "Trace answers to PDF sources and review evidence audits.",
    ]):
        with col:
            st.markdown(f'<div class="feature-symbol">{symbol}</div><div class="feature-title">{heading}</div><div class="feature-desc">{desc}</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sources & verification</div>',unsafe_allow_html=True)
    source_tab, explain_tab, audit_tab=st.tabs(["Passages & PDFs","Explainability","Evidence audit"])
    with source_tab:
        st.caption("Original passages, PDF pages and raw retrieval/reranking scores.")
        for passage in run["passages"]:
            rank=passage["rank"]
            pages=", ".join(str(p) for p in passage.get("pages",[])) or "—"
            with st.expander(f"SOURCE {rank:02d} · {passage.get('document','Document')} · p. {pages}"):
                st.markdown(f'<span class="source-marker">SOURCE {rank:02d}</span>',unsafe_allow_html=True)
                st.write(passage.get("text",""))
                st.caption("Chunk: "+str(passage.get("chunk_id","—")))
                s1,s2=st.columns(2)
                s1.metric("Retrieval score",f"{float(passage.get('retrieval_score',0)):.4f}")
                s2.metric("Reranking score",f"{float(passage.get('rerank_score',0)):.4f}")
                if passage.get("evidence_intent"):
                    st.caption("Intent: "+passage["evidence_intent"]+" · Period: "+str(passage.get("target_period","")))
                url=approved_source(passage.get("source_url",""))
                if url:
                    st.link_button("Open official PDF ↗",url)
                else:
                    st.caption("Official PDF link unavailable.")
    with explain_tab:
        if run.get("mode")=="temporal":
            st.caption("Temporal retrieval uses per-period RRF and ColBERT; raw scores are not probabilities.")
            for p in run["passages"]:
                st.write(f"SOURCE {p['rank']} · {p.get('target_period','')} · {p.get('evidence_intent','')} · ColBERT {float(p.get('rerank_score',0)):.3f}")
        else:
            if st.button("Compute detailed scores",key="detailed_scores"):
                try:
                    with st.spinner("Computing diagnostics..."):
                        st.session_state["rag_explanation"]=build_explanation(run,with_scores=True)
                except Exception as exc:
                    st.error(f"Diagnostics unavailable: {exc}")
            try:
                explanation=st.session_state.get("rag_explanation")
                if explanation is None:
                    explanation=build_explanation(run,with_scores=False)
                citation=explanation.get("citation_audit",{})
                c1,c2=st.columns(2)
                c1.metric("Valid citation IDs",len(citation.get("valid_citation_ids",[])))
                c2.metric("Unknown citations",len(citation.get("invalid_citation_ids",[])))
                diagnostics=explanation.get("retrieval_diagnostics")
                if diagnostics:
                    rows=[]
                    for x in diagnostics.get("passages",[]):
                        rows.append({"Final rank":x.get("rank_after_reranking"),"Chunk":x.get("chunk_id"),"BM25":x.get("bm25_score"),"BM25 rank":x.get("bm25_rank_full"),"E5 cosine":x.get("dense_cosine_score"),"E5 rank":x.get("dense_rank_full"),"RRF BM25":x.get("rrf_from_bm25_top20"),"RRF E5":x.get("rrf_from_dense_top20"),"Reranker":x.get("rerank_score_recorded")})
                    st.dataframe(rows,hide_index=True,use_container_width=True)
                st.download_button("Export explainability (.md)",markdown_report(explanation),file_name="fomc_explanation.md",mime="text/markdown")
            except Exception as exc:
                st.error(f"Explanation unavailable: {exc}")
        st.caption("Scores from different models are not directly comparable.")
    with audit_tab:
        if not run.get("structured"):
            st.info("Enable structured answers in the sidebar and rerun the question.")
        else:
            coverage=run.get("coverage_audit",{})
            citation=run.get("citation_audit",{})
            c1,c2=st.columns(2)
            c1.metric("Sections completed & cited",f"{coverage.get('complete_sections',0)}/{coverage.get('total_sections',0)}")
            c2.metric("Unknown citations",len(citation.get("invalid_citation_ids",[])))
            rows=[{"Section":x.get("title",k),"Complete":x.get("complete",False),"Sources":str(x.get("source_ids",[])),"Issues":"; ".join(x.get("errors",[]))} for k,x in coverage.get("sections",{}).items()]
            if rows:
                st.dataframe(rows,hide_index=True,use_container_width=True)
            try:
                plan=run.get("plan") or build_sections(run.get("periods",[]) if run.get("mode")=="temporal" else [],run["question"])
                evidence=audit_evidence(structured=run["structured_answer"],passages=run["passages"],periods=run.get("periods",[]) if run.get("mode")=="temporal" else [],plan=plan)
                st.metric("Heuristic sections without errors",f"{evidence['sections_without_blocking_issues']}/{evidence['total_sections']}")
                for key,item in evidence.get("sections",{}).items():
                    with st.expander(f"{key} · {len(item.get('issues',[]))} issue(s), {len(item.get('warnings',[]))} warning(s)"):
                        st.write(item.get("claim",""))
                        st.caption("Sources: "+str(item.get("source_ids",[])))
                        for issue in item.get("issues",[]):st.error(issue)
                        for warning in item.get("warnings",[]):st.warning(warning)
                st.download_button("Export evidence audit (.json)",dumps(evidence),file_name="fomc_evidence_audit.json",mime="application/json")
            except Exception as exc:
                st.error(f"Audit unavailable: {exc}")
            st.caption("Heuristic checks do not establish semantic or factual faithfulness.")
else:
    st.markdown('<div class="rule"></div><p class="subtle">Results, passages and evidence audits will appear here after your first question.</p>',unsafe_allow_html=True)

st.markdown('<div class="rule"></div><div class="section-title">Dataset & evaluation</div>',unsafe_allow_html=True)
corpus,qrels=corpus_summary(),qrels_summary()
s1,s2,s3,s4=st.columns(4)
s1.metric("PDF documents",corpus.get("documents","—"))
s2.metric("Chunks",corpus.get("chunks","—"))
s3.metric("Evaluation questions",qrels.get("questions","—"))
s4.metric("Silver qrels",qrels.get("judgments","—"))
with st.expander("Evaluation methodology & limitations"):
    st.write("Retrieval: BM25, Dense E5, Hybrid RRF. Reranking: ColBERT MaxSim and cross-encoder.")
    st.write("Metrics: Recall@10/20, MRR@10, nDCG@10; page-level silver qrels may be incomplete.")
    st.caption("Results and audits must not be treated as automatic proof of semantic faithfulness.")
    report_path=ROOT/'results'/'RAPPORT.md'
    if report_path.exists():
        st.markdown(report_path.read_text(encoding='utf-8'))
