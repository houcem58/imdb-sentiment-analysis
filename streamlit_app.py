"""
IMDb Sentiment Analysis — live demo
Primary: distilbert-base-uncased-finetuned-sst-2-english via HuggingFace Inference API
Fallback: VADER (pure-Python lexicon) when HF API is unreachable
"""
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.metric-box {
    border-radius: 8px; padding: 1rem;
    text-align: center; font-size: 1.1rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

EXAMPLES = [
    "Christopher Nolan has outdone himself. Inception is a masterpiece of science fiction cinema — layered, ambitious, and visually stunning.",
    "Terrible film. The plot was incoherent, the acting was wooden, and I fell asleep halfway through. A complete waste of two hours.",
    "An okay movie. Some parts were interesting but overall it felt too long and the ending was entirely predictable.",
    "One of the greatest films ever made. Every frame is deliberate, every line perfectly delivered. I was completely absorbed.",
    "The sequel nobody asked for. The original had soul; this one has budget. CGI spectacle with zero emotional payoff.",
]

_vader = SentimentIntensityAnalyzer()


def _vader_classify(text: str) -> tuple[float, float, str]:
    scores = _vader.polarity_scores(text)
    pos = scores["pos"]
    neg = scores["neg"]
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return pos, neg, label


def query_hf(text: str) -> list | dict | None:
    """Call HuggingFace Inference API. Returns None if unreachable."""
    hf_token = ""
    try:
        hf_token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        pass
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        resp = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": text[:512]},
            timeout=15,
        )
        return resp.json()
    except (requests.ConnectionError, requests.Timeout):
        return None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 IMDb Sentiment Analysis")
st.caption(
    f"[`{MODEL}`](https://huggingface.co/{MODEL}) via HuggingFace Inference API · "
    f"Part of the [imdb-sentiment-analysis](https://github.com/houcem58/imdb-sentiment-analysis) NLP study"
)
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
review = st.text_area(
    "Movie Review",
    placeholder="Enter any movie review to classify its sentiment…",
    height=140,
    key="review_input",
)

with st.expander("Load an example review"):
    for ex in EXAMPLES:
        if st.button(ex[:80] + "…", key=ex, use_container_width=True):
            st.session_state["review_input"] = ex
            st.rerun()

analyse = st.button("Analyse Sentiment", type="primary", use_container_width=True,
                    disabled=not (review or "").strip())

# ── Inference ─────────────────────────────────────────────────────────────────
if analyse and review.strip():
    with st.spinner("Running inference…"):
        result = query_hf(review.strip())

    pos, neg, label, engine = None, None, None, None

    if result is None:
        # HF API unreachable — fall back to VADER
        pos, neg, label = _vader_classify(review.strip())
        engine = "VADER (offline fallback — HuggingFace API unreachable)"
    elif isinstance(result, dict) and "error" in result:
        if "loading" in result["error"].lower():
            st.warning(
                "Model is warming up on HuggingFace — wait 20 s and try again.  \n"
                f"*(Error: {result['error']})*"
            )
            # Show VADER while warming up
            pos, neg, label = _vader_classify(review.strip())
            engine = "VADER (shown while DistilBERT warms up)"
        else:
            st.error(f"HuggingFace API error: {result['error']}")
            st.stop()
    else:
        scores_raw = result[0] if (isinstance(result, list) and isinstance(result[0], list)) else result
        score_map = {r["label"]: r["score"] for r in scores_raw}
        pos = score_map.get("POSITIVE", 0.0)
        neg = score_map.get("NEGATIVE", 0.0)
        label = "POSITIVE" if pos >= neg else "NEGATIVE"
        engine = "DistilBERT (HuggingFace Inference API)"

    if pos is not None:
        st.divider()
        conf = pos if label == "POSITIVE" else neg

        if label == "POSITIVE":
            st.success("## 😊 POSITIVE")
        elif label == "NEGATIVE":
            st.error("## 😞 NEGATIVE")
        else:
            st.info("## 😐 NEUTRAL")

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{conf:.1%}")
        c2.metric("Positive", f"{pos:.1%}")
        c3.metric("Negative", f"{neg:.1%}")

        bar_filled = int(conf * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        st.code(f"Confidence  [{bar}]  {conf:.1%}", language=None)

        st.caption(f"Engine: {engine}  |  Characters: {min(len(review), 512)} / 512")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Stack:** HuggingFace Transformers · VADER · Streamlit · Python  |  "
    "**Model:** DistilBERT fine-tuned on Stanford Sentiment Treebank  |  "
    "**Author:** [Houcem Hammami](https://github.com/houcem58)"
)
