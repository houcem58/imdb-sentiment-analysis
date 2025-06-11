"""
IMDb Sentiment Analysis — live demo via HuggingFace Inference API
Calls distilbert-base-uncased-finetuned-sst-2-english (same model used in the notebook).
No local model download — inference runs on HuggingFace infrastructure.
"""
import requests
import streamlit as st

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


def query(text: str) -> list | dict:
    hf_token = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    resp = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text[:512]},
        timeout=30,
    )
    return resp.json()


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
    with st.spinner("Running inference on HuggingFace…"):
        try:
            result = query(review.strip())
        except requests.Timeout:
            st.error("Request timed out. Try again in a moment.")
            st.stop()
        except Exception as exc:
            st.error(f"Inference error: {exc}")
            st.stop()

    # Handle "model loading" response
    if isinstance(result, dict) and "error" in result:
        st.warning(
            f"Model is warming up on HuggingFace servers — wait 20 seconds and try again.  \n"
            f"*(Error: {result['error']})*"
        )
        st.stop()

    # Normalise response shape: [[{label, score}]] or [{label, score}]
    scores_raw = result[0] if (isinstance(result, list) and isinstance(result[0], list)) else result
    score_map = {r["label"]: r["score"] for r in scores_raw}

    pos = score_map.get("POSITIVE", 0.0)
    neg = score_map.get("NEGATIVE", 0.0)

    st.divider()

    if pos >= neg:
        st.success(f"## 😊 POSITIVE")
        conf, other = pos, neg
    else:
        st.error(f"## 😞 NEGATIVE")
        conf, other = neg, pos

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{conf:.1%}")
    c2.metric("Positive", f"{pos:.1%}")
    c3.metric("Negative", f"{neg:.1%}")

    # Confidence bar
    bar_filled = int(conf * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    st.code(f"Confidence  [{bar}]  {conf:.1%}", language=None)

    st.caption(f"Characters analysed: {min(len(review), 512)} / 512 (DistilBERT max)")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Stack:** HuggingFace Transformers · Streamlit · Python  |  "
    "**Model:** DistilBERT fine-tuned on Stanford Sentiment Treebank  |  "
    "**Author:** [Houcem Hammami](https://github.com/houcem58)"
)
