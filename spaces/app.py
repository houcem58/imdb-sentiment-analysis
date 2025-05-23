"""
IMDb Sentiment Analysis — HuggingFace Spaces demo
Uses the same DistilBERT SST-2 model applied in the notebook pipeline.
"""
import gradio as gr
from transformers import pipeline

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

sentiment = pipeline(
    "text-classification",
    model=MODEL_ID,
    return_all_scores=True,
    truncation=True,
    max_length=512,
)


def analyze(review: str):
    if not review.strip():
        return "⚠️ Please enter a movie review.", "", "", ""

    scores = {r["label"]: r["score"] for r in sentiment(review)[0]}
    pos = scores.get("POSITIVE", 0.0)
    neg = scores.get("NEGATIVE", 0.0)

    if pos >= neg:
        label   = "😊 POSITIVE"
        conf    = pos
        bar     = "🟢" * int(pos * 10) + "⬜" * (10 - int(pos * 10))
    else:
        label   = "😞 NEGATIVE"
        conf    = neg
        bar     = "🔴" * int(neg * 10) + "⬜" * (10 - int(neg * 10))

    detail = (
        f"**Positive:** {pos:.1%}   |   "
        f"**Negative:** {neg:.1%}   |   "
        f"Confidence bar: {bar}"
    )
    model_note = f"Model: `{MODEL_ID}` · Characters processed: {min(len(review), 512)}"

    return label, f"{conf:.1%}", detail, model_note


EXAMPLES = [
    ["Christopher Nolan has outdone himself. Inception is a masterpiece of science fiction cinema — layered, ambitious, and visually stunning."],
    ["Terrible film. The plot was incoherent, the acting was wooden, and I fell asleep halfway through. A complete waste of two hours."],
    ["An okay movie. Some scenes were interesting but overall it felt too long and the ending was predictable."],
    ["One of the greatest films ever made. Every frame is deliberate, every line perfectly delivered. I was completely absorbed."],
    ["The sequel nobody asked for. The original had soul; this one has budget. CGI spectacle with zero emotional payoff."],
]

with gr.Blocks(title="IMDb Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 IMDb Movie Review Sentiment Analysis
        **DistilBERT SST-2** fine-tuned on Stanford Sentiment Treebank ·
        Part of the [imdb-sentiment-analysis](https://github.com/houcem58/imdb-sentiment-analysis) NLP study.

        Enter any movie review to classify it as Positive or Negative with a confidence score.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            review_input = gr.Textbox(
                label="Movie Review",
                placeholder="Enter a movie review here…",
                lines=5,
                max_lines=12,
            )
            submit_btn = gr.Button("Analyse Sentiment", variant="primary")

        with gr.Column(scale=2):
            label_out   = gr.Textbox(label="Sentiment")
            conf_out    = gr.Textbox(label="Confidence")
            detail_out  = gr.Markdown(label="Score breakdown")
            model_out   = gr.Markdown()

    gr.Examples(
        examples=EXAMPLES,
        inputs=review_input,
        label="Example reviews — click to load",
    )

    submit_btn.click(
        fn=analyze,
        inputs=review_input,
        outputs=[label_out, conf_out, detail_out, model_out],
    )
    review_input.submit(
        fn=analyze,
        inputs=review_input,
        outputs=[label_out, conf_out, detail_out, model_out],
    )

    gr.Markdown(
        """
        ---
        **Stack:** HuggingFace Transformers · Gradio · Python
        **Model:** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
        **Author:** [Houcem Hammami](https://github.com/houcem58)
        """
    )

demo.launch()
