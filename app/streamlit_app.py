"""
Beverage Recognition — Streamlit MVP
Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from inference.predict import Predictor
from inference.retriever import ProductRetriever
from utils.config_loader import load_config, resolve_path


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Beverage Recognizer",
    page_icon="🥤",
    layout="wide",
)


# ── Styles ────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* Base */
.stApp { background-color: #0d1117; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
}
#MainMenu, footer, header { visibility: hidden; }
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important; }

/* App header */
.app-header { margin-bottom: 1.75rem; border-bottom: 1px solid #21262d; padding-bottom: 1rem; }
.app-title  { font-size: 1.35rem; font-weight: 700; color: #e6edf3; margin: 0; }
.app-sub    { font-size: 0.83rem; color: #8b949e; margin: 0.2rem 0 0 0; }

/* Card */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.85rem;
}

/* Product name + confidence badge */
.product-row  { display: flex; align-items: baseline; gap: 0.65rem; margin-bottom: 0.2rem; }
.product-name { font-size: 1.75rem; font-weight: 700; color: #e6edf3; line-height: 1.2; }
.conf-badge   { font-size: 0.95rem; font-weight: 600; padding: 0.15rem 0.55rem; border-radius: 20px; }
.conf-green   { background: rgba(63,185,80,0.12);  color: #3fb950; }
.conf-orange  { background: rgba(210,153,34,0.12); color: #d29922; }
.conf-red     { background: rgba(248,81,73,0.12);  color: #f85149; }

/* Meta line under product name */
.meta-row { font-size: 0.83rem; color: #8b949e; margin-bottom: 1rem; }
.meta-val { color: #c9d1d9; }

/* Detected flavor pill */
.flavor-pill {
    display: inline-block;
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.82rem;
    color: #58a6ff;
    margin-bottom: 0.85rem;
}

/* Info key-value grid */
.info-grid {
    display: grid;
    grid-template-columns: 130px 1fr;
    row-gap: 0.35rem;
    column-gap: 0.5rem;
    font-size: 0.865rem;
    margin-bottom: 0.9rem;
}
.info-key { color: #8b949e; }
.info-val { color: #e6edf3; }

/* Section label */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #6e7681;
    margin-bottom: 0.5rem;
}

/* Size chips */
.size-list { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.1rem; }
.size-chip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.8rem;
    color: #8b949e;
}
.size-chip-active {
    background: rgba(88,166,255,0.1);
    border-color: rgba(88,166,255,0.35);
    color: #58a6ff;
}

/* Prediction bars */
.pred-item   { margin-bottom: 0.65rem; }
.pred-header { display: flex; justify-content: space-between; margin-bottom: 0.2rem; }
.pred-class  { font-size: 0.84rem; color: #c9d1d9; }
.pred-pct    { font-size: 0.84rem; color: #6e7681; }
.bar-track   { background: #21262d; border-radius: 3px; height: 5px; overflow: hidden; }
.bar-fill         { height: 5px; border-radius: 3px; background: #1f6feb; }
.bar-fill-primary { height: 5px; border-radius: 3px; background: #3fb950; }

/* Supported classes pills */
.class-pill {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.8rem;
    color: #8b949e;
    margin: 0.2rem;
}

/* Warning / error boxes */
.warn-box {
    background: rgba(210,153,34,0.07);
    border: 1px solid rgba(210,153,34,0.3);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    color: #d29922;
    margin-bottom: 0.75rem;
}
</style>
"""


# ── Pipeline loader ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_pipeline():
    cfg = load_config()
    ckpt_path = resolve_path(cfg["paths"]["models"]) / cfg["training"]["checkpoint_filename"]
    predictor = Predictor(ckpt_path, cfg)
    retriever = ProductRetriever(cfg)
    return predictor, retriever, cfg


# ── Render helpers ────────────────────────────────────────────────────────────

def _conf_css_class(conf: float, threshold: float) -> str:
    if conf >= 0.85:
        return "conf-green"
    if conf >= threshold:
        return "conf-orange"
    return "conf-red"


def _render_product_card(product: dict, prediction: dict) -> None:
    volume_ml: int | None = prediction["volume_ml"]
    flavor: str | None    = prediction["flavor"]

    html: list[str] = ['<div class="card">']

    # Type · origin
    html.append(
        f'<div class="meta-row">'
        f'<span class="info-key">{product["product_type"]}</span>'
        f'&nbsp;·&nbsp;'
        f'<span class="meta-val">{product["brand_origin"]}</span>'
        f'</div>'
    )

    # Detected flavor
    if flavor:
        html.append(f'<span class="flavor-pill">✦ {flavor} variant detected from label</span>')

    # Manufacturer info
    html.append('<div class="info-grid">')
    html.append(f'<span class="info-key">Manufacturer</span><span class="info-val">{product["manufacturer"]}</span>')
    if product.get("local_manufacturer"):
        html.append(f'<span class="info-key">Bottled by</span><span class="info-val">{product["local_manufacturer"]}</span>')
    else:
        html.append(f'<span class="info-key">Bottled by</span><span class="info-val">Imported</span>')
    html.append(f'<span class="info-key">Flavors</span><span class="info-val">{", ".join(product["flavors"])}</span>')
    html.append('</div>')

    # Packaging sizes
    html.append('<div class="section-label">Available sizes</div>')
    html.append('<div class="size-list">')
    for v in product["packaging"]:
        active = "size-chip-active" if v["volume_ml"] == volume_ml else ""
        html.append(f'<span class="size-chip {active}">{v["type"]} · {v["volume_ml"]} ml</span>')
    html.append('</div>')

    # Ingredients
    html.append('<div class="section-label" style="margin-top:0.75rem;">Ingredients</div>')
    html.append(f'<div style="font-size:0.84rem;color:#8b949e;line-height:1.6;">{", ".join(product["ingredients"])}</div>')

    html.append('</div>')
    st.markdown("\n".join(html), unsafe_allow_html=True)

    if product.get("website"):
        st.markdown(
            f'<a href="{product["website"]}" target="_blank" '
            f'style="color:#58a6ff;font-size:0.875rem;text-decoration:none;">Official website ↗</a>',
            unsafe_allow_html=True,
        )


def _render_top_k(top_k: list[dict]) -> None:
    html: list[str] = ['<div class="card">', '<div class="section-label">Top predictions</div>']
    for i, entry in enumerate(top_k):
        pct = int(entry["confidence"] * 100)
        bar_cls = "bar-fill-primary" if i == 0 else "bar-fill"
        html.append(
            f'<div class="pred-item">'
            f'<div class="pred-header">'
            f'<span class="pred-class">{entry["class"]}</span>'
            f'<span class="pred-pct">{pct}%</span>'
            f'</div>'
            f'<div class="bar-track"><div class="{bar_cls}" style="width:{pct}%"></div></div>'
            f'</div>'
        )
    html.append('</div>')
    st.markdown("\n".join(html), unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="app-header">'
        '<p class="app-title">🥤 Beverage Recognizer</p>'
        '<p class="app-sub">Upload a photo of a packaged beverage to identify it.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    predictor, retriever, cfg = load_pipeline()
    threshold: float = cfg["inference"]["confidence_threshold"]

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded is None:
        pills = "".join(
            f'<span class="class-pill">{c}</span>'
            for c in sorted(retriever.known_classes)
        )
        st.markdown(
            f'<div style="margin-top:2rem">'
            f'<div class="section-label">Supported beverages</div>'
            f'{pills}'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    image = Image.open(uploaded).convert("RGB")
    img_col, result_col = st.columns([1, 1], gap="large")

    with img_col:
        st.image(image, use_column_width=True)

    with result_col:
        with st.spinner("Analysing…"):
            prediction = predictor.predict(image)

        conf    = prediction["confidence"]
        css_cls = _conf_css_class(conf, threshold)

        st.markdown(
            f'<div class="product-row">'
            f'<span class="product-name">{prediction["class"]}</span>'
            f'<span class="conf-badge {css_cls}">{conf * 100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if conf < threshold:
            st.markdown(
                f'<div class="warn-box">'
                f'⚠ Confidence too low ({conf * 100:.1f}% &lt; {threshold * 100:.0f}%). '
                f'Try a clearer, well-lit photo with the label facing the camera.'
                f'</div>',
                unsafe_allow_html=True,
            )
            _render_top_k(prediction["top_k"])
            return

        product = retriever.get(prediction["class"])
        if product is None:
            st.error("Product details not found in the knowledge base.")
            _render_top_k(prediction["top_k"])
            return

        _render_product_card(product, prediction)
        _render_top_k(prediction["top_k"])


if __name__ == "__main__":
    main()
