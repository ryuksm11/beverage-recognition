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


# ── Pipeline loader (cached across reruns) ────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    cfg = load_config()
    ckpt_path = resolve_path(cfg["paths"]["models"]) / cfg["training"]["checkpoint_filename"]
    predictor = Predictor(ckpt_path, cfg)
    retriever = ProductRetriever(cfg)
    return predictor, retriever, cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_color(conf: float, threshold: float) -> str:
    if conf >= 0.85:
        return "green"
    if conf >= threshold:
        return "orange"
    return "red"


def _render_product_card(product: dict, prediction: dict) -> None:
    """Render the main product information card."""
    volume_ml: int | None = prediction["volume_ml"]
    flavor: str | None = prediction["flavor"]

    # ── Type + origin ─────────────────────────────────────────────────────────
    st.markdown(
        f"**Type:** {product['product_type']} &nbsp;&nbsp; **Brand origin:** {product['brand_origin']}",
        unsafe_allow_html=True,
    )

    # ── OCR-detected flavor ───────────────────────────────────────────────────
    if flavor:
        st.info(f"**Detected flavor:** {flavor}")

    # ── Packaging sizes ───────────────────────────────────────────────────────
    matched_variant = next(
        (v for v in product["packaging"] if v["volume_ml"] == volume_ml),
        None,
    ) if volume_ml is not None else None

    if matched_variant:
        st.markdown(
            f"**Size (detected from label):** {matched_variant['type']} · {matched_variant['volume_ml']} ml"
        )
    else:
        st.markdown("**Available sizes**")
        for variant in product["packaging"]:
            st.markdown(f"- {variant['type']} · {variant['volume_ml']} ml")

    # ── Manufacturer info ─────────────────────────────────────────────────────
    st.markdown(f"**Manufacturer:** {product['manufacturer']}")
    if product.get("local_manufacturer"):
        st.caption(f"Bottled in India by: {product['local_manufacturer']}")
    else:
        st.caption("Imported (not bottled locally)")

    # ── Available flavors ─────────────────────────────────────────────────────
    st.markdown(f"**Available flavors:** {', '.join(product['flavors'])}")

    # ── Ingredients ───────────────────────────────────────────────────────────
    with st.expander("Ingredients"):
        st.write(", ".join(product["ingredients"]))

    # ── Website ───────────────────────────────────────────────────────────────
    if product.get("website"):
        st.markdown(f"[Official website]({product['website']})")


def _render_top_k(top_k: list[dict]) -> None:
    st.markdown("**Top predictions**")
    for entry in top_k:
        bar_pct = int(entry["confidence"] * 100)
        st.write(f"{entry['class']}")
        st.progress(bar_pct, text=f"{bar_pct}%")


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Beverage Recognizer")
    st.caption("Upload a photo of a packaged beverage to identify it.")

    predictor, retriever, cfg = load_pipeline()
    threshold: float = cfg["inference"]["confidence_threshold"]

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown("---")
        st.markdown(
            "Supported beverages: "
            + ", ".join(sorted(retriever.known_classes))
        )
        return

    image = Image.open(uploaded).convert("RGB")

    img_col, result_col = st.columns([1, 1], gap="large")

    with img_col:
        st.image(image, use_column_width=True)

    with result_col:
        with st.spinner("Analysing..."):
            prediction = predictor.predict(image)

        conf = prediction["confidence"]
        color = _confidence_color(conf, threshold)

        st.markdown(
            f"### {prediction['class']} &nbsp; "
            f"<span style='color:{color}'>{conf * 100:.1f}%</span>",
            unsafe_allow_html=True,
        )

        if conf < threshold:
            st.warning(
                f"Confidence too low ({conf * 100:.1f}% < {threshold * 100:.0f}%). "
                "Cannot reliably identify this beverage. Try a clearer photo."
            )
            st.markdown("---")
            _render_top_k(prediction["top_k"])
            return

        product = retriever.get(prediction["class"])
        if product is None:
            st.error("Product details not found in the knowledge base.")
            _render_top_k(prediction["top_k"])
            return

        _render_product_card(product, prediction)
        st.markdown("---")
        _render_top_k(prediction["top_k"])


if __name__ == "__main__":
    main()
