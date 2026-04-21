# app.py
# Run locally:
#   python -m streamlit run app.py

import io
import json
import re
import datetime as dt
import hashlib
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import matplotlib.pyplot as plt


# =============================
# STREAMLIT VERSION-SAFE IMAGE
# =============================
def st_image(img, **kwargs):
    """
    Streamlit changed st.image() sizing arg across versions:
    - newer: use_container_width
    - older: use_column_width
    This wrapper makes your app version-safe.
    """
    try:
        st.image(img, use_container_width=True, **kwargs)  # newer Streamlit
    except TypeError:
        st.image(img, use_column_width=True, **kwargs)     # older Streamlit


# =============================
# OPTIONAL DEPENDENCIES
# =============================
try:
    import cv2  # type: ignore
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False

try:
    import pytesseract  # type: ignore

    tesseract_from_env = os.getenv("TESSERACT_CMD")
    tesseract_auto = shutil.which("tesseract")

    if tesseract_from_env:
        pytesseract.pytesseract.tesseract_cmd = tesseract_from_env
    elif tesseract_auto:
        pytesseract.pytesseract.tesseract_cmd = tesseract_auto

    PYTESS_OK = bool(tesseract_from_env or tesseract_auto)
except Exception:
    PYTESS_OK = False


# =============================
# JSON-SAFE CONVERSION
# =============================
def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# =============================
# DATA STRUCTURES
# =============================
@dataclass
class AnalysisResult:
    overall_score: int
    subscores: Dict[str, int]
    suggestions: List[str]
    overlay_image: Image.Image
    saliency_heatmap: Image.Image
    debug: Dict[str, float]


# =============================
# IMAGE HELPERS
# =============================
def _pil_to_np_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def _np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _safe_resize(pil_img: Image.Image, max_side: int = 1400) -> Image.Image:
    w, h = pil_img.size
    s = min(1.0, max_side / max(w, h))
    return pil_img.resize((max(1, int(w * s)), max(1, int(h * s))))


def _safe_resize_for_store(pil_img: Image.Image, max_side: int = 900) -> Image.Image:
    """Smaller cap for session storage / previews to keep memory reasonable."""
    w, h = pil_img.size
    s = min(1.0, max_side / max(w, h))
    return pil_img.resize((max(1, int(w * s)), max(1, int(h * s))))


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    if OPENCV_OK:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)


def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_pil(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


# =============================
# VISUAL COMPLEXITY OVERLAY
# =============================
def _make_visual_complexity_overlay(rgb: np.ndarray, strength: float = 0.55) -> Image.Image:
    if not OPENCV_OK:
        return _np_rgb_to_pil(rgb)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    heat = np.abs(lap)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=5, sigmaY=5)

    heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    blended = rgb.astype(np.float32) * (1.0 - strength) + heat_color.astype(np.float32) * strength
    return _np_rgb_to_pil(blended)


# =============================
# SALIENCY (spectral residual)
# =============================
def _spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)

    fft = np.fft.fft2(g)
    mag = np.abs(fft)
    phase = np.angle(fft)
    log_mag = np.log(mag + 1e-8)

    avg = cv2.blur(log_mag.astype(np.float32), (3, 3)) if OPENCV_OK else log_mag
    res = log_mag - avg

    sal = np.abs(np.fft.ifft2(np.exp(res) * np.exp(1j * phase))) ** 2
    if OPENCV_OK:
        sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)

    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    return sal.astype(np.float32)


def _saliency_focus_metrics(sal: np.ndarray) -> Dict[str, float]:
    flat = sal.flatten().astype(np.float32)
    total = float(flat.sum()) + 1e-9
    p = flat / total
    N = p.size

    ent = -float((p[p > 0] * np.log(p[p > 0])).sum())
    ent_norm = ent / (np.log(N) + 1e-9)

    k = max(1, int(0.10 * N))
    top = np.partition(flat, -k)[-k:]
    top_energy = float(top.sum()) / total

    return {"sal_entropy": ent_norm, "top10_energy": top_energy}


def _saliency_to_heatmap_rgb(sal: np.ndarray) -> Image.Image:
    sal8 = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
    if OPENCV_OK:
        heat = cv2.applyColorMap(sal8, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return _np_rgb_to_pil(heat)
    heat = np.stack([sal8, sal8, sal8], axis=-1)
    return _np_rgb_to_pil(heat)


# =============================
# OCR (optional)
# =============================
_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _ocr_metrics(rgb: np.ndarray) -> Dict[str, float]:
    if not PYTESS_OK:
        return {"ocr_present": 0.0, "ocr_avg_conf": 0.0, "ocr_gibberish_ratio": 0.0}

    pil = _np_rgb_to_pil(rgb)
    try:
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    except Exception:
        return {"ocr_present": 0.0, "ocr_avg_conf": 0.0, "ocr_gibberish_ratio": 0.0}

    tokens, confs = [], []
    for i, txt in enumerate(data.get("text", [])):
        txt = (txt or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        tokens.append(txt)
        confs.append(conf)

    if not tokens:
        return {"ocr_present": 0.0, "ocr_avg_conf": 0.0, "ocr_gibberish_ratio": 0.0}

    gib = sum(1 for t in tokens if not _WORD_RE.search(t)) / max(1, len(tokens))
    avg_conf = (float(np.mean(confs)) / 100.0) if confs else 0.0

    return {
        "ocr_present": 1.0,
        "ocr_avg_conf": float(max(0.0, min(1.0, avg_conf))),
        "ocr_gibberish_ratio": float(max(0.0, min(1.0, gib))),
    }


# =============================
# BASE METRICS
# =============================
def _base_metrics(rgb: np.ndarray) -> Dict[str, float]:
    gray = _to_gray(rgb)

    contrast = float(np.std(gray) / 64.0)
    contrast = max(0.0, min(1.0, contrast))

    brightness = float(np.mean(gray) / 255.0)

    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 8.0)

    if OPENCV_OK:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        gy, gx = np.gradient(gray.astype(np.float32))
        grad = np.sqrt(gx * gx + gy * gy)
        lap_var = float(np.var(grad))

    clarity = (lap_var - 20.0) / 400.0
    clarity = max(0.0, min(1.0, clarity))

    return {
        "contrast": contrast,
        "brightness": brightness,
        "entropy": entropy,
        "clarity": clarity,
        "laplacian_var": lap_var,
    }


# =============================
# SCORING + SUGGESTIONS
# =============================
def _score_and_suggest(
    base: Dict[str, float],
    focus_m: Dict[str, float],
    ocr_m: Dict[str, float]
) -> Tuple[int, Dict[str, int], List[str]]:
    suggestions: List[str] = []

    clutter = int(round(100 * (1.0 - base["entropy"])))
    clarity = int(round(100 * base["clarity"]))
    contrast = int(round(100 * base["contrast"]))

    b = base["brightness"]
    brightness = int(round(100 * (1.0 - abs(b - 0.55) / 0.55)))
    brightness = max(0, min(100, brightness))

    focus_quality = (1.0 - focus_m["sal_entropy"]) * 0.55 + focus_m["top10_energy"] * 0.45
    focus = int(round(100 * max(0.0, min(1.0, focus_quality))))

    if ocr_m["ocr_present"] < 0.5:
        text = 90
    else:
        text_q = ocr_m["ocr_avg_conf"] * (1.0 - 0.85 * ocr_m["ocr_gibberish_ratio"])
        text = int(round(100 * max(0.0, min(1.0, text_q))))

    overall = int(round(
        0.26 * clutter +
        0.20 * clarity +
        0.18 * contrast +
        0.14 * brightness +
        0.12 * focus +
        0.10 * text
    ))
    overall = max(0, min(100, overall))

    if clutter < 70:
        suggestions.append("Reduce background clutter (simplify textures, remove extra objects, add whitespace).")
    if clarity < 65:
        suggestions.append("Improve clarity (avoid blur/noise; sharpen the main subject).")
    if contrast < 60:
        suggestions.append("Increase contrast between subject and background for easier recognition.")
    if brightness < 60:
        suggestions.append("Adjust brightness to a comfortable mid-range (avoid too dark or washed out).")
    if focus < 70:
        suggestions.append("Strengthen the focal point (increase subject size/contrast; reduce competing salient regions).")
    if ocr_m["ocr_present"] >= 0.5 and ocr_m["ocr_gibberish_ratio"] > 0.4:
        suggestions.append("Replace distorted/nonsensical text with clean readable text to avoid confusion.")

    if not suggestions:
        suggestions.append("Looks good: keep a single focal subject, simple background, and clear readable text (if any).")

    subs = {
        "Overall": overall,
        "Clutter": clutter,
        "Clarity": clarity,
        "Contrast": contrast,
        "Brightness": brightness,
        "Focus": focus,
        "Text": text,
    }
    return overall, subs, suggestions


# =============================
# ANALYSIS
# =============================
def analyze_image(img: Image.Image, overlay_strength: float) -> AnalysisResult:
    img = _safe_resize(img)
    rgb = _pil_to_np_rgb(img)

    base = _base_metrics(rgb)
    sal = _spectral_residual_saliency(_to_gray(rgb))
    focus_m = _saliency_focus_metrics(sal)
    sal_heat = _saliency_to_heatmap_rgb(sal)
    ocr_m = _ocr_metrics(rgb)

    overall, subs, sugg = _score_and_suggest(base, focus_m, ocr_m)
    overlay = _make_visual_complexity_overlay(rgb, strength=overlay_strength)

    debug = {**base, **focus_m, **ocr_m}
    return AnalysisResult(
        overall_score=overall,
        subscores=subs,
        suggestions=sugg,
        overlay_image=overlay,
        saliency_heatmap=sal_heat,
        debug=debug,
    )


# =============================
# OVERALL SCORE COLOR (ONLY)
# =============================
def overall_score_color(score: int) -> str:
    if score >= 80:
        return "#2ecc71"
    if score >= 60:
        return "#f1c40f"
    return "#e74c3c"


# =============================
# UI TABLES
# =============================
def render_score_table(subscores: Dict[str, int]):
    order = ["Overall", "Clutter", "Clarity", "Contrast", "Brightness", "Focus", "Text"]

    meaning = {
        "Overall": "Combined estimate (higher is better).",
        "Clutter": "Lower complexity is better.",
        "Clarity": "Sharpness proxy (higher is better).",
        "Contrast": "Foreground/background separation.",
        "Brightness": "Comfortable mid-range brightness.",
        "Focus": "Attention concentration (less dispersion).",
        "Text": "OCR confidence with gibberish penalty.",
    }

    def status_for(score: int):
        if score >= 80:
            return "✅", "🟢", "Good"
        if score >= 60:
            return "⚠️", "🟡", "Needs attention"
        return "❌", "🔴", "High risk"

    def bar(score: int, width: int = 14) -> str:
        filled = int(round((score / 100) * width))
        return "█" * filled + "░" * (width - filled)

    rows = []
    for k in order:
        if k not in subscores:
            continue
        v = int(subscores[k])
        icon, dot, status = status_for(v)
        rows.append({
            "Icon": icon,
            "Level": dot,
            "Metric": k,
            "Score": v,
            "Bar": bar(v),
            "Status": status,
            "Meaning": meaning.get(k, "")
        })

    df = pd.DataFrame(rows)

    st.markdown(
        """
<div class="ja-card ja-scores-header">
  <div class="ja-section-title">Scores Table</div>
</div>
""",
        unsafe_allow_html=True
    )
    st.dataframe(
        df,
        use_container_width=False,
        hide_index=True,
        column_config={
            "Icon": st.column_config.TextColumn(width="small"),
            "Level": st.column_config.TextColumn(width="small"),
            "Metric": st.column_config.TextColumn(width="medium"),
            "Score": st.column_config.NumberColumn(format="%d", width="small"),
            "Bar": st.column_config.TextColumn(width="medium"),
            "Status": st.column_config.TextColumn(width="medium"),
            "Meaning": st.column_config.TextColumn(width="large"),
        }
    )


def render_suggestions_table(suggestions: List[str]):
    def classify(s: str) -> Tuple[str, str, str]:
        t = s.lower()
        level = "🟡"
        if "reduce" in t or "avoid" in t or "replace" in t:
            level = "🔴"
        if "looks good" in t:
            level = "🟢"

        if "clutter" in t or "background" in t or "whitespace" in t or "textures" in t:
            return "🧹", level, "Clutter / Simplicity"
        if "clarity" in t or "blur" in t or "sharpen" in t or "noise" in t:
            return "🔍", level, "Clarity"
        if "contrast" in t:
            return "⚫⚪", level, "Contrast"
        if "brightness" in t or "dark" in t or "washed" in t:
            return "☀️", level, "Brightness"
        if "focal" in t or "focus" in t:
            return "🎯", level, "Focus"
        if "text" in t or "readable" in t or "letter" in t or "nonsensical" in t:
            return "🔤", level, "Text"
        return "💡", level, "General"

    rows = []
    for i, s in enumerate(suggestions, start=1):
        icon, level, cat = classify(s)
        rows.append({
            "No of Suggestions": i,
            "Icon": icon,
            "Level": level,
            "Category": cat,
            "Suggestion": s
        })

    df = pd.DataFrame(rows)

    HEADER_PX = 38
    ROW_PX = 35
    n_rows = max(1, len(df))
    height = HEADER_PX + ROW_PX * n_rows

    HEADER_LABEL = "Suggestion"
    HEADER_CHAR_PX = 9
    HEADER_PAD_PX = 32
    suggestion_px = len(HEADER_LABEL) * HEADER_CHAR_PX + HEADER_PAD_PX

    st.markdown(
        """
<div class="ja-card ja-suggestions-header">
  <div class="ja-section-title">Suggestions Table</div>
</div>
""",
        unsafe_allow_html=True
    )
    st.dataframe(
        df,
        use_container_width=False,
        hide_index=True,
        height=height,
        column_config={
            "No of Suggestions": st.column_config.NumberColumn(width="small"),
            "Icon": st.column_config.TextColumn(width="small"),
            "Level": st.column_config.TextColumn(width="small"),
            "Category": st.column_config.TextColumn(width="medium"),
            "Suggestion": st.column_config.TextColumn(width=f"{suggestion_px}px"),
        }
    )


# =============================
# FIGURE SIZING / FONTS
# =============================
FIG_W = 8.4
FIG_H_BAR = 3.4
FIG_H_BOX = 3.6
FIG_H_GROUP = 3.6

FONT_TICK = 9
FONT_LABEL = 10
FONT_TITLE = 11
FONT_LEGEND = 9


# =============================
# STREAMLIT APP
# =============================
st.set_page_config(
    page_title="Cognitive Accessibility Analyzer for AI-generated Images",
    page_icon="🧠👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# SESSION STATE (ACCUMULATE + SELECTED IMAGE)
# =============================
if "history" not in st.session_state:
    st.session_state["history"] = []
if "image_store" not in st.session_state:
    st.session_state["image_store"] = {}
if "report_store" not in st.session_state:
    st.session_state["report_store"] = {}
if "selected_image_id" not in st.session_state:
    st.session_state["selected_image_id"] = None


# =============================
# CSS
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
footer { visibility: hidden; }

html, body, [class*="st-"], .stApp {
  font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
}

.ja-card {
  border: 1px solid rgba(128,128,128,0.20);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.02);
  margin-bottom: 8px;
}
.ja-badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(128,128,128,0.30);
  font-size: 15px;
  opacity: 0.95;
}
.ja-section-title {
  font-size: 25px;
  font-weight: 800;
  margin: 0 0 6px 0;
}
.ja-section-sub {
  font-size: 13px;
  opacity: 0.80;
  margin: 0;
}

[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(128,128,128,0.20);
}
div[data-testid="stDataFrame"] div[role="grid"] {
  overflow-x: auto !important;
  overflow-y: auto !important;
}
div[data-testid="stDataFrame"] div[role="grid"] [role="gridcell"] {
  white-space: nowrap !important;
}
div[data-testid="stDataFrame"] div[role="columnheader"],
div[data-testid="stDataFrame"] div[role="columnheader"] * {
  font-size: 18px !important;
  font-weight: 800 !important;
  color: inherit;
}

button[data-baseweb="tab"] > div {
  font-size: 25px !important;
  font-weight: 800 !important;
  padding: 10px 14px !important;
}

button[data-baseweb="tab"][aria-selected="true"] > div {
  font-size: 27px !important;
  font-weight: 900 !important;
}

div[data-baseweb="tab-list"] {
  margin-bottom: 10px;
}

.ja-dashboard-header,
.ja-compare-header,
.ja-details-header,
.ja-export-header {
  background: linear-gradient(90deg, #ffedd5, #fee2e2);
  border: 1px solid #fb7185;
  padding: 18px 20px;
}
.ja-dashboard-header .ja-section-title,
.ja-compare-header .ja-section-title,
.ja-details-header .ja-section-title,
.ja-export-header .ja-section-title { color: #0f172a; }

.ja-scores-header {
  background: linear-gradient(90deg, #ecfdf5, #d1fae5);
  border: 2px solid #bbf7d0;
  padding: 18px 20px;
}
.ja-scores-header .ja-section-title { color: #0f172a; }

.ja-suggestions-header {
  background: linear-gradient(90deg, #fff7ed, #fffbeb);
  border: 2px solid #fed7aa;
  padding: 18px 20px;
}
.ja-suggestions-header .ja-section-title { color: #0f172a; }

.ja-overall-score {
  border: 0.5px solid #fb7185;
  background: linear-gradient(135deg, #fff7ed 0%, #fee2e2 100%);
  border-radius: 18px;
}

.ja-card.ja-quick-interpretation {
  width: 100%;
  box-sizing: border-box;
  border: 0.5px solid #f59e0b;
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
  border-radius: 18px;
}

.ja-card.ja-output-card {
  width: 100%;
  box-sizing: border-box;
  border: 0.5px solid #fb7185;
  background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
  border-radius: 18px;
}

/* Hide uploaded file list */
div[data-testid="stFileUploader"] ul {
  display: none !important;
}

div[data-testid="stFileUploader"] {
  background: linear-gradient(90deg, #fff7ed, #fff1f2);
  border: 1px solid #fb7185;
  border-radius: 16px;
  padding: 3px;
  padding-top: 8px !important;
}

div[data-testid="stFileUploader"] section {
  background: transparent;
  padding-top: 0 !important;
  margin-top: 0 !important;
}

div[data-testid="stFileUploader"] label {
  font-weight: 700;
  margin-top: 0 !important;
  margin-bottom: 6px !important;
  padding: 0 !important;
  height: auto !important;
}

div[data-testid="stFileUploader"] button {
  border-radius: 10px !important;
}

div[data-testid="stFileUploader"] > div {
  margin-top: 0 !important;
}
</style>
""",
    unsafe_allow_html=True
)

# ---- Header / Hero ----
st.markdown('<div class="ja-card">', unsafe_allow_html=True)
left, right = st.columns([0.72, 0.28], gap="large")
with left:
    st.markdown("## 🧠👁️ Cognitive Accessibility Analyzer for AI-generated Images")
    st.markdown(
        "Assess AI-generated images for cognitive accessibility proxies (clarity, clutter, contrast, brightness, focus, and text quality) "
        "with explainable overlays and structured suggestions."
    )
with right:
    opencv_badge = "✅ OpenCV" if OPENCV_OK else "❌ OpenCV"
    ocr_badge = "✅ OCR" if PYTESS_OK else "❌ OCR"
    st.markdown(
        f"""
<div class="ja-card">
  <div class="ja-section-title">System</div>
  <div class="ja-section-sub">Runtime capabilities</div>
  <span class="ja-badge">{opencv_badge}</span>
  <span class="ja-badge" style="margin-left:6px;">{ocr_badge}</span>
</div>
""",
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### Controls")
    overlay_strength = st.slider("Overlay strength", 0.0, 0.85, 0.55, 0.05)
    st.caption("Higher values make the overlay more visible.")

    st.markdown("### Upload")
    upload_mode = st.radio(
        "Upload mode",
        ["Batch (upload multiple at once)", "Single (upload one-by-one and accumulate)"],
        index=0
    )
    accept_multi = (upload_mode == "Batch (upload multiple at once)")

    st.markdown(
    "<div style='font-weight:700; margin-bottom:6px;'>Choose image(s) (PNG/JPG)</div>",
    unsafe_allow_html=True
    )
    uploaded = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=accept_multi
    )

    uploaded_files = uploaded if isinstance(uploaded, list) else ([uploaded] if uploaded else [])

    if uploaded_files:
        f = uploaded_files[-1]
        size_kb = f.size / 1024

        st.markdown(
        f"""
    <div style="
        margin-top: 6px;
        padding: 10px 12px;
        border-radius: 12px;
        background: linear-gradient(90deg, #fff7ed, #ffe4e6);
        border: 2px solid #fb7185;
        font-size: 13px;">
        📄 <strong>{f.name}</strong><br>
        <span style="opacity:0.8;">{size_kb:.1f} KB</span><br>
        <span style="opacity:0.8;">Selected: {len(uploaded_files)} file(s)</span>
</div>
""",
            unsafe_allow_html=True
        )

    st.caption("Results accumulate during your session. Use Compare/Export tabs for comparison and downloads.")

    st.markdown("---")
    st.markdown("### About overlays")
    st.caption("• **Complexity overlay** highlights texture/edge-dense areas (visual complexity proxy).")
    st.caption("• **Saliency heatmap** shows attention-attracting regions (saliency proxy).")

# ---- Normalize uploads to list ----
files = []
if uploaded is None:
    files = []
elif isinstance(uploaded, list):
    files = uploaded
else:
    files = [uploaded]

# Empty state
if len(files) == 0 and len(st.session_state["history"]) == 0:
    st.markdown(
        """
<div class="ja-card">
  <div class="ja-section-title">Get started</div>
  <div class="ja-section-sub">Upload an image from the left sidebar to view results and comparisons.</div>
</div>
""",
        unsafe_allow_html=True
    )
    st.stop()

# ---- Analyze current upload (if any) ----
latest_results = []
existing_ids = {r.get("image_id") for r in st.session_state["history"]}

for up in files:
    img_bytes = up.read()
    img_orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result = analyze_image(img_orig, overlay_strength=overlay_strength)

    img_id = hashlib.md5(img_bytes).hexdigest()
    filename = getattr(up, "name", "uploaded_image")

    if img_id not in st.session_state["image_store"]:
        o = _safe_resize_for_store(img_orig, max_side=900)
        ov = _safe_resize_for_store(result.overlay_image, max_side=900)
        hm = _safe_resize_for_store(result.saliency_heatmap, max_side=900)
        st.session_state["image_store"][img_id] = {
            "orig": _pil_to_png_bytes(o),
            "overlay": _pil_to_png_bytes(ov),
            "heat": _pil_to_png_bytes(hm),
        }

    safe_report = _json_safe({
        "overall_score": result.overall_score,
        "subscores": result.subscores,
        "suggestions": result.suggestions,
        "metrics": result.debug,
        "dependencies": {"opencv": OPENCV_OK, "ocr": PYTESS_OK},
        "filename": filename,
        "image_id": img_id,
        "overlay_strength": float(overlay_strength),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    })
    st.session_state["report_store"][img_id] = safe_report

    row = {
        "timestamp": safe_report["timestamp"],
        "filename": filename,
        "image_id": img_id,
        "overlay_strength": float(overlay_strength),
        "overall": int(result.overall_score),
        **{f"sub_{k.lower()}": int(v) for k, v in result.subscores.items()},
        **{f"m_{k}": float(v) for k, v in result.debug.items()},
    }

    if img_id not in existing_ids:
        st.session_state["history"].append(row)
        existing_ids.add(img_id)

    latest_results.append((img_orig, result, row))

# ---- Ensure a selected image exists ----
hist_df = pd.DataFrame(st.session_state["history"])
if st.session_state["selected_image_id"] is None:
    if latest_results:
        st.session_state["selected_image_id"] = latest_results[0][2]["image_id"]
    elif not hist_df.empty:
        best = hist_df.sort_values("overall", ascending=False).iloc[0].to_dict()
        st.session_state["selected_image_id"] = best["image_id"]

# ---- Tabs ----
tab_dash, tab_compare, tab_details, tab_export = st.tabs(
    ["📊 Dashboard", "📈 Compare", "🧾 Details", "⬇️ Export"]
)

# =============================
# DASHBOARD
# =============================
with tab_dash:
    st.markdown(
        """
<div class="ja-card ja-dashboard-header">
  <div class="ja-section-title">Dashboard</div>
  <div class="ja-section-sub"> Visual overview of the selected image, combining explainable overlays, accessibility scores, and actionable suggestions.</div>
</div>
""",
        unsafe_allow_html=True
    )

    if latest_results:
        names = [r[2]["filename"] for r in latest_results]
        ids = [r[2]["image_id"] for r in latest_results]
        try:
            default_idx = ids.index(st.session_state["selected_image_id"])
        except ValueError:
            default_idx = 0

        pick_name = st.selectbox("**Preview one image (from this upload)", names, index=default_idx)
        pick_idx = names.index(pick_name)
        st.session_state["selected_image_id"] = ids[pick_idx]
    else:
        if hist_df.empty:
            st.info("Upload at least one image to start.")
            st.stop()

        names_all = hist_df["filename"].astype(str).tolist()
        ids_all = hist_df["image_id"].astype(str).tolist()
        try:
            default_idx = ids_all.index(st.session_state["selected_image_id"])
        except ValueError:
            default_idx = 0

        pick_name = st.selectbox("Preview one image (from session history)", names_all, index=default_idx)
        pick_idx = names_all.index(pick_name)
        st.session_state["selected_image_id"] = ids_all[pick_idx]

    sel_id = st.session_state["selected_image_id"]
    rep = st.session_state["report_store"].get(sel_id)
    store = st.session_state["image_store"].get(sel_id)

    if rep is None or store is None:
        st.warning("Selected image data not found. Please upload again.")
        st.stop()

    img_preview = _png_bytes_to_pil(store["orig"])
    overlay_img = _png_bytes_to_pil(store["overlay"])
    heat_img = _png_bytes_to_pil(store["heat"])
    subs = rep["subscores"]
    sugg = rep["suggestions"]
    overall = int(rep["overall_score"])

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("""
        <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">Original Image</div>
        """, unsafe_allow_html=True)
        st_image(img_preview)
    with c2:
        st.markdown("""
        <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">Complexity Overlay</div>
        """, unsafe_allow_html=True)
        st_image(overlay_img)
    with c3:
        st.markdown("""
        <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">Saliency Heatmap</div>
        """, unsafe_allow_html=True)
        st_image(heat_img)

    st.markdown("")

    a, b, c = st.columns([0.34, 0.33, 0.33], gap="large")
    with a:
        score_color = overall_score_color(overall)
        st.markdown(
            f"""
<div class="ja-card ja-overall-score">
  <div class="ja-section-title">Overall Score</div>
  <div style="font-size:30px;font-weight:800;line-height:1.0;color:{score_color};">
    {overall}%
  </div>
  <div class="ja-section-sub">out of 100</div>
</div>
""",
            unsafe_allow_html=True
        )
        st.progress(overall / 100.0)
    with b:
        st.markdown(
            """
<div class="ja-card ja-quick-interpretation">
  <div class="ja-section-title">Quick Interpretation</div>
  <div class="ja-section-sub">
    🟢 Green (≥ 80): Good.<br>
    🟡 Yellow (60–79): Needs Attention.<br>
    🔴 Red (&lt; 60): High Risk.
  </div>
</div>
""",
            unsafe_allow_html=True
        )
    with c:
        st.markdown(
            """
<div class="ja-card ja-output-card">
  <div class="ja-section-title">Output</div>
  <div class="ja-section-sub">
    Overall Score Table, Suggestions Table, Per-image Report, Session-wide Comparisons, CSV/JSON Files.
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("")
    render_score_table(subs)
    st.markdown("")
    render_suggestions_table(sugg)

# =============================
# COMPARE
# =============================
with tab_compare:
    st.markdown(
        """
<div class="ja-card ja-compare-header">
  <div class="ja-section-title">Comparative Results of Evaluated AI-generated Images</div>
  <div class="ja-section-sub"> Comparative analysis across evaluated images, highlighting relative performance, metric distributions, and accessibility patterns.</div>
</div>
""",
        unsafe_allow_html=True
    )

    hist = pd.DataFrame(st.session_state["history"])
    if hist.empty:
        st.info("No accumulated results yet. Upload images first.")
        st.stop()

    comp = hist.copy().sort_values("overall", ascending=False)

    names_all = comp["filename"].astype(str).tolist()
    ids_all = comp["image_id"].astype(str).tolist()

    try:
        default_idx = ids_all.index(st.session_state["selected_image_id"])
    except Exception:
        default_idx = 0

    pick_name = st.selectbox(
        "**Preview one image (from this upload)**",
        options=names_all,
        index=default_idx,
        key="compare_preview_select"
    )
    st.session_state["selected_image_id"] = ids_all[names_all.index(pick_name)]

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    })

    st.markdown("### Overall score")
    labels = comp["filename"].astype(str).tolist()
    values = comp["overall"].astype(float).tolist()

    FIG_W_BAR = 4
    FIG_H_BAR_LOCAL = 3

    fig = plt.figure(figsize=(FIG_W_BAR, FIG_H_BAR_LOCAL))
    plt.bar(range(len(values)), values)
    plt.title("Overall score across evaluated images", fontsize=7)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=5)
    plt.yticks(fontsize=5)
    plt.ylabel("Overall", fontsize=7)
    plt.ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Metric heatmap (higher is better)")
    metric_cols = [c for c in comp.columns if c.startswith("sub_") and c not in ["sub_overall"]]
    if metric_cols:
        metric_view = comp.set_index("filename")[metric_cols].copy()
        styled = metric_view.style.format("{:.0f}").background_gradient(axis=None)
        st.dataframe(styled, use_container_width=True, height=300)
    else:
        st.info("No sub-metric columns found to build the heatmap.")

    st.markdown("### Metric heatmap (visualization)")
    metric_cols = [c for c in comp.columns if c.startswith("sub_") and c not in ["sub_overall"]]
    if metric_cols:
        metric_view = comp.set_index("filename")[metric_cols].copy()
        x_labels = [c.replace("sub_", "").title() for c in metric_view.columns]
        y_labels = metric_view.index.astype(str).tolist()
        data = metric_view.values.astype(float)

        heat_w = 6.6
        heat_h = min(6.0, max(3.4, 0.35 * len(y_labels)))

        fig = plt.figure(figsize=(heat_w, heat_h))
        ax = plt.gca()
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=100)

        ax.set_title("Sub-metric scores heatmap (higher is better)", fontsize=FONT_TITLE)
        ax.set_xlabel("Metrics", fontsize=FONT_LABEL)
        ax.set_ylabel("Images", fontsize=FONT_LABEL)

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=FONT_TICK)

        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=FONT_TICK)

        if len(y_labels) <= 12 and len(x_labels) <= 10:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{int(round(data[i, j]))}", ha="center", va="center", fontsize=8)

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.ax.tick_params(labelsize=FONT_TICK)
        cbar.set_label("Score", fontsize=FONT_LABEL)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No sub-metric columns found to build the heatmap.")

    st.markdown("### Box plot (score distributions across images)")
    box_cols = ["overall"] + [c for c in metric_cols if c in comp.columns]
    box_df = comp[box_cols].copy()

    nice_names = []
    for c in box_df.columns:
        if c == "overall":
            nice_names.append("Overall")
        else:
            nice_names.append(c.replace("sub_", "").title())
    box_df.columns = nice_names

    fig = plt.figure(figsize=(FIG_W, FIG_H_BOX))
    plt.boxplot([box_df[c].dropna().values for c in box_df.columns], labels=box_df.columns, showfliers=True)
    plt.title("Score distributions (box plot) across evaluated images", fontsize=FONT_TITLE)
    plt.xticks(rotation=30, ha="right", fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.ylabel("Score", fontsize=FONT_LABEL)
    plt.ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Compare selected images (grouped metrics)")
    picks = st.multiselect(
        "Select images",
        options=comp["filename"].tolist(),
        default=comp["filename"].tolist()[:min(4, len(comp))]
    )

    if picks:
        sel = comp[comp["filename"].isin(picks)].copy()
        metric_cols2 = [c for c in sel.columns if c.startswith("sub_") and c not in ["sub_overall"]]
        if metric_cols2:
            sel_plot = sel.set_index("filename")[metric_cols2]

            fig = plt.figure(figsize=(FIG_W, FIG_H_GROUP))
            x = np.arange(len(sel_plot.index))
            width = max(0.08, 0.8 / max(1, len(metric_cols2)))

            for i, m in enumerate(metric_cols2):
                plt.bar(x + i * width, sel_plot[m].values, width=width, label=m.replace("sub_", "").title())

            plt.title("Metric comparison for selected images", fontsize=FONT_TITLE)
            plt.xticks(x + (len(metric_cols2) * width) / 2, sel_plot.index, rotation=30, ha="right", fontsize=FONT_TICK)
            plt.yticks(fontsize=FONT_TICK)
            plt.ylim(0, 100)
            plt.ylabel("Score", fontsize=FONT_LABEL)
            plt.legend(ncol=3, fontsize=FONT_LEGEND)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No sub-metric columns found for grouped comparison.")

    st.markdown("---")

    st.markdown("### Visual inspection panels (overlays only)")
    st.caption("Expand an image to see its Original / Overlay / Heatmap. Score tables remain on the Dashboard.")

    comp_iter = comp.copy()
    if "timestamp" in comp_iter.columns:
        comp_iter = comp_iter.sort_values("timestamp", ascending=False)

    for _, row in comp_iter.iterrows():
        img_id = row.get("image_id")
        filename = row.get("filename", "image")
        overall = int(row.get("overall", 0))

        title = f"{filename} — Overall: {overall}%"
        with st.expander(title, expanded=False):
            store = st.session_state["image_store"].get(img_id)
            if store is None:
                st.warning("Stored visuals not found for this image (upload again to regenerate previews).")
                continue

            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                st.markdown("**Original**")
                st_image(_png_bytes_to_pil(store["orig"]))
            with c2:
                st.markdown("**Complexity overlay**")
                st_image(_png_bytes_to_pil(store["overlay"]))
            with c3:
                st.markdown("**Saliency heatmap**")
                st_image(_png_bytes_to_pil(store["heat"]))

# =============================
# DETAILS
# =============================
with tab_details:
    st.markdown(
        """
<div class="ja-card ja-details-header">
  <div class="ja-section-title">Details</div>
  <div class="ja-section-sub">
    Raw metrics used internally to compute the scores for the selected/previewed image.
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    sel_id = st.session_state["selected_image_id"]
    rep = st.session_state["report_store"].get(sel_id)
    if rep is None:
        st.info("No details available yet. Upload an image first.")
        st.stop()

    st.json(_json_safe(rep.get("metrics", {})))

    with st.expander("How to interpret overlays"):
        st.write("**Complexity overlay** highlights texture/edge-dense areas (visual complexity proxy).")
        st.write("**Saliency heatmap** highlights attention-attracting regions (saliency proxy).")
        st.write("Neither overlay alone equals “inaccessible”; they indicate *risk areas* depending on context.")

# =============================
# EXPORT
# =============================
with tab_export:
    st.markdown(
        """
<div class="ja-card ja-export-header">
  <div class="ja-section-title">Export</div>
  <div class="ja-section-sub">
    Download the selected image report, and export all accumulated results as CSV/JSON.
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    sel_id = st.session_state["selected_image_id"]
    rep = st.session_state["report_store"].get(sel_id)

    st.markdown("**Selected image report (preview):**")
    if rep is None:
        st.info("No preview report available yet.")
    else:
        st.download_button(
            "Download JSON report (selected image)",
            data=json.dumps(_json_safe(rep), indent=2).encode("utf-8"),
            file_name="report_selected.json",
            mime="application/json"
        )

    st.markdown("---")

    hist = pd.DataFrame(st.session_state["history"])
    if hist.empty:
        st.info("No accumulated results yet.")
    else:
        st.markdown("### Accumulated results (session)")
        st.dataframe(hist.drop(columns=["image_id"], errors="ignore"), use_container_width=True)

        st.download_button(
            "Download CSV (all analyses)",
            data=hist.to_csv(index=False).encode("utf-8"),
            file_name="all_scores.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download JSON (all analyses)",
            data=json.dumps(_json_safe(st.session_state["history"]), indent=2).encode("utf-8"),
            file_name="all_scores.json",
            mime="application/json",
        )

        colA, colB = st.columns([0.6, 0.4])
        with colA:
            st.caption("Clearing removes session results + stored previews. (Uploads are not deleted from your device.)")
        with colB:
            if st.button("Clear history & previews"):
                st.session_state["history"] = []
                st.session_state["image_store"] = {}
                st.session_state["report_store"] = {}
                st.session_state["selected_image_id"] = None
                st.rerun()
