# =========================================================
# Nusantara Transit Intelligence
# Indonesian Tourism Analytics Dashboard
# =========================================================

from pathlib import Path
import base64
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

# =========================================================
# PAGE CONFIG (must be first Streamlit call)
# =========================================================
st.set_page_config(
    page_title="Nusantara Transit Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS & ASSETS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "tourism_indonesia_2016-2026_clean.csv"
MODEL_PATH = BASE_DIR / "model" / "kmeans_clustering_final.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler_clustering.pkl"
SARIMA_PATH = BASE_DIR / "model" / "tourism_sarima_model.pkl"
BG_IMAGE_PATH = BASE_DIR / "assets" / "aviation_art_bg.jpg"


@st.cache_data
def get_aviation_bg_base64() -> str:
    """Load and base64-encode background aviation art for instant CSS injection"""
    if BG_IMAGE_PATH.exists():
        with open(BG_IMAGE_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


BG_BASE64 = get_aviation_bg_base64()

# =========================================================
# =========================================================
# GLOBAL CSS - Aviation Glassmorphism & Atmospheric Live Background
# =========================================================
CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Baloo+2:wght@500;600;700;800&display=swap');

:root {
    --bg: #0d2217;
    --bg-soft: #f5f8ea;
    --card: rgba(255, 255, 255, 0.88);
    --card-2: rgba(230, 245, 225, 0.82);
    --text: #133426;
    --muted: #2d4d33;
    --green: #1f5137;
    --green-dark: #0f2e1a;
    --emerald-accent: #00a86b;
    --mint: #bfe0bd;
    --yellow: #f1cc6d;
    --orange: #f77f00;
    --blue: #0077b6;
    --sea: #00a86b;
    --shadow-card: 0 16px 36px rgba(9, 24, 18, 0.18), inset 0 1px 2px rgba(255, 255, 255, 0.85);
    --radius: 26px;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
    color: var(--text);
}

/* =========================================================
   ANIMATED AVIATION ART BACKGROUND
   ========================================================= */
.stApp {
    position: relative;
    background-color: transparent !important;
}

/* Smooth, Cinematic Moving Background */
.stApp::before {
    content: '';
    position: fixed;
    top: -5%;
    left: -5%;
    width: 110%;
    height: 110%;
    background-image: url('data:image/jpeg;base64,__BG_BASE64__');
    background-size: cover;
    background-position: center 40%;
    background-repeat: no-repeat;
    z-index: -2;
    animation: flightDrift 45s ease-in-out infinite alternate;
    pointer-events: none;
    transform-origin: center center;
    will-change: transform;
}

@keyframes flightDrift {
    0% {
        transform: scale(1.0) translate(0px, 0px);
    }
    35% {
        transform: scale(1.06) translate(-16px, -10px);
    }
    70% {
        transform: scale(1.03) translate(12px, 8px);
    }
    100% {
        transform: scale(1.08) translate(-8px, 14px);
    }
}

/* Atmospheric Horizon Tint Overlay */
.stApp::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: radial-gradient(circle at 75% 22%, rgba(15, 52, 32, 0.16) 0%, rgba(9, 24, 18, 0.55) 100%);
    backdrop-filter: blur(1.5px);
    -webkit-backdrop-filter: blur(1.5px);
    z-index: -1;
    pointer-events: none;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Streamlit Container Transparency & Clear Z-Index Stacking */
div[data-testid="stAppViewContainer"] {
    background: transparent !important;
    position: relative !important;
    z-index: 1 !important;
}

div[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
    position: relative !important;
    z-index: 2 !important;
}

.block-container {
    position: relative !important;
    z-index: 5 !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* Glassmorphic Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(238, 246, 235, 0.90) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.60) !important;
    box-shadow: 6px 0 24px rgba(9, 24, 18, 0.15) !important;
    position: relative !important;
    z-index: 9999 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Sidebar Brand Card */
.sidebar-brand {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 248, 235, 0.90) 100%);
    backdrop-filter: blur(12px);
    padding: 22px 18px;
    border-radius: 24px;
    box-shadow: 0 10px 24px rgba(9, 24, 18, 0.12), inset 0 1px 1px rgba(255, 255, 255, 0.9);
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.sidebar-brand::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #00a86b 0%, #0077b6 50%, #f77f00 100%);
}
.sidebar-brand .brand-badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    background: #cde8ca;
    color: #0f3320;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.sidebar-brand .brand-title {
    font-family: "Baloo 2", sans-serif;
    font-size: 1.42rem;
    font-weight: 800;
    color: var(--green-dark);
    line-height: 1.1;
    margin: 0;
}
.sidebar-brand .brand-sub {
    font-size: 0.74rem;
    color: #2d4d33;
    font-weight: 600;
    margin-top: 5px;
    opacity: 0.9;
}

/* Sidebar Navigation Header */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div:first-child:not([role="radiogroup"]) {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 9px !important;
    padding: 2px 0 !important;
}

/* Hide raw radio input and bullet circles */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child:not([data-testid="stMarkdownContainer"]) {
    display: none !important;
}

/* Glassmorphic Nav Pill Buttons */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.92) !important;
    border-radius: 16px !important;
    padding: 12px 18px !important;
    margin: 0 !important;
    border: 1.5px solid rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 4px 12px rgba(9, 24, 18, 0.08) !important;
    cursor: pointer !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label span,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label * {
    color: #133426 !important;
    -webkit-text-fill-color: #133426 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    opacity: 1 !important;
    margin: 0 !important;
}

/* Hover State */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: #ffffff !important;
    transform: translateX(4px) translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(9, 24, 18, 0.14) !important;
    border-color: rgba(0, 168, 107, 0.4) !important;
}

/* ACTIVE / SELECTED STATE: Forest Deep Green dengan aksen glow */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #0b5e39 0%, #073822 100%) !important;
    border: 1.5px solid #00a86b !important;
    box-shadow: 0 6px 20px rgba(11, 94, 57, 0.40), inset 0 1px 2px rgba(255, 255, 255, 0.3) !important;
    transform: translateX(6px) !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) *,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 800 !important;
}

/* Sidebar Specs Card */
.sidebar-meta-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.92) 0%, rgba(240, 248, 235, 0.88) 100%);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 16px;
    box-shadow: 0 8px 20px rgba(9, 24, 18, 0.12), inset 0 1px 1px rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.8);
    margin-top: 18px;
}
.sidebar-meta-card .meta-status {
    font-size: 0.74rem;
    font-weight: 700;
    color: #1f5137;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
}
.sidebar-meta-card .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #27ae60;
    box-shadow: 0 0 8px rgba(39, 174, 96, 0.8);
    display: inline-block;
}
.sidebar-meta-card .meta-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 6px;
}
.sidebar-meta-card .meta-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.74rem;
    padding: 3px 0;
    border-bottom: 1px dashed rgba(31, 81, 55, 0.15);
}
.sidebar-meta-card .meta-label { color: #55755e; font-weight: 600; }
.sidebar-meta-card .meta-val { color: #133426; font-weight: 700; }
.sidebar-meta-card .meta-foot {
    font-size: 0.68rem;
    color: #43664d;
    text-align: center;
    margin-top: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Frosted Glassmorphism Primary Cards */
.clay-card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: var(--radius);
    box-shadow: 0 16px 36px rgba(9, 24, 18, 0.20), inset 0 1px 2px rgba(255, 255, 255, 0.95);
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.75);
    color: var(--green);
    margin-bottom: 16px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.clay-card:hover {
    box-shadow: 0 20px 42px rgba(9, 24, 18, 0.26), inset 0 1px 2px rgba(255, 255, 255, 1);
}
.clay-card, .clay-card p, .clay-card div, .clay-card span,
.clay-card h1, .clay-card h2, .clay-card h3,
.clay-card h4, .clay-card h5, .clay-card h6, .clay-card label {
    color: var(--green);
}
.clay-card .cluster-0 { color: #2e1554 !important; }
.clay-card .cluster-1 { color: #0f3320 !important; }
.cluster-members { color: #1f5137 !important; font-weight: 600; line-height: 1.8; }

.stMarkdown, .stMarkdown p, .stMarkdown li,
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    color: #1f5137 !important;
}

/* Glass Selectbox Control */
div[data-baseweb="select"] {
    background: transparent !important;
    border-radius: 16px !important;
}
div[data-baseweb="select"] > div {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 248, 235, 0.92) 100%) !important;
    backdrop-filter: blur(12px) !important;
    border: 1.5px solid rgba(31, 81, 55, 0.25) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 14px rgba(78, 93, 73, 0.10), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
    overflow: hidden !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(31, 81, 55, 0.5) !important;
    box-shadow: 0 6px 18px rgba(78, 93, 73, 0.15) !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #00a86b !important;
    box-shadow: 0 0 0 3px rgba(0, 168, 107, 0.22) !important;
}
div[data-baseweb="select"] > div * {
    background: transparent !important;
    background-color: transparent !important;
    color: #133426 !important;
    -webkit-text-fill-color: #133426 !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] svg {
    fill: #133426 !important;
    color: #133426 !important;
}

/* Light Glass Dropdown Popover & Options Menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [role="listbox"],
ul[data-baseweb="menu"],
div[data-baseweb="menu"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-radius: 16px !important;
    border: 1.5px solid rgba(31, 81, 55, 0.2) !important;
    box-shadow: 0 14px 32px rgba(9, 24, 18, 0.22) !important;
    padding: 6px !important;
    overflow: hidden !important;
}

/* Option Items inside Dropdown */
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] div[role="option"],
ul[data-baseweb="menu"] li,
div[data-baseweb="menu"] div[role="option"] {
    background-color: #ffffff !important;
    color: #133426 !important;
    -webkit-text-fill-color: #133426 !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    margin: 2px 0 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
div[data-baseweb="popover"] li *,
div[data-baseweb="popover"] div[role="option"] *,
ul[data-baseweb="menu"] li *,
div[data-baseweb="menu"] div[role="option"] * {
    color: #133426 !important;
    -webkit-text-fill-color: #133426 !important;
    font-weight: 600 !important;
}

/* Hover & Selected Option State */
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] div[role="option"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"],
div[data-baseweb="popover"] div[role="option"][aria-selected="true"],
ul[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] div[role="option"]:hover,
ul[data-baseweb="menu"] li[aria-selected="true"],
div[data-baseweb="menu"] div[role="option"][aria-selected="true"] {
    background-color: #d8edd2 !important;
    background: linear-gradient(135deg, #d8edd2 0%, #c4e4bc 100%) !important;
    color: #0c2b18 !important;
    -webkit-text-fill-color: #0c2b18 !important;
    font-weight: 700 !important;
}
div[data-baseweb="popover"] li:hover *,
div[data-baseweb="popover"] div[role="option"]:hover *,
div[data-baseweb="popover"] li[aria-selected="true"] *,
div[data-baseweb="popover"] div[role="option"][aria-selected="true"] * {
    color: #0c2b18 !important;
    -webkit-text-fill-color: #0c2b18 !important;
    font-weight: 700 !important;
}

/* Fix text input fields */
input, textarea {
    color: #133426 !important;
    -webkit-text-fill-color: #133426 !important;
    caret-color: #133426 !important;
    background: linear-gradient(145deg, #ffffff, #edf4e2) !important;
    border-radius: 14px !important;
}
input::placeholder, textarea::placeholder {
    color: #63856d !important;
    -webkit-text-fill-color: #63856d !important;
    opacity: 0.85;
}

.stTextInput label, .stSelectbox label, .stMultiSelect label {
    color: #1f5137 !important;
}
[data-testid="stDataFrame"] { color: #1f5137 !important; }

/* Aero Glassmorphism Hero Banner */
.hero {
    position: relative; overflow: hidden;
    background:
        radial-gradient(circle at 88% 28%, rgba(141,195,213,.35), transparent 28%),
        radial-gradient(circle at 17% 85%, rgba(241,204,109,.25), transparent 28%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(240, 248, 235, 0.88));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    min-height: 310px;
    padding: 36px;
    border-radius: var(--radius);
    box-shadow: 0 18px 40px rgba(9, 24, 18, 0.22), inset 0 1px 2px rgba(255, 255, 255, 0.95);
    border: 1.5px solid rgba(255, 255, 255, 0.85);
}
.hero .eyebrow {
    display: inline-block; padding: 7px 12px;
    border-radius: 999px; background: #c3dfb3; color: #0f2e1a;
    font-size: .76rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: 12px;
}
.hero h1 {
    font-family: "Baloo 2", sans-serif;
    font-size: clamp(2.4rem, 5vw, 4.8rem);
    line-height: .9; margin: 0;
    color: var(--green-dark); max-width: 60%;
}
.hero p {
    color: #2b3e2e; max-width: 55%;
    font-size: 1rem; font-weight: 500;
    line-height: 1.65; margin-top: 14px;
}
.section-title {
    font-family: "Baloo 2", sans-serif;
    font-size: 1.8rem; font-weight: 800;
    color: var(--green-dark); margin: 22px 0 10px;
}
.section-subtitle { color: #2d4d33; font-weight: 600; margin-bottom: 16px; }

/* KPI Cards with Frosted Glass */
.kpi {
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: rgba(255, 255, 255, 0.90) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.80) !important;
    box-shadow: 0 12px 28px rgba(9, 24, 18, 0.16), inset 0 1px 2px rgba(255, 255, 255, 0.9) !important;
}
.kpi .label { font-size: .82rem; color: #2d4d33; font-weight: 700; }
.kpi .value {
    font-family: "Baloo 2", sans-serif; font-size: 2rem;
    font-weight: 800; color: var(--green-dark); line-height: 1;
}
.kpi .detail { font-size: .75rem; color: #2d4d33; font-weight: 600; }

.transport-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 14px; border-radius: 999px;
    margin-right: 8px; margin-bottom: 8px;
    background: rgba(237, 243, 223, 0.90);
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 10px rgba(91,100,85,.16), inset 0 1px 1px rgba(255,255,255,.9);
    border: 1px solid rgba(255, 255, 255, 0.7);
    font-size: .84rem; font-weight: 700; color: #1f5137;
}

.note-box {
    background: rgba(255, 247, 220, 0.92);
    backdrop-filter: blur(12px);
    border-radius: 20px; padding: 16px 18px;
    box-shadow: 0 10px 24px rgba(116,99,45,.14);
    border: 1px solid rgba(255, 255, 255, 0.8);
    color: #3d3010; font-size: .88rem; font-weight: 500;
}

.cluster-chip {
    display: inline-block; padding: 8px 13px;
    border-radius: 999px; font-weight: 800; font-size: .8rem;
}
.cluster-0 { background: #c8b4e8; color: #2e1554; }
.cluster-1 { background: #a8d9b5; color: #0f3320; }

.insight-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.92), rgba(235, 246, 232, 0.88));
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 20px; padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(9, 24, 18, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-left: 5px solid #1f5137; color: #1f5137; margin-bottom: 12px;
}
.insight-card .insight-icon { font-size: 1.5rem; margin-bottom: 6px; }
.insight-card .insight-text { font-size: .9rem; font-weight: 600; line-height: 1.5; color: #1f5137; }

.overview-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.92), rgba(240, 247, 235, 0.88));
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 22px; padding: 18px 20px;
    box-shadow: 0 12px 26px rgba(9, 24, 18, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.8);
    text-align: center; color: #1f5137;
}
.overview-card .ov-label {
    font-size: .78rem; font-weight: 700; color: #2d4d33;
    text-transform: uppercase; letter-spacing: .06em;
}
.overview-card .ov-value {
    font-family: "Baloo 2", sans-serif; font-size: 1.1rem;
    font-weight: 800; color: var(--green-dark); margin-top: 4px; line-height: 1.2;
}

.method-step {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(242, 249, 238, 0.88));
    backdrop-filter: blur(12px);
    border-radius: 16px; padding: 14px 16px;
    box-shadow: 0 8px 20px rgba(9, 24, 18, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.8);
    text-align: center; color: #1f5137; font-size: .82rem; font-weight: 700;
}

.best-badge {
    display: inline-block; background: #a8d9b5; color: #0f3320;
    padding: 3px 10px; border-radius: 999px;
    font-size: .72rem; font-weight: 800; letter-spacing: .04em;
}

/* Policy Simulator & Anomaly Detection Glass Styles */
.policy-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(240, 248, 236, 0.88));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 12px 28px rgba(9, 24, 18, 0.16);
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    margin-bottom: 14px;
}
.policy-card h4 {
    margin: 0 0 8px 0;
    font-family: 'Baloo 2', sans-serif;
    color: var(--green-dark);
}
.tag-badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
}
.tag-positive { background: #d4edda; color: #155724; }
.tag-negative { background: #f8d7da; color: #721c24; }
.tag-warning { background: #fff3cd; color: #856404; }
.tag-info { background: #d1ecf1; color: #0c5460; }

.dossier-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(244, 249, 240, 0.88));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(9, 24, 18, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-left: 5px solid #1f5137;
    margin-bottom: 12px;
}
.dossier-card.crisis { border-left-color: #d32f2f; }
.dossier-card.rebound { border-left-color: #27ae60; }
.dossier-card.event { border-left-color: #f39c12; }
.dossier-card h4 {
    margin: 0 0 6px 0;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--green-dark);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.dossier-card p {
    font-size: 0.88rem;
    color: #2d4d33;
    line-height: 1.55;
    margin: 0;
}

/* Page Context Banner for Educational & Academic Value */
.page-context-banner {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(240, 248, 238, 0.90));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 18px 22px;
    box-shadow: 0 12px 30px rgba(9, 24, 18, 0.16);
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    margin: 16px 0 20px 0;
    position: relative;
    overflow: hidden;
}
.page-context-banner::before {
    content: "";
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 5px;
    background: linear-gradient(180deg, #1f5137 0%, #00a86b 50%, #f1cc6d 100%);
}
.context-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
}
.context-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.context-label {
    font-size: 0.72rem;
    font-weight: 800;
    color: #133426;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 5px;
}
.context-text {
    font-size: 0.84rem;
    color: #2d4d33;
    line-height: 1.48;
    font-weight: 500;
}

footer { display: none; }
</style>
"""

st.markdown(CSS_TEMPLATE.replace("__BG_BASE64__", BG_BASE64), unsafe_allow_html=True)

# =========================================================
# SAFE TRANSPORT SVG ART - No HTML injection
# =========================================================
def get_transport_svg():
    """Return transport illustration as clean SVG string for safe HTML injection"""
    return '''<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" 
        style="width:100%;height:auto;display:block;max-width:480px;">
        <defs>
        <filter id="tshadow" x="-30%" y="-30%" width="160%" height="160%">
        <feDropShadow dx="0" dy="12" stdDeviation="9" flood-color="#7c8d78" flood-opacity=".24"/>
        </filter>
        </defs>
        <g fill="#ffffff" opacity=".72">
        <circle cx="530" cy="55" r="28"/>
        <circle cx="560" cy="55" r="36"/>
        <circle cx="595" cy="55" r="24"/>
        <rect x="520" y="54" width="95" height="27" rx="13"/>
        <circle cx="160" cy="40" r="20"/>
        <circle cx="185" cy="38" r="28"/>
        <circle cx="215" cy="40" r="18"/>
        <rect x="148" y="38" width="77" height="22" rx="11"/>
        </g>
        <g transform="translate(380 40) rotate(-7)" filter="url(#tshadow)">
        <path d="M0 32 L125 0 L160 16 L118 29 L157 50 L143 61 L103 42 L68 53 L54 98 L40 100 L42 56 L6 50 Z" fill="#e8995d"/>
        <path d="M46 37 L90 24 L92 36 L54 50 Z" fill="#f1cc6d"/>
        </g>
        <path d="M0 287 C190 242, 330 256, 700 284" fill="none" stroke="#a9ae9d" stroke-width="18" stroke-linecap="round"/>
        <path d="M0 287 C190 242, 330 256, 700 284" fill="none" stroke="#f6f2db" stroke-width="3" stroke-dasharray="19 18"/>
        <g transform="translate(110 188)" filter="url(#tshadow)">
        <rect x="0" y="42" width="190" height="70" rx="25" fill="#2f6f4e"/>
        <rect x="21" y="10" width="108" height="48" rx="17" fill="#8dc3d5"/>
        <rect x="34" y="19" width="38" height="24" rx="7" fill="#edf5ed"/>
        <rect x="78" y="19" width="37" height="24" rx="7" fill="#edf5ed"/>
        <rect x="148" y="58" width="22" height="13" rx="5" fill="#f1cc6d"/>
        <circle cx="42" cy="115" r="18" fill="#33443b"/>
        <circle cx="151" cy="115" r="18" fill="#33443b"/>
        <circle cx="42" cy="115" r="8" fill="#dfe6d2"/>
        <circle cx="151" cy="115" r="8" fill="#dfe6d2"/>
        </g>
        <path d="M360 275 Q410 251 460 275 T560 275 T660 275" fill="none" stroke="#8fbfc2" stroke-width="12" stroke-linecap="round"/>
        <path d="M360 300 Q410 276 460 300 T560 300 T660 300" fill="none" stroke="#b7d8d7" stroke-width="7" stroke-linecap="round"/>
        <g transform="translate(495 207)" filter="url(#tshadow)">
        <path d="M0 72 Q78 84 160 72 L138 102 Q79 120 22 102 Z" fill="#e8995d"/>
        <rect x="38" y="37" width="81" height="15" rx="7" fill="#ffffff"/>
        <path d="M78 4 L78 38 L111 38 Z" fill="#f1cc6d"/>
        <rect x="46" y="48" width="62" height="7" rx="3" fill="#2f6f4e"/>
        </g>
        <g transform="translate(645 175)" stroke="#2f6f4e" stroke-width="5" stroke-linecap="round" fill="none">
        <path d="M0 90 C-7 45 -4 15 4 -14"/>
        <path d="M2 4 C-24 -18 -42 -19 -57 -13"/>
        <path d="M4 2 C26 -18 45 -19 59 -13"/>
        <path d="M5 11 C30 3 48 8 59 18"/>
        <path d="M2 12 C-24 6 -39 11 -52 23"/>
        </g>
        </svg>'''
# =========================================================
# HELPERS
# =========================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and parse tourism dataset"""
    df = pd.read_csv(path)
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    return df


@st.cache_resource
def load_clustering_models(model_path: str, scaler_path: str):
    """Load clustering model and scaler"""
    km = joblib.load(model_path)
    sc = joblib.load(scaler_path)
    return km, sc


@st.cache_resource
def load_forecasting_model(sarima_path: str):
    """Load SARIMA forecasting model artifact"""
    return joblib.load(sarima_path)


def fmt_number(value) -> str:
    """Format number for display (K, M suffixes)"""
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f} M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f} K"
    return f"{value:,.0f}"


def fmt_number_id(value) -> str:
    """Format number with Indonesian comma style"""
    if pd.isna(value):
        return "—"
    return f"{int(round(float(value))):,}".replace(",", ".")
def fmt_pct(value) -> str:
    """Format percentage"""
    if pd.isna(value):
        return "—"
    return f"{float(value):.1f}%"


@st.cache_data
def make_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create 14-feature table for clustering (cached for high performance)"""
    base = df[
        (df["Jenis Data"] == "Pintu Masuk") & (df["Tahun"].between(2016, 2019))
    ].copy()
    monthly = base.groupby(["Pintu Masuk", "Bulan_Num"])["Jumlah"].mean().unstack()
    month_cols = [
        "Jan_Index", "Feb_Index", "Mar_Index", "Apr_Index",
        "May_Index", "Jun_Index", "Jul_Index", "Aug_Index",
        "Sep_Index", "Oct_Index", "Nov_Index", "Dec_Index",
    ]
    monthly.columns = month_cols
    mean_volume = monthly.mean(axis=1)
    seasonal = monthly.div(mean_volume, axis=0)
    seasonal.columns = month_cols
    feature_volume = np.log1p(mean_volume)
    feature_volume.name = "Log_Mean_Volume"
    recovery_base = df[
        (df["Jenis Data"] == "Pintu Masuk") & (df["Tahun"].isin([2019, 2025]))
    ].copy()
    recovery_total = (
        recovery_base.groupby(["Pintu Masuk", "Tahun"])["Jumlah"]
        .sum(min_count=1).unstack()
    )
    recovery_total = recovery_total.reindex(columns=[2019, 2025])
    recovery_total.columns = ["Baseline_2019", "Value_2025"]
    recovery_total["Recovery_2025"] = (
        recovery_total["Value_2025"] / recovery_total["Baseline_2019"]
    ) * 100
    features = seasonal.join(feature_volume).join(recovery_total["Recovery_2025"])
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features[month_cols + ["Log_Mean_Volume", "Recovery_2025"]]


@st.cache_data
def transport_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract transportation mode data (cached)"""
    route_map = {
        "A. Pintu Udara": "Udara",
        "B. Pintu Laut": "Laut",
        "C. Pintu Darat": "Darat",
    }
    route = df[df["Pintu Masuk"].isin(route_map.keys())].copy()
    route["Rute"] = route["Pintu Masuk"].map(route_map)
    return (
        route.groupby(["Tahun", "Rute"])["Jumlah"]
        .sum(min_count=1).reset_index()
    )


@st.cache_data
def compute_clustering_pipeline(df: pd.DataFrame, model_path: str, scaler_path: str):
    """Compute and cache K-Means clustering features and labels (runs once, 0ms on rerun)"""
    model, scaler = load_clustering_models(model_path, scaler_path)
    features = make_feature_table(df)
    X_scaled = scaler.transform(features)
    labels = model.predict(X_scaled)
    features_with_label = features.copy()
    features_with_label["Cluster"] = labels
    return features, labels, features_with_label


@st.cache_data
def get_dashboard_summary(df: pd.DataFrame):
    """Compute annual totals and KPI snapshots with caching"""
    individual = df[df["Jenis Data"].isin(["Pintu Masuk", "Pintu Masuk Lainnya"])]
    annual = individual.groupby("Tahun")["Jumlah"].sum(min_count=1).reset_index()
    latest_year = int(annual["Tahun"].max())
    latest_total = float(annual.loc[annual["Tahun"] == latest_year, "Jumlah"].iloc[0])
    total_period = float(individual["Jumlah"].sum())
    return annual, latest_year, latest_total, total_period


COORDINATES_MAP = {
    "Adi Sucipto": {"lat": -7.7882, "lon": 110.4319, "provinsi": "DI Yogyakarta", "moda": "Udara", "fasilitas": "Bandara Internasional Adisutjipto"},
    "Ahmad Yani": {"lat": -6.9744, "lon": 110.3744, "provinsi": "Jawa Tengah", "moda": "Udara", "fasilitas": "Bandara Jenderal Ahmad Yani"},
    "Aruk": {"lat": 1.6367, "lon": 109.6586, "provinsi": "Kalimantan Barat", "moda": "Darat", "fasilitas": "PLBN Aruk (Perbatasan Sarawak/Malaysia)"},
    "Atambua": {"lat": -9.0375, "lon": 124.9083, "provinsi": "Nusa Tenggara Timur", "moda": "Darat", "fasilitas": "PLBN Motaain / Atambua (Timor Leste)"},
    "Bandara Int. Lombok": {"lat": -8.7578, "lon": 116.2758, "provinsi": "Nusa Tenggara Barat", "moda": "Udara", "fasilitas": "Bandara Internasional Lombok (ZAM)"},
    "Batam": {"lat": 1.1301, "lon": 104.0530, "provinsi": "Kepulauan Riau", "moda": "Laut", "fasilitas": "Pelabuhan Internasional Feri Batam"},
    "Entikong": {"lat": 0.9995, "lon": 110.3458, "provinsi": "Kalimantan Barat", "moda": "Darat", "fasilitas": "PLBN Entikong (Perbatasan Sarawak/Malaysia)"},
    "Hasanuddin": {"lat": -5.0617, "lon": 119.5540, "provinsi": "Sulawesi Selatan", "moda": "Udara", "fasilitas": "Bandara Internasional Sultan Hasanuddin"},
    "Husein Sastranegara": {"lat": -6.9006, "lon": 107.5761, "provinsi": "Jawa Barat", "moda": "Udara", "fasilitas": "Bandara Husein Sastranegara"},
    "Jayapura": {"lat": -2.5769, "lon": 140.5164, "provinsi": "Papua", "moda": "Darat", "fasilitas": "PLBN Skouw / Gerbang Jayapura (Papua Nugini)"},
    "Juanda": {"lat": -7.3798, "lon": 112.7876, "provinsi": "Jawa Timur", "moda": "Udara", "fasilitas": "Bandara Internasional Juanda"},
    "Kualanamu": {"lat": 3.6422, "lon": 98.8853, "provinsi": "Sumatera Utara", "moda": "Udara", "fasilitas": "Bandara Internasional Kualanamu"},
    "Minangkabau": {"lat": -0.7872, "lon": 100.2806, "provinsi": "Sumatera Barat", "moda": "Udara", "fasilitas": "Bandara Internasional Minangkabau"},
    "Nanga Badau": {"lat": 0.9928, "lon": 111.8742, "provinsi": "Kalimantan Barat", "moda": "Darat", "fasilitas": "PLBN Nanga Badau (Perbatasan Sarawak/Malaysia)"},
    "Ngurah Rai": {"lat": -8.7482, "lon": 115.1672, "provinsi": "Bali", "moda": "Udara", "fasilitas": "Bandara Internasional I Gusti Ngurah Rai (Mega-Hub)"},
    "Sam Ratulangi": {"lat": 1.5492, "lon": 124.9264, "provinsi": "Sulawesi Utara", "moda": "Udara", "fasilitas": "Bandara Internasional Sam Ratulangi"},
    "Soekarno-Hatta": {"lat": -6.1256, "lon": 106.6559, "provinsi": "Banten / DKI Jakarta", "moda": "Udara", "fasilitas": "Bandara Internasional Soekarno-Hatta (Mega-Hub)"},
    "Sultan Badaruddin II": {"lat": -2.8986, "lon": 104.7003, "provinsi": "Sumatera Selatan", "moda": "Udara", "fasilitas": "Bandara Sultan Mahmud Badaruddin II"},
    "Sultan Iskandar Muda": {"lat": 5.5225, "lon": 95.4194, "provinsi": "Aceh", "moda": "Udara", "fasilitas": "Bandara Sultan Iskandar Muda"},
    "Sultan Syarif Kasim II": {"lat": 0.4608, "lon": 101.4447, "provinsi": "Riau", "moda": "Udara", "fasilitas": "Bandara Sultan Syarif Kasim II"},
    "Supadio": {"lat": -0.1506, "lon": 109.4039, "provinsi": "Kalimantan Barat", "moda": "Udara", "fasilitas": "Bandara Supadio"},
    "Tanjung Balai Karimun": {"lat": 0.9931, "lon": 103.4356, "provinsi": "Kepulauan Riau", "moda": "Laut", "fasilitas": "Pelabuhan Tanjung Balai Karimun"},
    "Tanjung Benoa": {"lat": -8.7564, "lon": 115.2178, "provinsi": "Bali", "moda": "Laut", "fasilitas": "Pelabuhan Cruise / Wisata Tanjung Benoa"},
    "Tanjung Mas": {"lat": -6.9536, "lon": 110.4225, "provinsi": "Jawa Tengah", "moda": "Laut", "fasilitas": "Pelabuhan Tanjung Emas"},
    "Tanjung Pinang": {"lat": 0.9250, "lon": 104.4456, "provinsi": "Kepulauan Riau", "moda": "Laut", "fasilitas": "Pelabuhan Sri Bintan Pura"},
    "Tanjung Uban": {"lat": 1.0706, "lon": 104.2217, "provinsi": "Kepulauan Riau", "moda": "Laut", "fasilitas": "Pelabuhan Feri Tanjung Uban (Bintan)"},
}


@st.cache_data
def build_geospatial_dataframe(df: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Build geospatial dataframe with cluster labels and visit statistics (cached)"""
    detail = df[df["Jenis Data"] == "Pintu Masuk"].copy()
    total_visits = detail.groupby("Pintu Masuk")["Jumlah"].sum()
    y2025_visits = detail[detail["Tahun"] == 2025].groupby("Pintu Masuk")["Jumlah"].sum()

    geo_rows = []
    pintu_list = cluster_df.index if not cluster_df.empty else detail["Pintu Masuk"].unique()
    for p in pintu_list:
        info = COORDINATES_MAP.get(p, {})
        c_label = cluster_df.loc[p, "Cluster"] if not cluster_df.empty and p in cluster_df.index else 1
        rec_val = cluster_df.loc[p, "Recovery_2025"] if not cluster_df.empty and p in cluster_df.index else 0.0
        tot = int(total_visits.get(p, 0))
        y25 = int(y2025_visits.get(p, 0))
        
        geo_rows.append({
            "Pintu Masuk": p,
            "Cluster": f"Cluster {int(c_label)} (" + ("Mega-Hub" if int(c_label) == 0 else "Regional Gateway") + ")",
            "Cluster_Id": int(c_label),
            "Latitude": info.get("lat", -2.5),
            "Longitude": info.get("lon", 118.0),
            "Provinsi": info.get("provinsi", "-"),
            "Moda": info.get("moda", "Lainnya"),
            "Fasilitas": info.get("fasilitas", "-"),
            "Total Kunjungan": tot,
            "Kunjungan 2025": y25,
            "Recovery 2025 (%)": round(float(rec_val), 1),
            "Size_Display": float(np.sqrt(tot) / 80 + 10),
        })
    return pd.DataFrame(geo_rows)


def metric_card(label: str, value: str, detail: str = "") -> None:
    """Display metric card"""
    st.markdown(
        f'<div class="clay-card kpi">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="detail">{detail}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_page_banner(tujuan: str, metode: str, value: str) -> None:
    """Render standardized academic & executive context banner on top of each page"""
    st.markdown(
        f"""
        <div class="page-context-banner">
            <div class="context-grid">
                <div class="context-item">
                    <span class="context-label">TUJUAN ANALISIS</span>
                    <span class="context-text">{tujuan}</span>
                </div>
                <div class="context-item">
                    <span class="context-label">PENDEKATAN METODE</span>
                    <span class="context-text">{metode}</span>
                </div>
                <div class="context-item">
                    <span class="context-label">NILAI BAGI PENGAMBIL KEPUTUSAN</span>
                    <span class="context-text">{value}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig):
    """Apply consistent, premium executive styling to all Plotly charts."""
    
    green = "#1f5137"
    white = "#ffffff"
    dark_bg = "#0f2e1a"
    grid_line = "rgba(31, 81, 55, 0.08)"
    axis_line = "rgba(31, 81, 55, 0.15)"
    spike_line = "rgba(31, 81, 55, 0.35)"

    layout_update = dict(
        font=dict(
            family="DM Sans, sans-serif",
            color=green,
            size=12
        ),
        legend=dict(
            font=dict(
                family="DM Sans, sans-serif",
                color=green,
                size=11
            ),
            bgcolor="rgba(255, 255, 255, 0.80)",
            bordercolor="rgba(31, 81, 55, 0.15)",
            borderwidth=1
        ),

        xaxis=dict(
            title_font=dict(color=green, size=12),
            tickfont=dict(color=green, size=11),
            showgrid=True,
            gridcolor=grid_line,
            griddash="dot",
            showline=True,
            linecolor=axis_line,
            linewidth=1,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1.2,
            spikedash="dot",
            spikecolor=spike_line,
        ),

        yaxis=dict(
            title_font=dict(color=green, size=12),
            tickfont=dict(color=green, size=11),
            showgrid=True,
            gridcolor=grid_line,
            griddash="dot",
            showline=True,
            linecolor=axis_line,
            linewidth=1,
            zeroline=False,
        ),

        polar=dict(
            bgcolor="rgba(255, 255, 255, 0.25)",
            angularaxis=dict(
                tickfont=dict(color=green, size=11),
                gridcolor=grid_line,
                linecolor=axis_line,
            ),
            radialaxis=dict(
                tickfont=dict(color=green, size=10),
                title_font=dict(color=green, size=11),
                gridcolor=grid_line,
                linecolor=axis_line,
            )
        ),

        # ==========================================
        # HOVER TOOLTIP (Sleek Dark Glass Pill)
        # ==========================================
        hoverlabel=dict(
            bgcolor=dark_bg,
            bordercolor="#00a86b",
            font=dict(
                family="DM Sans, sans-serif",
                color=white,
                size=12
            ),
            align="left",
            namelength=-1
        )
    )

    if fig.layout.title and fig.layout.title.text:
        layout_update["title_font"] = dict(family="Baloo 2, sans-serif", color="#133426", size=15)

    fig.update_layout(**layout_update)

    # Data label / text & bar styling pada chart
    for trace in fig.data:
        if hasattr(trace, "textfont"):
            try:
                trace.textfont = dict(
                    family="DM Sans, sans-serif",
                    color=green,
                    size=11
                )
            except Exception:
                pass
        # Rounded corners for bar charts if supported
        if trace.type == "bar":
            if hasattr(trace, "marker") and trace.marker is not None:
                try:
                    trace.marker.cornerradius = 6
                except Exception:
                    pass

    return fig


PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
}


def render_chart(fig):
    """Render Plotly chart with optimized, lightweight toolbar and responsive settings."""
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


@st.cache_data
def build_monthly_series(df: pd.DataFrame) -> pd.Series:
    """Build monthly tourism series — sama dengan logic forecasting.ipynb (cached)."""
    
    df_detail = df[
        df["Jenis Data"].isin([
            "Pintu Masuk",
            "Pintu Masuk Lainnya"
        ])
    ].copy()

    monthly = (
        df_detail
        .groupby("Tanggal")["Jumlah"]
        .sum(min_count=1)
        .sort_index()
    )

    # Pastikan index berupa DatetimeIndex
    monthly.index = pd.to_datetime(monthly.index)

    # Gunakan frekuensi bulanan MS
    monthly = monthly.asfreq("MS")

    # Ambil hanya bulan yang memiliki data aktual
    monthly_known = monthly.dropna()

    return monthly_known


BULAN_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}


def fmt_date_id(ts) -> str:
    """Format timestamp as Indonesian month-year"""
    return f"{BULAN_ID.get(ts.month, ts.strftime('%B'))} {ts.year}"


@st.cache_data
def run_granular_forecast(df: pd.DataFrame, target_type: str, target_name: str, steps: int = 12):
    """Extract time-series and compute SARIMA(0,1,1)(0,1,1,12) forecast for specific entry point or mode."""
    if target_type == "nasional":
        df_detail = df[df["Jenis Data"].isin(["Pintu Masuk", "Pintu Masuk Lainnya"])].copy()
        s = df_detail.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()
        s.index = pd.to_datetime(s.index)
        s = s.asfreq("MS").dropna()
    elif target_type == "moda":
        route_map = {"Udara": "A. Pintu Udara", "Laut": "B. Pintu Laut", "Darat": "C. Pintu Darat"}
        p_key = route_map.get(target_name, target_name)
        sub = df[df["Pintu Masuk"] == p_key].copy()
        s = sub.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()
        s.index = pd.to_datetime(s.index)
        s = s.asfreq("MS").dropna()
    else:  # 'pintu_masuk'
        sub = df[df["Pintu Masuk"] == target_name].copy()
        s = sub.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()
        s.index = pd.to_datetime(s.index)
        s = s.asfreq("MS").dropna()

    model = SARIMAX(
        s,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps)
    mean_val = fc.predicted_mean.clip(lower=0)
    conf_val = fc.conf_int().clip(lower=0)
    return s, mean_val, conf_val


@st.cache_data
def compute_policy_simulation(
    base_forecast_mean: pd.Series,
    base_forecast_ci: pd.DataFrame,
    flight_cap_pct: float,
    visa_ease_pct: float,
    aviation_cost_pct: float,
    beyond_bali_pct: float,
    spend_per_tourist_usd: float = 1350.0,
    usd_idr_rate: float = 16000.0,
):
    """
    Simulate tourism volume and foreign exchange (devisa) based on policy levers.
    """
    net_growth_pct = (flight_cap_pct * 0.65) + (visa_ease_pct * 0.50) - (aviation_cost_pct * 0.45)
    multiplier = max(0.1, 1.0 + (net_growth_pct / 100.0))

    sim_mean = (base_forecast_mean * multiplier).clip(lower=0)
    sim_ci_lower = (base_forecast_ci.iloc[:, 0] * multiplier).clip(lower=0)
    sim_ci_upper = (base_forecast_ci.iloc[:, 1] * multiplier).clip(lower=0)

    total_base_wisman = float(base_forecast_mean.sum())
    total_sim_wisman = float(sim_mean.sum())
    delta_wisman = total_sim_wisman - total_base_wisman
    pct_growth_effective = (
        ((total_sim_wisman - total_base_wisman) / total_base_wisman) * 100.0
        if total_base_wisman > 0
        else 0.0
    )

    devisa_base_usd = (total_base_wisman * spend_per_tourist_usd) / 1e6
    devisa_sim_usd = (total_sim_wisman * spend_per_tourist_usd) / 1e6
    delta_devisa_usd = devisa_sim_usd - devisa_base_usd
    delta_devisa_idr = (delta_devisa_usd * 1e6 * usd_idr_rate) / 1e12

    c1_base_share = 0.195
    c1_target_share = min(0.45, c1_base_share * (1.0 + beyond_bali_pct / 100.0))
    c0_target_share = 1.0 - c1_target_share

    c0_base_total = total_base_wisman * (1.0 - c1_base_share)
    c1_base_total = total_base_wisman * c1_base_share

    c0_sim_total = total_sim_wisman * c0_target_share
    c1_sim_total = total_sim_wisman * c1_target_share

    c0_delta = c0_sim_total - c0_base_total
    c1_delta = c1_sim_total - c1_base_total

    df_sim = pd.DataFrame(
        {
            "Tanggal": base_forecast_mean.index,
            "Baseline_SARIMA": base_forecast_mean.values,
            "Skenario_Kebijakan": sim_mean.values,
            "Selisih_Wisman": (sim_mean - base_forecast_mean).values,
            "CI_Lower": sim_ci_lower.values,
            "CI_Upper": sim_ci_upper.values,
        }
    )
    df_sim["Devisa_Baseline_MUSD"] = (df_sim["Baseline_SARIMA"] * spend_per_tourist_usd) / 1e6
    df_sim["Devisa_Skenario_MUSD"] = (df_sim["Skenario_Kebijakan"] * spend_per_tourist_usd) / 1e6
    df_sim["Delta_Devisa_MUSD"] = df_sim["Devisa_Skenario_MUSD"] - df_sim["Devisa_Baseline_MUSD"]
    df_sim["Delta_Devisa_TIDR"] = (df_sim["Delta_Devisa_MUSD"] * 1e6 * usd_idr_rate) / 1e12

    return {
        "df_sim": df_sim,
        "total_base_wisman": total_base_wisman,
        "total_sim_wisman": total_sim_wisman,
        "delta_wisman": delta_wisman,
        "pct_growth_effective": pct_growth_effective,
        "net_growth_pct": net_growth_pct,
        "devisa_base_usd": devisa_base_usd,
        "devisa_sim_usd": devisa_sim_usd,
        "delta_devisa_usd": delta_devisa_usd,
        "delta_devisa_idr": delta_devisa_idr,
        "c0_base_total": c0_base_total,
        "c0_sim_total": c0_sim_total,
        "c0_delta": c0_delta,
        "c1_base_total": c1_base_total,
        "c1_sim_total": c1_sim_total,
        "c1_delta": c1_delta,
        "c1_base_share": c1_base_share * 100.0,
        "c1_target_share": c1_target_share * 100.0,
    }


@st.cache_data
def detect_anomalies_pipeline(
    df: pd.DataFrame,
    target_type: str = "nasional",
    target_name: str = "Nasional",
    contamination: float = 0.08,
):
    """
    Extract time-series and detect crisis anomalies and positive surges using Isolation Forest.
    """
    if target_type == "nasional":
        sub = df[df["Jenis Data"].isin(["Pintu Masuk", "Pintu Masuk Lainnya"])].copy()
        s = sub.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()
    elif target_type == "moda":
        route_map = {"Udara": "A. Pintu Udara", "Laut": "B. Pintu Laut", "Darat": "C. Pintu Darat"}
        p_key = route_map.get(target_name, target_name)
        sub = df[df["Pintu Masuk"] == p_key].copy()
        s = sub.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()
    else:  # specific entry point
        sub = df[df["Pintu Masuk"] == target_name].copy()
        s = sub.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()

    s.index = pd.to_datetime(s.index)
    s = s.asfreq("MS").dropna()

    df_ts = pd.DataFrame({"Jumlah": s})
    df_ts["MoM_Pct"] = df_ts["Jumlah"].pct_change() * 100
    df_ts["YoY_Pct"] = df_ts["Jumlah"].pct_change(12) * 100

    base_16_19 = df_ts[df_ts.index.year.isin(range(2016, 2020))]
    if len(base_16_19) > 0:
        seasonal_mean = base_16_19.groupby(base_16_19.index.month)["Jumlah"].mean()
        df_ts["Seasonal_Mean"] = [seasonal_mean.get(m, df_ts["Jumlah"].mean()) for m in df_ts.index.month]
        df_ts["Seasonal_Dev_Pct"] = (
            (df_ts["Jumlah"] - df_ts["Seasonal_Mean"]) / df_ts["Seasonal_Mean"].replace(0, 1)
        ) * 100
    else:
        df_ts["Seasonal_Mean"] = df_ts["Jumlah"].mean()
        df_ts["Seasonal_Dev_Pct"] = 0.0

    features = df_ts[["Jumlah", "MoM_Pct", "YoY_Pct", "Seasonal_Dev_Pct"]].fillna(0)
    iso = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    df_ts["Is_Anomaly"] = iso.fit_predict(features) == -1
    df_ts["Anomaly_Score"] = -iso.score_samples(features)

    def classify_type(row):
        if not row["Is_Anomaly"]:
            return "Normal"
        if row["Seasonal_Dev_Pct"] < -25 or row["MoM_Pct"] < -25 or row["YoY_Pct"] < -30:
            return "Guncangan Negatif (Shock)"
        elif row["Seasonal_Dev_Pct"] > 20 or row["MoM_Pct"] > 25 or row["YoY_Pct"] > 40:
            return "Lonjakan Positif (Surge)"
        return "Variasi Anomali"

    df_ts["Anomaly_Type"] = df_ts.apply(classify_type, axis=1)

    event_map = {
        "2017-11": "Erupsi Gunung Agung Bali (Penutupan Bandara)",
        "2017-12": "Erupsi Gunung Agung Bali (Drop Kunjungan Akhir Tahun)",
        "2020-03": "Awal Pandemi COVID-19 & Global Travel Advisory",
        "2020-04": "Lockdown Global COVID-19 & Border Closure Masif",
        "2020-05": "Lockdown Global COVID-19 & Penerbangan Internasional Terhenti",
        "2021-07": "Puncak Gelombang Varian Delta & PPKM Darurat Level 4",
        "2021-08": "PPKM Darurat Varian Delta (Second Trough)",
        "2022-03": "Re-Opening Perbatasan Bali & Kebijakan Bebas Karantina",
        "2022-04": "Perluasan VoA & Rebound Arus Wisatawan Pasca-Pelonggaran",
        "2022-05": "Akselerasi Rebound Pasca-Pelonggaran PCR Internasional",
        "2022-11": "Penyelenggaraan KTT Presidensi G20 Bali",
        "2022-12": "Rebound Akhir Tahun & Pemulihan Liburan Global",
        "2023-01": "Pembukaan Kembali Perbatasan Tiongkok Pasca Zero-COVID",
        "2024-07": "Puncak Musim Panas Global (Summer Peak) Pasca-Pandemi",
    }
    df_ts["Event_Name"] = [event_map.get(d.strftime("%Y-%m"), "") for d in df_ts.index]

    return df_ts


@st.cache_data
def compute_rebound_analysis(df: pd.DataFrame):
    """
    Compute post-crisis rebound velocity and recovery milestones.
    """
    sub = df[df["Jenis Data"].isin(["Pintu Masuk", "Pintu Masuk Lainnya"])].copy()
    s = sub.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()
    s.index = pd.to_datetime(s.index)

    base_2019_mean = float(s[s.index.year == 2019].mean())
    trough_val = float(s["2020-01-01":"2021-12-01"].min())
    trough_date = s["2020-01-01":"2021-12-01"].idxmin()

    post_trough = s[s.index >= trough_date]
    milestones = {}
    for pct in [25, 50, 75, 100]:
        target = base_2019_mean * (pct / 100.0)
        reached = post_trough[post_trough >= target]
        if len(reached) > 0:
            d = reached.index[0]
            months_diff = (d.year - trough_date.year) * 12 + (d.month - trough_date.month)
            milestones[f"{pct}%"] = {
                "date_str": fmt_date_id(d),
                "months": months_diff,
                "target": target,
                "achieved": True,
            }
        else:
            milestones[f"{pct}%"] = {
                "date_str": "Dalam Proyeksi",
                "months": None,
                "target": target,
                "achieved": False,
            }

    route_map = {"Udara": "A. Pintu Udara", "Laut": "B. Pintu Laut", "Darat": "C. Pintu Darat"}
    moda_stats = {}
    for m_label, p_key in route_map.items():
        m_sub = df[df["Pintu Masuk"] == p_key].copy()
        ms = m_sub.groupby("Tanggal")["Jumlah"].sum(min_count=1).sort_index()
        ms.index = pd.to_datetime(ms.index)
        ms_19 = float(ms[ms.index.year == 2019].mean()) if len(ms[ms.index.year == 2019]) > 0 else 1.0
        ms_25 = float(ms[ms.index.year == 2025].mean()) if len(ms[ms.index.year == 2025]) > 0 else 0.0
        rec_rate = (ms_25 / ms_19 * 100.0) if ms_19 > 0 else 0.0
        moda_stats[m_label] = {
            "mean_2019": ms_19,
            "mean_2025": ms_25,
            "recovery_rate": rec_rate,
        }

    return {
        "base_2019_mean": base_2019_mean,
        "trough_val": trough_val,
        "trough_date": trough_date,
        "milestones": milestones,
        "moda_stats": moda_stats,
    }


HISTORICAL_EVENTS_DOSSIER = [
    {
        "periode": "Nov 2017 – Jan 2018",
        "judul": "Erupsi Gunung Agung (Bali)",
        "tipe": "crisis",
        "tipe_label": "Bencana Alam & Penutupan Bandara",
        "lokasi": "Bali (Bandara I Gusti Ngurah Rai)",
        "dampak": "Penurunan tajam kunjungan wisman ke Bali hingga -48.5% dalam rentang 4 bulan akibat abu vulkanik dan penutupan ruang udara internasional.",
        "icon": "",
    },
    {
        "periode": "Maret – April 2020",
        "judul": "Awal Pandemi COVID-19 & Global Lockdown",
        "tipe": "crisis",
        "tipe_label": "Krisis Global & Penutupan Perbatasan",
        "lokasi": "Nasional & Seluruh Pintu Masuk",
        "dampak": "Guncangan terdalam dalam sejarah pariwisata modern. Kunjungan wisman anjlok dari 1.25 juta menjadi 158 ribu (-87.6% YoY) dalam kurun waktu 60 hari.",
        "icon": "",
    },
    {
        "periode": "Juli – Agustus 2021",
        "judul": "Gelombang Varian Delta & PPKM Darurat",
        "tipe": "crisis",
        "tipe_label": "Puncak Pandemi Gelombang II",
        "lokasi": "Jawa-Bali & Pintu Internasional",
        "dampak": "Titik nadir kedua (second trough) dengan rata-rata kunjungan nasional bertahan di bawah 125 ribu kunjungan/bulan selama pembatasan mobilitas level 4.",
        "icon": "",
    },
    {
        "periode": "Februari – Mei 2022",
        "judul": "Pelonggaran Karantina & Re-opening VoA Khusus",
        "tipe": "rebound",
        "tipe_label": "Titik Balik Pemulihan (Turning Point)",
        "lokasi": "Ngurah Rai Bali & Soekarno-Hatta",
        "dampak": "Pemberlakuan bebas karantina dan pembukaan Visa on Arrival khusus wisata memicu akselerasi kunjungan hingga melonjak +154% YoY dalam kurun waktu 3 bulan.",
        "icon": "",
    },
    {
        "periode": "November 2022",
        "judul": "KTT Presidensi G20 Bali",
        "tipe": "event",
        "tipe_label": "Event Internasional Multilateral",
        "lokasi": "Nusa Dua, Bali",
        "dampak": "Penyelenggaraan KTT G20 menjadi katalis pemulihan citra pariwisata Indonesia, mendatangkan ribuan delegasi dan mendongkrak kunjungan nasional hingga +482% YoY.",
        "icon": "",
    },
    {
        "periode": "Tahun 2023 – 2025",
        "judul": "Akselerasi Event Super & Pemulihan Penuh",
        "tipe": "event",
        "tipe_label": "Ekspansi Rute & Destinasi Super Prioritas",
        "lokasi": "Mandalika (MotoGP), Jakarta, Bali, Danau Toba",
        "dampak": "Kunjungan bulanan kembali menembus angka 1.34 juta kunjungan pada Juni 2025, secara resmi melampaui rata-rata bulanan pra-pandemi 2019 (100% Recovery).",
        "icon": "",
    },
]


# =========================================================
# DATA & MODEL LOADING
# =========================================================
if not DATA_PATH.exists():
    st.error(f"Dataset tidak ditemukan.\n`{DATA_PATH}`")
    st.stop()

df = load_data(str(DATA_PATH))

# Initialize clustering components (fully cached for instant response)
clustering_ok = MODEL_PATH.exists() and SCALER_PATH.exists()
cluster_features = pd.DataFrame()
cluster_labels = np.array([])
cluster_features_with_label = pd.DataFrame()
if clustering_ok:
    try:
        cluster_features, cluster_labels, cluster_features_with_label = compute_clustering_pipeline(
            df, str(MODEL_PATH), str(SCALER_PATH)
        )
    except Exception as e:
        clustering_ok = False
        st.warning(f"Clustering model gagal dimuat: {e}")

# Initialize SARIMA components
sarima_ok = SARIMA_PATH.exists()
sarima_artifact = None
if sarima_ok:
    try:
        sarima_artifact = load_forecasting_model(str(SARIMA_PATH))
    except Exception as e:
        sarima_ok = False
        st.warning(f"Model SARIMA gagal dimuat: {e}")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="brand-badge">INDONESIA TOURISM</div>
        <div class="brand-title">Nusantara<br>Transit Intelligence</div>
        <div class="brand-sub">Mobility & Tourism Analytics • 2016–2026</div>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_PAGES = {
    "Dashboard": "Dashboard",
    "Peta Geospasial": "Peta Geospasial",
    "Cluster Explorer": "Cluster Explorer", 
    "Forecasting": "Forecasting",
    "Simulator Kebijakan": "Simulator Kebijakan",
    "Deteksi Anomali": "Deteksi Anomali",
    "Data Explorer": "Data Explorer",
    "Tentang Model": "Tentang Model",
}

selected_label = st.sidebar.radio(
    "Navigasi Utama",
    list(NAV_PAGES.values()),
    label_visibility="collapsed",
)
page = [k for k, v in NAV_PAGES.items() if v == selected_label][0]

st.sidebar.markdown(
    """
    <div class="sidebar-meta-card">
        <div class="meta-status">
            <span class="status-dot"></span> ML System Active
        </div>
        <div class="meta-grid">
            <div class="meta-item">
                <span class="meta-label">Tema</span>
                <span class="meta-val">Aviation Glassmorphism</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Clustering</span>
                <span class="meta-val">K-Means (k=2)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Forecasting</span>
                <span class="meta-val">SARIMA (12-Mo Ahead)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Simulasi</span>
                <span class="meta-val">What-If DSS Engine</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Anomali</span>
                <span class="meta-val">Isolation Forest (127 Bln)</span>
            </div>
        </div>
        <div class="meta-foot">Polman Machine Learning • Kelompok 6</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Executive Intelligence • Macro Overview</div>'
        "<h1>Indonesia Tourism,<br>in Number.</h1>"
        "<p>Pusat kendali analitik dan pemantauan makro kunjungan wisatawan "
        "mancanegara ke Indonesia (2016–2026) berbasis data BPS dan integrasi Machine Learning.</p>"
        '<div style="margin-top:16px;">'
        '<span class="transport-pill">Pintu Udara</span>'
        '<span class="transport-pill">Pintu Darat (PLBN)</span>'
        '<span class="transport-pill">Pintu Laut (Pelabuhan)</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Menyajikan gambaran makro dan indikator kinerja utama (KPI) pariwisata mancanegara Indonesia periode 2016–2026 pasca-krisis global.",
        metode="Agregasi data bulanan 26 pintu masuk resmi BPS, dikombinasikan dengan dekomposisi moda transportasi dan ringkasan model AI.",
        value="Membuktikan secara data bahwa pariwisata nasional telah resmi pulih melampaui level pra-pandemi 2019, dengan dominasi mutlak jalur penerbangan (>70%).",
    )
# KPI snapshot (cached)
    annual, latest_year, latest_total, total_period = get_dashboard_summary(df)
    n_entry = int(cluster_features.index.nunique()) if clustering_ok else "—"
    n_cluster = int(len(np.unique(cluster_labels))) if clustering_ok else "—"

    st.markdown('<div class="section-title">Indikator Kunci Pariwisata Nasional</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Kunjungan 2016–2026", fmt_number(total_period), "Data bersih BPS")
    with cols[1]:
        metric_card(f"Kunjungan Tahun {latest_year}", fmt_number(latest_total), "Tahun berjalan")
    with cols[2]:
        metric_card("Pintu Masuk Terpantau", str(n_entry), "26 Pintu Utama BPS")
    with cols[3]:
        metric_card("Struktur Klaster AI", str(n_cluster), "K-Means (k = 2)")

    st.markdown(
        '<div class="note-box" style="margin-top:8px;">'
        "<b>Catatan Integritas Data:</b> Tahun 2026 memuat data aktual hingga <b>Juli 2026</b>. "
        "Periode Agustus 2026 – Juli 2027 sepenuhnya merupakan hasil proyeksi model time-series SARIMA."
        "</div>",
        unsafe_allow_html=True,
    )

    # Executive Insights
    st.markdown('<div class="section-title">Insight Strategis untuk Audiens & Pimpinan</div>', unsafe_allow_html=True)
    insights = []
    pre_covid = annual[annual["Tahun"] < 2020]
    if not pre_covid.empty:
        peak_row = pre_covid.loc[pre_covid["Jumlah"].idxmax()]
        insights.append((
            "",
            f"<b>Puncak Pra-Pandemi:</b> Rekor tertinggi sebelum krisis terjadi pada <b>{int(peak_row['Tahun'])}</b> "
            f"dengan total <b>{fmt_number(peak_row['Jumlah'])}</b> wisman.",
        ))
    y2019 = annual.loc[annual["Tahun"] == 2019, "Jumlah"]
    y2025 = annual.loc[annual["Tahun"] == 2025, "Jumlah"]
    if not y2019.empty and not y2025.empty:
        rec = float(y2025.iloc[0]) / float(y2019.iloc[0]) * 100
        insights.append((
            "",
            f"<b>Milestone Pemulihan:</b> Tahun 2025 mencatat pemulihan sebesar <b>{rec:.1f}%</b> "
            f"terhadap baseline 2019, membuktikan ketahanan pariwisata Indonesia.",
        ))
    route_df = transport_mix(df)
    latest_route = route_df[route_df["Tahun"] == latest_year]
    if not latest_route.empty:
        dom = latest_route.loc[latest_route["Jumlah"].idxmax()]
        dom_share = dom["Jumlah"] / latest_route["Jumlah"].sum() * 100
        insights.append((
            "",
            f"<b>Jalur Masuk Utama:</b> Moda <b>{dom['Rute']}</b> menyerap <b>{dom_share:.1f}%</b> "
            f"dari seluruh wisman, menjadi tulang punggung mobilitas devisa.",
        ))
    if clustering_ok and len(cluster_labels) > 0:
        c0n = int(np.sum(cluster_labels == 0))
        c1n = int(np.sum(cluster_labels == 1))
        insights.append((
            "",
            f"<b>Dualitas Gerbang Masuk:</b> Model K-Means mengidentifikasi <b>{c0n} Mega-Hubs</b> (Bali & Jakarta) "
            f"dan <b>{c1n} Regional Gateways</b> yang membutuhkan akselerasi rute baru.",
        ))

    cols_ins = st.columns(2)
    for i, (icon, text) in enumerate(insights):
        with cols_ins[i % 2]:
            st.markdown(
                f'<div class="insight-card">'
                f''
                f'<div class="insight-text">{text}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # TABS: EKSPLORASI TERINTEGRASI (MAKRO, TRANSPORTASI, KLASTER)
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Eksplorasi Analitik Terpadu</div>', unsafe_allow_html=True)
    dash_tab1, dash_tab2, dash_tab3 = st.tabs([
        "Tren Makro Tahunan",
        "Jalur & Moda Transportasi (Terintegrasi)",
        "Struktur Klaster & Teaser Prediksi",
    ])

    with dash_tab1:
        st.markdown(
            '<div class="clay-card">'
            '<h3 style="font-family:\'Baloo 2\',sans-serif;margin:0 0 6px;">Trajektori Kunjungan Wisman Tahunan (2016–2026)</h3>'
            '<p style="font-size:0.85rem;color:#55755e;margin-bottom:12px;">'
            'Perjalanan kurva V-shape: dari pertumbuhan pra-pandemi, guncangan COVID-19 (2020–2021), '
            'hingga fase pemulihan eksponensial (2022–2026).'
            '</p>',
            unsafe_allow_html=True,
        )
        fig_ann = go.Figure()
        fig_ann.add_trace(
            go.Scatter(
                x=annual["Tahun"],
                y=annual["Jumlah"],
                mode="lines+markers",
                name="Wisatawan Mancanegara",
                line=dict(color="#00A86B", width=4, shape="spline", smoothing=1.3),
                fill="tozeroy",
                fillcolor="rgba(0, 168, 107, 0.16)",
                marker=dict(
                    size=10,
                    color="#ffffff",
                    line=dict(color="#00A86B", width=3),
                ),
                hovertemplate="<b>Tahun %{x}</b><br>Kunjungan: <b>%{y:,.0f} Wisman</b><extra></extra>",
            )
        )

        # Milestone Callout Annotations
        fig_ann.add_annotation(
            x=2019,
            y=16106954,
            text="<b>Rekor Pra-Pandemi</b><br>16,11 Jt Wisman",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#00A86B",
            ax=0,
            ay=-55,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#00A86B",
            borderwidth=1.5,
            borderpad=6,
            font=dict(size=11, color="#133426", family="DM Sans"),
        )
        fig_ann.add_annotation(
            x=2021,
            y=1557530,
            text="<b>Palung Pandemi</b><br>1,56 Jt Wisman",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#e63946",
            ax=0,
            ay=-50,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e63946",
            borderwidth=1.5,
            borderpad=6,
            font=dict(size=11, color="#900C3F", family="DM Sans"),
        )
        fig_ann.add_annotation(
            x=2025,
            y=15386646,
            text="<b>Pemulihan 2025</b><br>15,39 Jt (95.5%)",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#0077b6",
            ax=-35,
            ay=-55,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#0077b6",
            borderwidth=1.5,
            borderpad=6,
            font=dict(size=11, color="#005f73", family="DM Sans"),
        )

        fig_ann.update_layout(
            margin=dict(l=10, r=10, t=25, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.45)",
            hovermode="x unified",
            height=400,
            xaxis=dict(
                title="Tahun",
                dtick=1,
                showgrid=True,
                gridcolor="rgba(31,81,55,0.08)",
            ),
            yaxis=dict(
                title="Wisatawan Mancanegara",
                showgrid=True,
                gridcolor="rgba(31,81,55,0.08)",
                ticksuffix=" ",
            ),
            showlegend=False,
        )
        style_plotly(fig_ann)
        render_chart(fig_ann)
        st.markdown("</div>", unsafe_allow_html=True)

    with dash_tab2:
        st.markdown(
            '<div class="clay-card">'
            '<h3 style="font-family:\'Baloo 2\',sans-serif;margin:0 0 6px;">Komposisi & Perbandingan Jalur Masuk (Udara, Laut, Darat)</h3>'
            '<p style="font-size:0.85rem;color:#55755e;margin-bottom:14px;">'
            'Evaluasi kontribusi masing-masing jalur transportasi dalam mendukung arus masuk wisatawan mancanegara.'
            '</p>',
            unsafe_allow_html=True,
        )

        route_colors = {
            "Udara": "#0077b6",  # Ocean Azure
            "Laut": "#00a86b",   # Sea Emerald
            "Darat": "#f77f00",  # Sun Amber
        }
        latest_year_rt = int(route_df["Tahun"].max())
        latest_route_sub = route_df[route_df["Tahun"] == latest_year_rt].copy()

        if not latest_route_sub.empty:
            total_rt = latest_route_sub["Jumlah"].sum()
            rt_cols = st.columns(3)
            for idx, rute_name in enumerate(["Udara", "Darat", "Laut"]):
                row_r = latest_route_sub[latest_route_sub["Rute"] == rute_name]
                val_r = float(row_r["Jumlah"].iloc[0]) if not row_r.empty else 0.0
                with rt_cols[idx]:
                    metric_card(
                        f"Jalur {rute_name} ({latest_year_rt})",
                        fmt_number(val_r),
                        f"{val_r/total_rt*100:.1f}% dari total" if total_rt > 0 else "",
                    )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(
                '<h4 style="font-family:\'Baloo 2\',sans-serif;margin:12px 0 6px;color:#133426;">Tren Moda Transportasi</h4>',
                unsafe_allow_html=True,
            )
            fig_tr_dash = px.area(
                route_df,
                x="Tahun",
                y="Jumlah",
                color="Rute",
                markers=True,
                labels={"Jumlah": "Wisatawan", "Tahun": "Tahun", "Rute": "Moda"},
                color_discrete_map=route_colors,
            )
            fig_tr_dash.update_traces(
                line_shape="spline",
                line=dict(width=3),
                marker=dict(size=7, line=dict(width=1.5, color="#ffffff")),
                hovertemplate="<b>%{data.name} (%{x})</b><br>Volume: <b>%{y:,.0f} Wisman</b><extra></extra>",
            )
            fig_tr_dash.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.40)",
                hovermode="x unified",
                height=320,
                xaxis=dict(dtick=1),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            style_plotly(fig_tr_dash)
            render_chart(fig_tr_dash)

        with col_t2:
            st.markdown(
                f'<h4 style="font-family:\'Baloo 2\',sans-serif;margin:12px 0 6px;color:#133426;">Komposisi {latest_year_rt}</h4>',
                unsafe_allow_html=True,
            )
            fig_bar_dash = px.bar(
                latest_route_sub.sort_values("Jumlah", ascending=False),
                x="Rute",
                y="Jumlah",
                text="Jumlah",
                color="Rute",
                labels={"Jumlah": "Wisatawan", "Rute": "Jalur"},
                color_discrete_map=route_colors,
            )
            fig_bar_dash.update_traces(
                texttemplate="<b>%{text:,.0f}</b>",
                textposition="outside",
                hovertemplate="<b>Jalur %{x}</b><br>Volume: <b>%{y:,.0f} Wisman</b><extra></extra>",
            )
            fig_bar_dash.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.40)",
                showlegend=False,
                height=320,
            )
            style_plotly(fig_bar_dash)
            render_chart(fig_bar_dash)

        # Pemulihan per moda terhadap 2019
        pivot_m = route_df[route_df["Tahun"].isin([2019, 2025])].pivot_table(
            index="Rute", columns="Tahun", values="Jumlah", aggfunc="sum"
        )
        if 2019 in pivot_m.columns and 2025 in pivot_m.columns:
            pivot_m["Recovery_2025"] = pivot_m[2025] / pivot_m[2019] * 100
            recovery_m = pivot_m.reset_index()[["Rute", "Recovery_2025"]]
            st.markdown(
                '<h4 style="font-family:\'Baloo 2\',sans-serif;margin:16px 0 6px;color:#133426;">Tingkat Pemulihan 2025 vs Baseline Pra-Pandemi (2019)</h4>',
                unsafe_allow_html=True,
            )
            fig_rec_m = px.bar(
                recovery_m.sort_values("Recovery_2025", ascending=False),
                x="Rute",
                y="Recovery_2025",
                text="Recovery_2025",
                color="Rute",
                labels={"Recovery_2025": "Pemulihan (%)", "Rute": "Jalur"},
                color_discrete_map=route_colors,
            )
            fig_rec_m.add_hline(
                y=100,
                line_dash="dash",
                line_color="#00A86B",
                line_width=2,
                annotation_text="Ambang 100% Pra-Pandemi",
                annotation_position="top left",
                annotation_font=dict(color="#00A86B", size=11, family="DM Sans"),
            )
            fig_rec_m.update_traces(
                texttemplate="<b>%{text:.1f}%</b>",
                textposition="outside",
                hovertemplate="<b>Jalur %{x}</b><br>Pemulihan: <b>%{y:.1f}%</b> vs 2019<extra></extra>",
            )
            fig_rec_m.update_layout(
                margin=dict(l=0, r=0, t=15, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.40)",
                showlegend=False,
                height=280,
                yaxis=dict(ticksuffix="%"),
            )
            style_plotly(fig_rec_m)
            render_chart(fig_rec_m)

        st.markdown("</div>", unsafe_allow_html=True)

    with dash_tab3:
        if clustering_ok:
            st.markdown(
                '<div class="clay-card">'
                '<h3 style="font-family:\'Baloo 2\',sans-serif;margin:0 0 6px;">Ringkasan Segmentasi AI (K-Means)</h3>'
                '<p style="font-size:0.85rem;color:#55755e;margin-bottom:14px;">'
                'Klasifikasi 26 pintu masuk ke dalam 2 klaster utama berdasarkan 14 fitur deret waktu.'
                '</p>',
                unsafe_allow_html=True,
            )
            profile = cluster_features_with_label.copy()
            profile.index.name = None
            profile = profile.reset_index().rename(columns={"index": "Pintu Masuk"})
            c0_list = profile[profile["Cluster"] == 0]["Pintu Masuk"].tolist()
            c1_list = profile[profile["Cluster"] == 1]["Pintu Masuk"].tolist()
            col_a, col_b = st.columns(2)
            with col_a:
                c0_names = " • ".join(c0_list)
                st.markdown(
                    f'<div class="clay-card" style="margin-bottom:0;">'
                    f'<span class="cluster-chip cluster-0">CLUSTER 0 • MEGA-HUBS</span>'
                    f'<h4 style="font-family:\'Baloo 2\',sans-serif;margin:10px 0 4px;">{len(c0_list)} Pintu Masuk Utama</h4>'
                    f'<div class="cluster-members">{c0_names}</div>'
                    f'<p style="font-size:.82rem;margin-top:10px;color:#2d4d33;">'
                    f"Menyerap >80% total wisman nasional dengan volume masif dan pola musiman puncak liburan dunia.</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_b:
                c1_preview = " • ".join(c1_list[:8])
                remainder = len(c1_list) - 8
                st.markdown(
                    f'<div class="clay-card" style="margin-bottom:0;">'
                    f'<span class="cluster-chip cluster-1">CLUSTER 1 • REGIONAL GATEWAYS</span>'
                    f'<h4 style="font-family:\'Baloo 2\',sans-serif;margin:10px 0 4px;">{len(c1_list)} Pintu Masuk Regional</h4>'
                    f'<div class="cluster-members">{c1_preview} ... dan {remainder} lainnya</div>'
                    f'<p style="font-size:.82rem;margin-top:10px;color:#2d4d33;">'
                    f"Gerbang perbatasan darat, pelabuhan laut, dan bandara sekunder sebagai penopang pemerataan Beyond Bali.</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Forecast teaser
        if sarima_ok and sarima_artifact is not None:
            try:
                _fc = sarima_artifact["model"].get_forecast(steps=12).predicted_mean
                avg_fc = _fc.mean()
                peak_fc = _fc.max()
                peak_m = _fc.idxmax()
                st.markdown(
                    '<div class="clay-card" style="margin-top:14px;">'
                    '<h3 style="font-family:\'Baloo 2\',sans-serif;margin:0 0 4px;">Teaser Peramalan SARIMA (12 Bulan Kedepan)</h3>'
                    '<p style="font-size:0.84rem;color:#55755e;margin-bottom:12px;">'
                    'Proyeksi otomatis model SARIMA(0,1,1)(0,1,1,12) untuk periode Agustus 2026 – Juli 2027. Buka menu <b>Forecasting</b> untuk rincian lengkap.'
                    '</p>',
                    unsafe_allow_html=True,
                )
                tc = st.columns(3)
                with tc[0]:
                    metric_card("Rata-Rata Proyeksi", fmt_number(avg_fc), "Agu 2026 – Jul 2027")
                with tc[1]:
                    metric_card("Puncak Kunjungan", fmt_number(peak_fc), fmt_date_id(peak_m))
                with tc[2]:
                    metric_card("Bulan Puncak Musiman", fmt_date_id(peak_m), "High Season")
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                pass

# =========================================================
# PAGE: PETA GEOSPASIAL
# =========================================================
elif page == "Peta Geospasial":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Geospatial Tourism Analytics</div>'
        "<h1>Peta Sebaran<br>Pintu Masuk Indonesia.</h1>"
        "<p>Visualisasi pemetaan 26 pintu masuk wisatawan mancanegara di nusantara, "
        "mengintegrasikan letak geografis, segmentasi K-Means, moda transportasi, "
        "serta laju pemulihan pascapandemi.</p>"
        '<div style="margin-top:16px;">'
        '<span class="transport-pill">26 Pintu Masuk</span>'
        '<span class="transport-pill">Udara</span>'
        '<span class="transport-pill">Laut</span>'
        '<span class="transport-pill">Darat</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Memvisualisasikan kesenjangan geografis (disparitas spasial) sebaran wisatawan mancanegara di 26 pintu gerbang utama Indonesia.",
        metode="Pemetaan koordinat latitude/longitude presisi menggunakan Plotly Mapbox, diintegrasikan dengan label klaster K-Means dan proporsi volume kedatangan.",
        value="Membuktikan perlunya akselerasi program 'Beyond Bali' dan 5 Destinasi Super Prioritas (5 DPSP) agar devisa tidak tersentralisasi hanya di Bali dan Jakarta.",
    )

    geo_data = build_geospatial_dataframe(df, cluster_features_with_label)

    # Filter Controls
    st.markdown('<div class="section-title">Kontrol & Filter Peta</div>', unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        moda_options = ["Semua Moda", "Udara", "Laut", "Darat"]
        sel_moda = st.selectbox("Filter Moda Transportasi", moda_options)
    with f_col2:
        cluster_options = ["Semua Cluster", "Cluster 0 (Mega-Hub)", "Cluster 1 (Regional Gateway)"]
        sel_cluster = st.selectbox("Filter Cluster", cluster_options)
    with f_col3:
        color_scheme = st.radio(
            "Skema Pewarnaan Titik",
            ["Berdasarkan Cluster K-Means", "Berdasarkan Moda Transportasi"],
            horizontal=True,
        )

    # Apply filters
    filtered_geo = geo_data.copy()
    if sel_moda != "Semua Moda":
        filtered_geo = filtered_geo[filtered_geo["Moda"] == sel_moda]
    if sel_cluster == "Cluster 0 (Mega-Hub)":
        filtered_geo = filtered_geo[filtered_geo["Cluster_Id"] == 0]
    elif sel_cluster == "Cluster 1 (Regional Gateway)":
        filtered_geo = filtered_geo[filtered_geo["Cluster_Id"] == 1]

    # Map Rendering
    color_var = "Cluster" if color_scheme == "Berdasarkan Cluster K-Means" else "Moda"
    color_map = {
        "Cluster 0 (Mega-Hub)": "#00A86B",
        "Cluster 1 (Regional Gateway)": "#f77f00",
        "Udara": "#0077b6",
        "Laut": "#00a86b",
        "Darat": "#f77f00",
    }
    hover_cfg = {
        "Latitude": False,
        "Longitude": False,
        "Provinsi": True,
        "Moda": True,
        "Fasilitas": True,
        "Cluster": True,
        "Total Kunjungan": ":,",
        "Kunjungan 2025": ":,",
        "Recovery 2025 (%)": ":.1f",
        "Size_Display": False,
    }

    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(
            filtered_geo,
            lat="Latitude",
            lon="Longitude",
            hover_name="Pintu Masuk",
            size="Size_Display",
            color=color_var,
            color_discrete_map=color_map,
            hover_data=hover_cfg,
            size_max=38,
            zoom=4.0,
            center={"lat": -2.2, "lon": 118.0},
            map_style="open-street-map",
        )
    elif hasattr(px, "scatter_mapbox"):
        fig_map = px.scatter_mapbox(
            filtered_geo,
            lat="Latitude",
            lon="Longitude",
            hover_name="Pintu Masuk",
            size="Size_Display",
            color=color_var,
            color_discrete_map=color_map,
            hover_data=hover_cfg,
            size_max=38,
            zoom=4.0,
            center={"lat": -2.2, "lon": 118.0},
            mapbox_style="open-street-map",
        )
    else:
        fig_map = px.scatter_geo(
            filtered_geo,
            lat="Latitude",
            lon="Longitude",
            hover_name="Pintu Masuk",
            size="Size_Display",
            color=color_var,
            color_discrete_map=color_map,
        )
        fig_map.update_geos(
            scope="asia",
            center=dict(lat=-2.2, lon=118.0),
            projection_scale=3.8,
            visible=True,
        )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="DM Sans, sans-serif", color="#1F5137", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(31,81,55,0.2)",
            borderwidth=1,
        ),
    )
    style_plotly(fig_map)
    render_chart(fig_map)

    st.markdown(
        '<div class="note-box" style="margin-top:4px;margin-bottom:18px;">'
        "<b>Panduan interaktif:</b> Ukuran lingkaran merefleksikan besaran volume kunjungan. "
        "Arahkan kursor (*hover*) pada lingkaran untuk melihat profil lengkap, atau pilih pintu masuk "
        "di bagian bawah untuk inspeksi mendalam."
        "</div>",
        unsafe_allow_html=True,
    )

    # Geospatial Snapshot KPIs
    st.markdown('<div class="section-title">Distribusi Geografis</div>', unsafe_allow_html=True)
    g_cols = st.columns(4)
    with g_cols[0]:
        metric_card("Pintu aktif ditampilkan", f"{len(filtered_geo)}", f"dari total {len(geo_data)} pintu")
    with g_cols[1]:
        n_air = int((filtered_geo["Moda"] == "Udara").sum())
        metric_card("Pintu Udara (Bandara)", f"{n_air}", "gerbang penerbangan")
    with g_cols[2]:
        n_sea = int((filtered_geo["Moda"] == "Laut").sum())
        metric_card("Pintu Laut (Pelabuhan)", f"{n_sea}", "feri & kapal pesiar")
    with g_cols[3]:
        n_land = int((filtered_geo["Moda"] == "Darat").sum())
        metric_card("Pintu Darat (PLBN)", f"{n_land}", "lintas batas negara")

    # Deep-Dive Inspector
    st.markdown('<div class="section-title">Inspeksi Profil Pintu Masuk</div>', unsafe_allow_html=True)
    available_pintu = sorted(filtered_geo["Pintu Masuk"].tolist())
    if available_pintu:
        default_idx = available_pintu.index("Ngurah Rai") if "Ngurah Rai" in available_pintu else 0
        sel_pintu = st.selectbox("Pilih Pintu Masuk untuk Analisis Spesifik:", available_pintu, index=default_idx)

        p_info = geo_data[geo_data["Pintu Masuk"] == sel_pintu].iloc[0]
        p_detail = df[df["Pintu Masuk"] == sel_pintu]
        annual_p = p_detail.groupby("Tahun")["Jumlah"].sum(min_count=1).reset_index()

        d_col1, d_col2 = st.columns([1, 1.4])
        with d_col1:
            st.markdown(
                f'<div class="clay-card" style="height:100%;">'
                f'<div class="eyebrow">{p_info["Moda"]} • {p_info["Provinsi"]}</div>'
                f'<h2 style="font-family:\'Baloo 2\',sans-serif;color:#133426;margin:0 0 10px 0;">{p_info["Pintu Masuk"]}</h2>'
                f'<p style="font-size:0.9rem;color:#2b3e2e;margin-bottom:14px;"><b>Fasilitas:</b> {p_info["Fasilitas"]}</p>'
                f'<div style="margin-bottom:14px;"><span class="cluster-chip cluster-{p_info["Cluster_Id"]}">{p_info["Cluster"]}</span></div>'
                f'<hr style="border:none;border-top:1px solid rgba(31,81,55,0.15);margin:12px 0;">'
                f'<p style="margin:4px 0;font-size:0.88rem;"><b>Total Kunjungan (2016–2026):</b> {fmt_number(p_info["Total Kunjungan"])}</p>'
                f'<p style="margin:4px 0;font-size:0.88rem;"><b>Kunjungan Tahun 2025:</b> {fmt_number(p_info["Kunjungan 2025"])}</p>'
                f'<p style="margin:4px 0;font-size:0.88rem;"><b>Tingkat Pemulihan 2025:</b> <b>{p_info["Recovery 2025 (%)"]}%</b> vs 2019 baseline</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with d_col2:
            fig_p = go.Figure()
            fig_p.add_trace(
                go.Scatter(
                    x=annual_p["Tahun"],
                    y=annual_p["Jumlah"],
                    mode="lines+markers",
                    name="Kunjungan",
                    line=dict(color="#00A86B", width=3.5, shape="spline", smoothing=1.3),
                    fill="tozeroy",
                    fillcolor="rgba(0, 168, 107, 0.12)",
                    marker=dict(size=8, color="#ffffff", line=dict(color="#00A86B", width=2.5)),
                    hovertemplate="<b>Tahun %{x}</b><br>Kunjungan: <b>%{y:,.0f} Wisman</b><extra></extra>",
                )
            )
            fig_p.update_layout(
                title=dict(text=f"Tren Kunjungan Wisman — {sel_pintu}", font=dict(family="Baloo 2, sans-serif", color="#133426", size=14)),
                margin=dict(l=0, r=0, t=35, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.40)",
                height=260,
                hovermode="x unified",
                xaxis=dict(dtick=1),
                showlegend=False,
            )
            style_plotly(fig_p)
            render_chart(fig_p)

    # Spatial Insights
    st.markdown('<div class="section-title">Wawasan Spasial & Geopolitik Pariwisata</div>', unsafe_allow_html=True)
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown(
            '<div class="insight-card">'
            ''
            '<div class="insight-text">'
            '<b>Polarisasi Mega-Hub:</b><br>'
            'Lebih dari 65% total kunjungan wisman udara terpusat pada koridor barat-tengah: '
            'Bandara Ngurah Rai (Bali) dan Soekarno-Hatta (Jakarta).'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with s_col2:
        st.markdown(
            '<div class="insight-card">'
            ''
            '<div class="insight-text">'
            '<b>Koridor Maritim Kepri:</b><br>'
            'Pelabuhan Batam, Tanjung Uban, dan Tanjung Pinang membentuk segitiga konektivitas laut '
            'terpadat yang menangkap wisatawan *cross-border* Singapura dan Malaysia.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with s_col3:
        st.markdown(
            '<div class="insight-card">'
            ''
            '<div class="insight-text">'
            '<b>Pemulihan Lintas Batas Darat:</b><br>'
            'PLBN di Kalimantan Barat (Aruk, Entikong, Badau) dan NTT (Atambua) mencatatkan recovery '
            'di atas 100% didorong oleh pemulihan mobilitas darat pascapandemi.'
            '</div></div>',
            unsafe_allow_html=True,
        )

# =========================================================
# PAGE: CLUSTER EXPLORER
# =========================================================
elif page == "Cluster Explorer":
    st.markdown(
        '<div class="clay-card">'
        '<div class="eyebrow">Model Explorer</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#205438;margin:0;">Cluster Explorer</h1>'
        '<p style="color:#2b3e2e;font-weight:500;margin-top:8px;">'
        "Jelajahi pintu masuk berdasarkan profil musiman, volume, dan recovery.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Mengidentifikasi segmentasi karakteristik perilaku dan pola musiman pintu masuk internasional secara objektif tanpa bias apriori.",
        metode="Unsupervised Machine Learning K-Means (k=2, Silhouette 0.630) berbasis 14 fitur rekayasa (12 indeks musiman, volume logaritmik, dan laju pemulihan 2025).",
        value="Membedakan perlakuan antara Mega-Hubs (Cluster 0) untuk manajemen lonjakan kapasitas, serta Regional Gateways (Cluster 1) untuk target ekspansi promosi dan diversifikasi rute baru.",
    )

    if not clustering_ok:
        st.warning(
            "Model clustering tidak tersedia. "
            "Pastikan `kmeans_clustering_final.pkl` dan `scaler_clustering.pkl` "
            "tersedia di folder `model/`."
        )
        st.stop()

    profile = cluster_features_with_label.copy()
    profile.index.name = None
    profile = profile.reset_index().rename(columns={"index": "Pintu Masuk"})

    search = st.text_input("Cari pintu masuk", placeholder="Contoh: Ngurah Rai")
    options = ["Semua"] + sorted(profile["Pintu Masuk"].tolist())
    selected = st.selectbox("Pilih pintu masuk", options)

    if selected != "Semua":
        selected_rows = profile[profile["Pintu Masuk"] == selected]
    elif search:
        selected_rows = profile[profile["Pintu Masuk"].str.contains(search, case=False, na=False)]
    else:
        selected_rows = profile.copy()

    c0 = profile[profile["Cluster"] == 0]["Pintu Masuk"].tolist()
    c1 = profile[profile["Cluster"] == 1]["Pintu Masuk"].tolist()
    st.markdown('<div class="section-title">Struktur cluster</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="clay-card"><span class="cluster-chip cluster-0">CLUSTER 0</span>'
            f'<h3 style="font-family:\'Baloo 2\',sans-serif;margin-bottom:4px;">Profil khusus</h3>'
            f'<p style="color:#2b3e2e;font-weight:500;">{len(c0)} pintu masuk</p>'
            f'<div class="cluster-members">{" • ".join(c0)}</div>'
            f'<p style="font-size:.82rem;margin-top:10px;color:#2d4d33;">'
            f"Pola musiman sangat khas dengan karakteristik volume dan recovery "
            f"yang berbeda dari mayoritas pintu masuk.</p></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="clay-card"><span class="cluster-chip cluster-1">CLUSTER 1</span>'
            f'<h3 style="font-family:\'Baloo 2\',sans-serif;margin-bottom:4px;">Profil umum</h3>'
            f'<p style="color:#2b3e2e;font-weight:500;">{len(c1)} pintu masuk</p>'
            f'<div class="cluster-members">{" • ".join(c1)}</div>'
            f'<p style="font-size:.82rem;margin-top:10px;color:#2d4d33;">'
            f"Kelompok mayoritas dengan pola musiman yang lebih umum "
            f"dan volume relatif lebih tinggi.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Detail profil</div>', unsafe_allow_html=True)

    if len(selected_rows) == 1:
        row = selected_rows.iloc[0]
        selected_name = row["Pintu Masuk"]
        selected_cluster = int(row["Cluster"])
        chip_cls = "cluster-0" if selected_cluster == 0 else "cluster-1"
        st.markdown(
            f'<div class="clay-card">'
            f'<span class="cluster-chip {chip_cls}">CLUSTER {selected_cluster}</span>'
            f'<h2 style="font-family:\'Baloo 2\',sans-serif;color:#205438;margin:10px 0 0;">'
            f"{selected_name}</h2></div>",
            unsafe_allow_html=True,
        )
        MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        seasonal_values = [float(row[f"{m}_Index"]) for m in MONTH_NAMES]
        peak_month_label = MONTH_NAMES[int(np.argmax(seasonal_values))]

        cols3 = st.columns(3)
        with cols3[0]:
            metric_card("Log mean volume", f"{row['Log_Mean_Volume']:.2f}", "feature clustering")
        with cols3[1]:
            metric_card("Recovery 2025", f"{row['Recovery_2025']:.1f}%", "dibanding baseline 2019")
        with cols3[2]:
            metric_card("Indeks musiman tertinggi", peak_month_label, "profil 2016–2019")

        # Radar chart — all 12 months explicitly displayed
        fig_season = go.Figure()

        # Netral baseline (1.0 = rata-rata normal)
        fig_season.add_trace(go.Scatterpolar(
            r=[1.0] * 13,
            theta=MONTH_NAMES + [MONTH_NAMES[0]],
            mode="lines",
            name="Baseline Netral (1.00)",
            line=dict(color="rgba(31, 81, 55, 0.35)", width=1.5, dash="dash"),
            hoverinfo="skip",
        ))

        fig_season.add_trace(go.Scatterpolar(
            r=seasonal_values + [seasonal_values[0]],
            theta=MONTH_NAMES + [MONTH_NAMES[0]],
            mode="lines+markers",
            fill="toself",
            name=selected_name,
            line=dict(color="#00A86B", width=3),
            fillcolor="rgba(0, 168, 107, 0.22)",
            marker=dict(size=8, color="#ffffff", line=dict(color="#00A86B", width=2.5)),
            hovertemplate="<b>%{theta}</b><br>Indeks Musiman: <b>%{r:.2f}</b><extra></extra>",
        ))
        fig_season.update_layout(
            polar=dict(
                bgcolor="rgba(255, 255, 255, 0.30)",
                radialaxis=dict(
                    visible=True,
                    tickfont=dict(color="#1e5c3a", size=10, family="DM Sans"),
                    gridcolor="rgba(31, 81, 55, 0.12)",
                    linecolor="rgba(31, 81, 55, 0.25)",
                ),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=MONTH_NAMES,
                    ticktext=MONTH_NAMES,
                    tickfont=dict(color="#133426", size=12, family="DM Sans"),
                    gridcolor="rgba(31, 81, 55, 0.12)",
                    linecolor="rgba(31, 81, 55, 0.25)",
                    direction="clockwise",
                    rotation=90,
                ),
            ),
            margin=dict(l=50, r=50, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            title=dict(
                text=f"Profil Musiman 12 Bulan — {selected_name}",
                font=dict(family="Baloo 2, sans-serif", color="#133426", size=15),
            ),
        )
        style_plotly(fig_season)
        render_chart(fig_season)
        interp = (
            "Pintu masuk ini memiliki <b>pola musiman yang sangat khas</b> "
            "dan berbeda dari mayoritas pintu masuk lainnya. "
            "Karakteristik volume dan recovery-nya membedakannya secara signifikan."
        ) if selected_cluster == 0 else (
            "Pintu masuk ini memiliki <b>pola musiman yang relatif umum</b>, "
            "serupa dengan mayoritas pintu masuk Indonesia. "
            "Volume kunjungan cenderung mengikuti tren nasional."
        )
        st.markdown(
            f'<div class="insight-card">'
            f''
            f'<div class="insight-text">{interp}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(
            selected_rows[["Pintu Masuk","Cluster","Log_Mean_Volume","Recovery_2025"]]
            .sort_values(["Cluster","Pintu Masuk"]),
            width="stretch", hide_index=True,
        )


# =========================================================
# PAGE: FORECASTING
# =========================================================
elif page == "Forecasting":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Forecasting Intelligence • Granular & National</div>'
        "<h1>Melihat Arah Perjalanan<br>Wisata Indonesia.</h1>"
        "<p>Prediksi dan peramalan jumlah kunjungan wisatawan mancanegara hingga 12 bulan "
        "ke depan (Agustus 2026 – Juli 2027) menggunakan arsitektur SARIMA musiman. "
        "Dapat dieksplorasi di tingkat nasional, moda transportasi, maupun pintu masuk spesifik.</p>"
        '<div style="margin-top:16px;">'
        '<span class="transport-pill">12-Month Ahead Forecast</span>'
        '<span class="transport-pill">Udara</span>'
        '<span class="transport-pill">Darat</span>'
        '<span class="transport-pill">Laut</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Memproyeksikan volume kunjungan wisman 12 bulan ke depan (Agustus 2026 – Juli 2027) untuk perencanaan strategis kapasitas dan promosi.",
        metode="Model Time-Series SARIMA(0,1,1)(0,1,1,12) terbaik dengan akurasi MAPE 6.36%, mengungguli Holt-Winters, Prophet, Random Forest, dan XGBoost.",
        value="Menyediakan peramalan terpercaya dengan 95% Confidence Interval serta kapabilitas granular (per moda & per pintu masuk) untuk mitigasi risiko ketidakpastian.",
    )

    # 1. Scope Selector Controls
    st.markdown('<div class="section-title">Konfigurasi Target Forecast</div>', unsafe_allow_html=True)
    fc_ctrl1, fc_ctrl2 = st.columns([1, 1.2])
    with fc_ctrl1:
        fc_scope = st.radio(
            "Tingkat Granularitas Peramalan",
            ["Nasional (Total Indonesia)", "Per Moda Transportasi", "Per Pintu Masuk Spesifik"],
            horizontal=False,
        )

    selected_target_type = "nasional"
    selected_target_name = "Nasional"
    target_badge = "Nasional (Total Indonesia)"

    with fc_ctrl2:
        if fc_scope == "Per Moda Transportasi":
            selected_target_type = "moda"
            selected_target_name = st.selectbox("Pilih Moda Transportasi:", ["Udara", "Laut", "Darat"])
            target_badge = f"Moda {selected_target_name}"
            st.caption(f"Memprediksi proyeksi pergerakan wisatawan untuk seluruh pintu {selected_target_name}.")
        elif fc_scope == "Per Pintu Masuk Spesifik":
            selected_target_type = "pintu_masuk"
            pintu_options = sorted(df[df["Jenis Data"] == "Pintu Masuk"]["Pintu Masuk"].unique().tolist())
            def_idx = pintu_options.index("Ngurah Rai") if "Ngurah Rai" in pintu_options else 0
            selected_target_name = st.selectbox("Pilih Pintu Masuk:", pintu_options, index=def_idx)
            target_badge = f"Pintu Masuk: {selected_target_name}"
            st.caption(f"Menjalankan model SARIMA khusus untuk profil deret waktu {selected_target_name}.")
        else:
            st.info("Menampilkan proyeksi deret waktu agregat total seluruh pintu masuk Indonesia.")

    # 2. Run / Retrieve Forecast
    with st.spinner("Menghitung proyeksi model SARIMA..."):
        if selected_target_type == "nasional" and sarima_ok and sarima_artifact is not None:
            try:
                final_fit = sarima_artifact["model"]
                model_name_str = sarima_artifact.get("model_name", "SARIMA(0,1,1)(0,1,1,12)")
                forecast_result = final_fit.get_forecast(steps=12)
                future_forecast = forecast_result.predicted_mean.clip(lower=0)
                conf_int = forecast_result.conf_int().clip(lower=0)
                lower_col, upper_col = conf_int.columns[0], conf_int.columns[1]
                series_known = build_monthly_series(df)
            except Exception:
                series_known, future_forecast, conf_int = run_granular_forecast(df, selected_target_type, selected_target_name)
                lower_col, upper_col = conf_int.columns[0], conf_int.columns[1]
                model_name_str = "SARIMA(0,1,1)(0,1,1,12)"
        else:
            series_known, future_forecast, conf_int = run_granular_forecast(df, selected_target_type, selected_target_name)
            lower_col, upper_col = conf_int.columns[0], conf_int.columns[1]
            model_name_str = "SARIMA(0,1,1)(0,1,1,12)"

    # Compute Growth vs 2025
    y2025_vals = series_known[series_known.index.year == 2025]
    avg_2025 = float(y2025_vals.mean()) if not y2025_vals.empty else 1.0
    avg_fc = float(future_forecast.mean())
    growth_pct = ((avg_fc - avg_2025) / avg_2025) * 100 if avg_2025 > 0 else 0.0
    max_fc = float(future_forecast.max())
    min_fc = float(future_forecast.min())
    total_12m = float(future_forecast.sum())
    peak_m_fc = future_forecast.idxmax()
    low_m_fc = future_forecast.idxmin()

    # 3. Overview cards
    st.markdown('<div class="section-title">Ringkasan Model & Target</div>', unsafe_allow_html=True)
    ts_start = series_known.index[0]
    ts_end = series_known.index[-1]
    ov_items = [
        ("Target Forecast", target_badge),
        ("Model Time-Series", "SARIMA(0,1,1)(0,1,1,12)"),
        ("Horizon Prediksi", "12 Bulan"),
        ("Rentang Waktu", "Agu 2026 – Jul 2027"),
        ("Data Historis", f"{len(series_known)} bulan ({fmt_date_id(ts_start)} – {fmt_date_id(ts_end)})"),
    ]
    ov_cols = st.columns(len(ov_items))
    for col, (lbl, val) in zip(ov_cols, ov_items):
        with col:
            st.markdown(
                f'<div class="overview-card">'
                f'<div class="ov-label">{lbl}</div>'
                f'<div class="ov-value">{val}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # 4. Key Forecast KPI Cards
    st.markdown('<div class="section-title">Key Forecast Metrics</div>', unsafe_allow_html=True)
    kpi5 = st.columns(5)
    with kpi5[0]:
        metric_card("Rata-rata Prediksi", fmt_number(avg_fc), "per bulan")
    with kpi5[1]:
        metric_card("Prediksi Tertinggi", fmt_number(max_fc), fmt_date_id(peak_m_fc))
    with kpi5[2]:
        metric_card("Prediksi Terendah", fmt_number(min_fc), fmt_date_id(low_m_fc))
    with kpi5[3]:
        metric_card("Estimasi Total 1 Tahun", fmt_number(total_12m), "akumulasi 12 bulan")
    with kpi5[4]:
        metric_card("Pertumbuhan vs 2025", f"{growth_pct:+.1f}%", "terhadap rata-rata 2025")

    # 5. Main Forecasting Chart
    st.markdown(f'<div class="section-title">Grafik Deret Waktu & Forecast — {selected_target_name}</div>', unsafe_allow_html=True)
    fig_fc = go.Figure()

    # Actual historical line
    fig_fc.add_trace(go.Scatter(
        x=series_known.index,
        y=series_known.values,
        name="Data Aktual (2016–Jul 2026)",
        line=dict(color="#00A86B", width=3, shape="spline", smoothing=0.8),
        mode="lines",
        hovertemplate="<b>%{x|%B %Y}</b><br>Aktual: <b>%{y:,.0f} Wisman</b><extra></extra>",
    ))

    # 95% Confidence Interval Ribbon
    ci_x = list(future_forecast.index) + list(future_forecast.index[::-1])
    ci_y = list(conf_int[upper_col]) + list(conf_int[lower_col][::-1])
    fig_fc.add_trace(go.Scatter(
        x=ci_x,
        y=ci_y,
        fill="toself",
        fillcolor="rgba(0, 119, 182, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval (CI)",
        showlegend=True,
        hoverinfo="skip",
    ))

    # SARIMA Forecast Line with Jewel Markers
    fig_fc.add_trace(go.Scatter(
        x=future_forecast.index,
        y=future_forecast.values,
        name="Forecast SARIMA (Agu 2026–Jul 2027)",
        line=dict(color="#f77f00", width=3.5, dash="dash", shape="spline", smoothing=1.0),
        mode="lines+markers",
        marker=dict(size=8, color="#ffffff", line=dict(color="#f77f00", width=2.5)),
        hovertemplate="<b>%{x|%B %Y}</b><br>Forecast: <b>%{y:,.0f} Wisman</b><extra></extra>",
    ))

    # Boundary V-Line
    boundary_ts = series_known.index[-1]
    fig_fc.add_vline(
        x=boundary_ts.value / 1e6,
        line_dash="dot",
        line_color="#1f5137",
        line_width=2,
        annotation_text="Mulai Forecast (Agu 2026)",
        annotation_position="top left",
        annotation_font=dict(color="#133426", size=11, family="DM Sans"),
    )

    # Milestone Annotation for Peak Forecast
    fig_fc.add_annotation(
        x=peak_m_fc,
        y=max_fc,
        text=f"<b>Puncak Forecast</b><br>{fmt_number(max_fc)}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f77f00",
        arrowsize=1,
        arrowwidth=1.5,
        ax=0,
        ay=-45,
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#f77f00",
        borderwidth=1.5,
        borderpad=5,
        font=dict(size=11, color="#7f4f00", family="DM Sans"),
    )

    fig_fc.update_layout(
        title=dict(
            text=f"Proyeksi Deret Waktu — {selected_target_name}",
            font=dict(family="Baloo 2, sans-serif", color="#133426", size=15),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.40)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Tanggal"),
        yaxis=dict(title="Jumlah Wisatawan Mancanegara", ticksuffix=" "),
        height=420,
    )
    style_plotly(fig_fc)
    render_chart(fig_fc)

    st.markdown(
        '<div class="note-box" style="margin-bottom:16px;">'
        f"<b>Catatan Data:</b> Deret waktu aktual <b>{selected_target_name}</b> tersedia hingga Juli 2026. "
        "Periode Agustus 2026 – Juli 2027 murni merupakan hasil proyeksi model SARIMA dengan interval kepercayaan 95%."
        "</div>",
        unsafe_allow_html=True,
    )

    # 6. Context-Aware Insight Card
    st.markdown('<div class="section-title">Wawasan Dinamis Segmen Terpilih</div>', unsafe_allow_html=True)
    if selected_target_type == "nasional":
        insight_msg = (
            "<b>Outlook Nasional:</b> Proyeksi menunjukkan tren pariwisata nasional yang konsisten bertumbuh "
            f"dengan estimasi akumulasi mencapai <b>{fmt_number(total_12m)}</b> wisman pada periode Agu 2026 – Jul 2027. "
            f"Bulan puncak diproyeksikan pada <b>{fmt_date_id(peak_m_fc)}</b> seiring dengan musim liburan global."
        )
    elif selected_target_type == "moda":
        insight_msg = (
            f"<b>Dinamika Moda {selected_target_name}:</b> Kunjungan via moda {selected_target_name} diproyeksikan "
            f"mencapai rata-rata <b>{fmt_number(avg_fc)}</b> wisman/bulan dengan pertumbuhan <b>{growth_pct:+.1f}%</b> "
            f"dibandingkan rata-rata tahun 2025. Puncak kunjungan diproyeksikan terjadi pada <b>{fmt_date_id(peak_m_fc)}</b>."
        )
    elif selected_target_name == "Ngurah Rai":
        insight_msg = (
            "<b>Pola Mega-Hub Bali (Ngurah Rai):</b> Menunjukkan karakter musiman yang sangat kuat dengan lonjakan tajam "
            f"pada pertengahan tahun (Juli-Agustus). Rata-rata per bulan diproyeksikan mencapai <b>{fmt_number(avg_fc)}</b> wisman."
        )
    elif selected_target_name == "Batam":
        insight_msg = (
            "<b>Pola Maritim Cross-Border (Batam):</b> Menunjukkan aliran kunjungan yang sangat stabil dengan fluktuasi akhir pekan "
            f"dan libur regional (Singapura/Malaysia). Proyeksi bulanan berada di kisaran <b>{fmt_number(avg_fc)}</b> wisman."
        )
    elif selected_target_name == "Soekarno-Hatta":
        insight_msg = (
            "<b>Gerbang Bisnis & Penerbangan (Soekarno-Hatta):</b> Karakteristik kunjungan didorong oleh penerbangan jarak jauh "
            f"(long-haul) dan transit bisnis, dengan rata-rata proyeksi mencapai <b>{fmt_number(avg_fc)}</b> wisman per bulan."
        )
    else:
        insight_msg = (
            f"<b>Profil Gateway Regional ({selected_target_name}):</b> Model memproyeksikan rata-rata bulanan sebesar "
            f"<b>{fmt_number(avg_fc)}</b> wisman ({growth_pct:+.1f}% vs 2025). Puncak musiman diprediksi terjadi pada <b>{fmt_date_id(peak_m_fc)}</b>."
        )

    st.markdown(
        f'<div class="insight-card">'
        f''
        f'<div class="insight-text">{insight_msg}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # 7. Forecast Table & Download
    st.markdown(f'<div class="section-title">Tabel Hasil Forecast — {selected_target_name}</div>', unsafe_allow_html=True)
    forecast_df = pd.DataFrame({
        "Tanggal": [fmt_date_id(ts) for ts in future_forecast.index],
        "Forecast (Wisman)": [fmt_number_id(v) for v in future_forecast.values],
        "Lower Bound (95%)": [fmt_number_id(v) for v in conf_int[lower_col].values],
        "Upper Bound (95%)": [fmt_number_id(v) for v in conf_int[upper_col].values],
    })
    st.dataframe(forecast_df, width="stretch", hide_index=True)

    csv_data = pd.DataFrame({
        "Tanggal": future_forecast.index.strftime("%Y-%m-%d"),
        "Bulan": [fmt_date_id(ts) for ts in future_forecast.index],
        "Forecast": future_forecast.values.round(2),
        "Lower_Bound_95": conf_int[lower_col].values.round(2),
        "Upper_Bound_95": conf_int[upper_col].values.round(2),
    }).to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download Data Forecast ({selected_target_name}) dalam CSV",
        data=csv_data,
        file_name=f"forecast_sarima_{selected_target_name.lower().replace(' ', '_')}_2026_2027.csv",
        mime="text/csv",
    )

    # 8. Model comparison table
    st.markdown('<div class="section-title">Perbandingan Model Benchmark (Nasional)</div>', unsafe_allow_html=True)
    comparison_data = [
        ("SARIMA",        79_871.19,  88_965.73,  6.36,  6.42,  True),
        ("XGBoost",       85_525.98,  128_949.51, 7.58,  6.98,  False),
        ("Random Forest", 106_029.15, 125_554.62, 8.71,  8.47,  False),
        ("Holt-Winters",  175_262.03, 193_497.40, 13.18, 14.18, False),
        ("Prophet",       615_747.89, 626_272.32, 47.69, 62.81, False),
    ]
    rows_html = ""
    for name, mae, rmse, mape, smape, is_best in comparison_data:
        badge = '<span class="best-badge">BEST</span>' if is_best else ""
        rows_html += (
            f"<tr>"
            f'<td style="padding:10px 14px;font-weight:700;">{name} {badge}</td>'
            f'<td style="padding:10px 14px;">{mae:,.2f}</td>'
            f'<td style="padding:10px 14px;">{rmse:,.2f}</td>'
            f'<td style="padding:10px 14px;">{mape:.2f}%</td>'
            f'<td style="padding:10px 14px;">{smape:.2f}%</td>'
            f"</tr>"
        )
    st.markdown(
        '<div class="clay-card" style="overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;color:#1f5137;font-size:.9rem;">'
        "<thead><tr>"
        '<th style="padding:10px 14px;text-align:left;font-family:\'Baloo 2\',sans-serif;">Model</th>'
        '<th style="padding:10px 14px;text-align:left;">MAE</th>'
        '<th style="padding:10px 14px;text-align:left;">RMSE</th>'
        '<th style="padding:10px 14px;text-align:left;">MAPE</th>'
        '<th style="padding:10px 14px;text-align:left;">sMAPE</th>'
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )
    # 6. MAPE bar chart
    st.markdown('<div class="section-title">Perbandingan MAPE</div>', unsafe_allow_html=True)
    comp_names = [r[0] for r in comparison_data]
    comp_mapes = [r[3] for r in comparison_data]
    comp_best  = [r[5] for r in comparison_data]
    # Strategic color grading: Best = Emerald, Excellent = Azure, Good = Cyan, High error = Amber/Red
    mape_colors = ["#00A86B" if b else ("#0077b6" if m < 8 else ("#48cae4" if m < 10 else ("#f77f00" if m < 20 else "#d62828"))) for b, m in zip(comp_best, comp_mapes)]
    fig_mape = go.Figure(go.Bar(
        x=comp_names,
        y=comp_mapes,
        text=[f"<b>{v:.2f}%</b>" for v in comp_mapes],
        textposition="outside",
        marker_color=mape_colors,
        hovertemplate="<b>Model: %{x}</b><br>MAPE: <b>%{y:.2f}%</b><extra></extra>",
    ))
    fig_mape.add_annotation(
        x="SARIMA",
        y=6.36,
        text="<b>Model Terpilih (6.36%)</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00A86B",
        arrowsize=1,
        arrowwidth=1.5,
        ax=0,
        ay=-45,
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#00A86B",
        borderwidth=1.5,
        borderpad=5,
        font=dict(size=11, color="#133426", family="DM Sans"),
    )
    fig_mape.update_layout(
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.40)",
        yaxis=dict(title="MAPE (%)", ticksuffix="%"),
        xaxis=dict(title=""),
        title=dict(
            text="Evaluasi Akurasi (MAPE) Antar Model — Semakin Rendah Semakin Presisi",
            font=dict(family="Baloo 2, sans-serif", color="#133426", size=15),
        ),
        height=340,
    )
    style_plotly(fig_mape)
    render_chart(fig_mape)

    # 7. Methodology
    st.markdown('<div class="section-title">Metodologi Forecasting</div>', unsafe_allow_html=True)
    steps = [
        ("", "Data Historis", "Jan 2016 – Jul 2026"),
        ("", "Agregasi Bulanan", "MS frequency"),
        ("", "Train / Test Split", "Train: s.d. 2024 | Test: 2025"),
        ("", "5 Algoritma", "HW · SARIMA · Prophet · RF · XGB"),
        ("", "Evaluasi", "MAE · RMSE · MAPE · sMAPE"),
        ("", "Model Terbaik", "SARIMA"),
        ("", "Forecast Final", "12 bulan ke depan"),
    ]
    step_cols = st.columns(len(steps))
    for col, (icon, title, sub) in zip(step_cols, steps):
        with col:
            st.markdown(
                f'<div class="method-step">'
                f'<div style="font-size:1.4rem;margin-bottom:6px;">{icon}</div>'
                f'<div style="font-weight:800;font-size:.84rem;">{title}</div>'
                f'<div style="font-size:.75rem;font-weight:500;margin-top:3px;">{sub}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    # 8. Kenapa SARIMA?
    st.markdown('<div class="section-title">Kenapa SARIMA?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="clay-card">'
        '<h3 style="font-family:\'Baloo 2\',sans-serif;color:#133426;margin-top:0;">'
        "Alasan Pemilihan Model SARIMA</h3>"
        '<ul style="color:#1f5137;font-size:.92rem;line-height:1.8;padding-left:20px;">'
        "<li><b>Menangkap pola musiman dan non-musiman:</b> SARIMA dirancang khusus "
        "untuk data time-series dengan siklus berulang. Periode musiman 12 (bulanan) "
        "efektif menangkap fluktuasi tahunan wisatawan.</li>"
        "<li><b>Dipilih berdasarkan metrik objektif:</b> SARIMA menghasilkan "
        "MAPE terendah (6.36%) di antara 5 model yang diuji.</li>"
        "<li><b>Disertai confidence interval:</b> Setiap prediksi dilengkapi "
        "batas bawah dan atas (CI 95%), memberikan gambaran ketidakpastian yang jujur.</li>"
        "<li><b>Catatan penting:</b> Hasil ini adalah <i>estimasi model</i>, "
        "bukan jaminan nilai masa depan yang pasti. Gunakan sebagai panduan perencanaan "
        "(<i>forecast</i>, bukan <i>certainty</i>).</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    # 9. Data note
    st.markdown(
        '<div class="note-box">'
        "<b>Catatan penting tentang data 2026:</b><br>"
        "Dataset bersih hanya memuat data aktual hingga <b>Juli 2026</b>. "
        "Periode <b>Agustus 2026 – Juli 2027</b> sepenuhnya merupakan "
        "<b>hasil prediksi SARIMA</b> dan tidak boleh diperlakukan sebagai "
        "data historis yang terukur. Prediksi dibuat berdasarkan pola dari "
        "127 observasi bulanan (Januari 2016 – Juli 2026)."
        "</div>",
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE: SIMULATOR KEBIJAKAN (WHAT-IF SCENARIO SIMULATOR)
# =========================================================
elif page == "Simulator Kebijakan":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Decision Support System • Strategic Planning</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#133426;margin:0;">'
        "Simulator Kebijakan Pariwisata</h1>"
        '<p style="color:#2b3e2e;font-weight:500;margin-top:12px;max-width:70%;">'
        "Uji skenario 'What-If' masa depan (Agustus 2026 – Juli 2027) berbasis model SARIMA "
        "untuk mengestimasi dampak intervensi kebijakan penerbangan, visa, biaya aviasi, "
        "serta redistribusi wisatawan ke destinasi regional (Beyond Bali).</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Mentransformasikan prediksi pasif menjadi Decision Support System (DSS) interaktif untuk menguji dampak berbagai skenario kebijakan publik.",
        metode="Simulasi elastisitas multivariat (kapasitas penerbangan, kemudahan visa, biaya tiket aviasi) dipadukan dengan pemodelan devisa BPS & redistribusi Beyond Bali.",
        value="Menghitung estimasi devisa tambahan negara (Juta USD & Triliun Rupiah) serta mengukur keberhasilan pengalihan kunjungan ke kawasan Indonesia Timur.",
    )

    if not sarima_ok or sarima_artifact is None:
        st.warning("Model SARIMA belum tersedia. Pastikan `model/tourism_sarima_model.pkl` ada.")
    else:
        sarima_res = sarima_artifact["model"]
        base_fc = sarima_res.get_forecast(steps=12)
        base_mean = base_fc.predicted_mean.clip(lower=0)
        base_ci = base_fc.conf_int().clip(lower=0)

        # ---------------------------------------------------------
        # PRESET SKENARIO CEPAT
        # ---------------------------------------------------------
        st.markdown(
            '<div class="section-title">Konfigurasi Tuas Kebijakan</div>'
            '<div class="section-subtitle">Pilih skenario siap pakai atau atur tuas parameter secara mandiri di bawah ini.</div>',
            unsafe_allow_html=True,
        )

        preset_options = [
            "Skenario Standar (Baseline SARIMA / Status Quo)",
            "Akselerasi Wisata Nasional (Visit Indonesia Golden Year)",
            "Krisis Aviasi Global & Lonjakan Biaya Tiket",
            "Desentralisasi Wisata 'Beyond Bali' (5 DPSP Priority)",
            "Kustom / Atur Mandiri",
        ]
        chosen_preset = st.selectbox(
            "Pilih Preset Skenario Strategis:",
            preset_options,
            index=1,  # Default to Akselerasi Wisata
            help="Memuat konfigurasi tuas kebijakan yang dirancang khusus untuk analisis skenario",
        )

        # Inisialisasi nilai default berdasarkan preset
        if "Standar" in chosen_preset:
            def_flight = 0.0
            def_visa = 0.0
            def_cost = 0.0
            def_beyond = 0.0
        elif "Akselerasi" in chosen_preset:
            def_flight = 20.0
            def_visa = 15.0
            def_cost = -5.0
            def_beyond = 10.0
        elif "Krisis" in chosen_preset:
            def_flight = -15.0
            def_visa = 0.0
            def_cost = 20.0
            def_beyond = 0.0
        elif "Desentralisasi" in chosen_preset:
            def_flight = 10.0
            def_visa = 10.0
            def_cost = 0.0
            def_beyond = 25.0
        else:
            def_flight = 10.0
            def_visa = 5.0
            def_cost = 0.0
            def_beyond = 5.0

        # ---------------------------------------------------------
        # KONTROL SLIDER KEBIJAKAN
        # ---------------------------------------------------------
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(
                '<div class="policy-card">'
                '<h4>Kapasitas & Aksesibilitas Internasional</h4>'
                '<p style="font-size:0.82rem;color:#55755e;margin-bottom:12px;">'
                'Ekspansi slot penerbangan langsung dan deregulasi perizinan masuk wisman.</p>',
                unsafe_allow_html=True,
            )
            flight_slider = st.slider(
                "Penambahan Kapasitas Kursi Penerbangan (%):",
                min_value=-30.0,
                max_value=30.0,
                value=float(def_flight),
                step=1.0,
                format="%+.0f%%",
                help="Penambahan slot penerbangan internasional ke bandara di Indonesia. Efek elastisitas supply = 0.65.",
            )
            visa_slider = st.slider(
                "Peluasan Bebas Visa Kunjungan / Kemudahan VoA (%):",
                min_value=0.0,
                max_value=25.0,
                value=float(max(0.0, def_visa)),
                step=1.0,
                format="+%.0f%%",
                help="Pemberian bebas visa baru atau digitalisasi Visa on Arrival bagi negara-negara pasar potensial.",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_s2:
            st.markdown(
                '<div class="policy-card">'
                '<h4>Beban Biaya & Pemerataan Wilayah</h4>'
                '<p style="font-size:0.82rem;color:#55755e;margin-bottom:12px;">'
                'Guncangan inflasi biaya aviasi dan redistribusi ke 5 Destinasi Super Prioritas.</p>',
                unsafe_allow_html=True,
            )
            cost_slider = st.slider(
                "Guncangan Biaya Aviasi & Harga Tiket Internasional (%):",
                min_value=-20.0,
                max_value=20.0,
                value=float(def_cost),
                step=1.0,
                format="%+.0f%%",
                help="Kenaikan fuel surcharge atau harga aviasi menekan minat bepergian (elastisitas harga = -0.45).",
            )
            beyond_slider = st.slider(
                "Target Kampanye 'Beyond Bali' / Destinasi Regional (%):",
                min_value=0.0,
                max_value=30.0,
                value=float(def_beyond),
                step=1.0,
                format="+%.0f%%",
                help="Persentase promosi penerbangan langsung yang diarahkan ke Cluster 1 (Lombok, Kualanamu, Yogyakarta, Manado, dll.).",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Parameter Asumsi Finansial (Devisa Pariwisata)", expanded=False):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                spend_val = st.slider(
                    "Rata-rata Pengeluaran Wisman per Kunjungan (USD):",
                    min_value=900,
                    max_value=2200,
                    value=1350,
                    step=50,
                    help="Berdasarkan Passenger Exit Survey (PES) BPS & Kemenparekraf (rata-rata pra & pasca pandemi).",
                )
            with col_f2:
                kurs_val = st.number_input(
                    "Asumsi Kurs Acuan (IDR / USD):",
                    min_value=14000,
                    max_value=18000,
                    value=16000,
                    step=100,
                )

        # ---------------------------------------------------------
        # KALKULASI HASIL SIMULASI
        # ---------------------------------------------------------
        sim = compute_policy_simulation(
            base_forecast_mean=base_mean,
            base_forecast_ci=base_ci,
            flight_cap_pct=flight_slider,
            visa_ease_pct=visa_slider,
            aviation_cost_pct=cost_slider,
            beyond_bali_pct=beyond_slider,
            spend_per_tourist_usd=float(spend_val),
            usd_idr_rate=float(kurs_val),
        )

        df_sim = sim["df_sim"]

        # ---------------------------------------------------------
        # RINGKASAN KPI DAMPAK KEBIJAKAN
        # ---------------------------------------------------------
        st.markdown(
            '<div class="section-title">Proyeksi Dampak Makro (12 Bulan Kedepan)</div>'
            '<div class="section-subtitle">Estimasi perbandingan antara Baseline SARIMA vs Skenario Kebijakan (Agustus 2026 – Juli 2027)</div>',
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            delta_w_sign = "+" if sim["delta_wisman"] >= 0 else ""
            st.markdown(
                f'<div class="clay-card kpi">'
                f'<div class="label">TOTAL WISMAN (12 BULAN)</div>'
                f'<div class="value">{sim["total_sim_wisman"]/1e6:.2f}M</div>'
                f'<div class="detail">{delta_w_sign}{fmt_number_id(sim["delta_wisman"])} ({sim["pct_growth_effective"]:+.1f}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="clay-card kpi">'
                f'<div class="label">ESTIMASI TOTAL DEVISA</div>'
                f'<div class="value">${sim["devisa_sim_usd"]/1e3:.2f}B</div>'
                f'<div class="detail">Setara Rp {sim["devisa_sim_usd"]*kurs_val/1e6:.1f} Triliun</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k3:
            d_sign = "+" if sim["delta_devisa_usd"] >= 0 else ""
            badge_class = "tag-positive" if sim["delta_devisa_usd"] >= 0 else "tag-negative"
            st.markdown(
                f'<div class="clay-card kpi">'
                f'<div class="label">NET TAMBAHAN DEVISA</div>'
                f'<div class="value">{d_sign}${sim["delta_devisa_usd"]:.1f}M</div>'
                f'<div class="detail"><span class="tag-badge {badge_class}">{d_sign}Rp {sim["delta_devisa_idr"]:.2f} Triliun</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="clay-card kpi">'
                f'<div class="label">DISTRIBUSI BEYOND BALI</div>'
                f'<div class="value">{sim["c1_sim_total"]/1e6:.2f}M</div>'
                f'<div class="detail">Pangsa {sim["c1_target_share"]:.1f}% (+{sim["c1_target_share"]-sim["c1_base_share"]:.1f}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------
        # GRAFIK DERET WAKTU SIMULASI
        # ---------------------------------------------------------
        st.markdown(
            '<div class="section-title">Visualisasi Tren: Baseline SARIMA vs Skenario Kebijakan</div>',
            unsafe_allow_html=True,
        )

        fig_sim = go.Figure()

        # CI 95% Skenario
        fig_sim.add_trace(
            go.Scatter(
                x=pd.concat([pd.Series(df_sim["Tanggal"]), pd.Series(df_sim["Tanggal"][::-1])]),
                y=pd.concat([df_sim["CI_Upper"], df_sim["CI_Lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(0, 119, 182, 0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval",
            )
        )

        # Baseline SARIMA
        fig_sim.add_trace(
            go.Scatter(
                x=df_sim["Tanggal"],
                y=df_sim["Baseline_SARIMA"],
                mode="lines+markers",
                name="Baseline SARIMA (Status Quo)",
                line=dict(color="#5a7d68", width=2.5, dash="dash"),
                marker=dict(size=7, color="#ffffff", line=dict(color="#5a7d68", width=2)),
                hovertemplate="<b>%{x|%B %Y}</b><br>Baseline: <b>%{y:,.0f} Wisman</b><extra></extra>",
            )
        )

        # Skenario Kebijakan
        line_color = "#00A86B" if sim["delta_wisman"] >= 0 else "#e63946"
        fill_shade = "rgba(0, 168, 107, 0.15)" if sim["delta_wisman"] >= 0 else "rgba(230, 57, 70, 0.15)"

        fig_sim.add_trace(
            go.Scatter(
                x=df_sim["Tanggal"],
                y=df_sim["Skenario_Kebijakan"],
                mode="lines+markers",
                name=f"Skenario: {chosen_preset.split(' ')[1]}",
                line=dict(color=line_color, width=3.8, shape="spline", smoothing=1.2),
                fill="tonexty",
                fillcolor=fill_shade,
                marker=dict(size=9, color="#ffffff", line=dict(color=line_color, width=2.8), symbol="diamond"),
                hovertemplate="<b>%{x|%B %Y}</b><br>Skenario: <b>%{y:,.0f} Wisman</b><br>Dampak Netto: <b>%{customdata:+,.0f} Wisman</b><extra></extra>",
                customdata=df_sim["Selisih_Wisman"],
            )
        )

        fig_sim.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.40)",
            title="Proyeksi Bulanan Kunjungan Wisatawan Mancanegara (Agt 2026 – Jul 2027)",
            xaxis=dict(title="Bulan Proyeksi", showgrid=True, gridcolor="rgba(31,81,55,0.08)"),
            yaxis=dict(title="Jumlah Wisatawan Mancanegara", showgrid=True, gridcolor="rgba(31,81,55,0.08)", ticksuffix=" "),
            height=440,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        style_plotly(fig_sim)
        render_chart(fig_sim)

        # ---------------------------------------------------------
        # BREAKDOWN FAKTOR & DISTRIBUSI CLUSTER
        # ---------------------------------------------------------
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.markdown(
                '<div class="clay-card">'
                '<div class="section-title" style="margin-top:0;font-size:1.3rem;">Kontribusi Faktor Intervensi</div>'
                '<p style="font-size:0.84rem;color:#55755e;margin-bottom:10px;">Estimasi bobot pengaruh masing-masing tuas kebijakan terhadap total variasi wisatawan.</p>',
                unsafe_allow_html=True,
            )
            factor_names = ["Kapasitas Penerbangan", "Kemudahan Visa/VoA", "Guncangan Biaya Aviasi"]
            factor_pcts = [
                flight_slider * 0.65,
                visa_slider * 0.50,
                -(cost_slider * 0.45),
            ]
            factor_colors = ["#00A86B" if v >= 0 else "#e63946" for v in factor_pcts]

            fig_factor = go.Figure(
                go.Bar(
                    x=factor_pcts,
                    y=factor_names,
                    orientation="h",
                    marker=dict(color=factor_colors),
                    text=[f"<b>{v:+.1f}%</b>" for v in factor_pcts],
                    textposition="auto",
                    hovertemplate="%{y}: <b>%{x:+.2f}%</b><extra></extra>",
                )
            )
            fig_factor.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.40)",
                xaxis=dict(title="Dampak Netto (%)", zeroline=True, zerolinecolor="#1f5137", ticksuffix="%"),
                yaxis=dict(autorange="reversed"),
                height=260,
                margin=dict(l=10, r=10, t=10, b=30),
            )
            style_plotly(fig_factor)
            render_chart(fig_factor)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b2:
            st.markdown(
                '<div class="clay-card">'
                '<div class="section-title" style="margin-top:0;font-size:1.3rem;">Pergeseran Distribusi Wilayah</div>'
                '<p style="font-size:0.84rem;color:#55755e;margin-bottom:10px;">Perbandingan serapan wisatawan antara Mega-Hubs (Cluster 0) vs Beyond Bali (Cluster 1).</p>',
                unsafe_allow_html=True,
            )

            fig_cluster_shift = go.Figure()
            categories = ["Cluster 0 (Mega-Hubs)", "Cluster 1 (Beyond Bali)"]
            base_vals = [sim["c0_base_total"], sim["c1_base_total"]]
            sim_vals = [sim["c0_sim_total"], sim["c1_sim_total"]]

            fig_cluster_shift.add_trace(
                go.Bar(
                    name="Baseline SARIMA",
                    x=categories,
                    y=base_vals,
                    marker=dict(color="#0077b6"),
                    text=[f"<b>{v/1e6:.2f}M</b>" for v in base_vals],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Baseline: <b>%{y:,.0f} Wisman</b><extra></extra>",
                )
            )
            fig_cluster_shift.add_trace(
                go.Bar(
                    name="Skenario Kebijakan",
                    x=categories,
                    y=sim_vals,
                    marker=dict(color="#00A86B"),
                    text=[f"<b>{v/1e6:.2f}M</b>" for v in sim_vals],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Skenario: <b>%{y:,.0f} Wisman</b><extra></extra>",
                )
            )
            fig_cluster_shift.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.40)",
                yaxis=dict(title="Wisatawan Mancanegara", showgrid=True, gridcolor="rgba(31,81,55,0.08)", ticksuffix=" "),
                height=260,
                margin=dict(l=10, r=10, t=10, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            style_plotly(fig_cluster_shift)
            render_chart(fig_cluster_shift)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # TABEL RINCIAN SIMULASI & DOWNLOAD
        # ---------------------------------------------------------
        st.markdown(
            '<div class="section-title">Tabel Rincian Proyeksi Bulanan</div>',
            unsafe_allow_html=True,
        )

        df_display_sim = df_sim.copy()
        df_display_sim["Bulan"] = [fmt_date_id(pd.to_datetime(d)) for d in df_display_sim["Tanggal"]]
        df_display_sim["Baseline SARIMA"] = df_display_sim["Baseline_SARIMA"].map(lambda x: f"{x:,.0f}")
        df_display_sim["Skenario Kebijakan"] = df_display_sim["Skenario_Kebijakan"].map(lambda x: f"{x:,.0f}")
        df_display_sim["Delta Wisman"] = df_display_sim["Selisih_Wisman"].map(lambda x: f"{x:+,.0f}")
        df_display_sim["Devisa Skenario (Juta USD)"] = df_display_sim["Devisa_Skenario_MUSD"].map(lambda x: f"${x:,.1f}M")
        df_display_sim["Delta Devisa (Miliar Rp)"] = (df_display_sim["Delta_Devisa_TIDR"] * 1000).map(lambda x: f"Rp {x:+,.1f} M")

        show_cols = [
            "Bulan",
            "Baseline SARIMA",
            "Skenario Kebijakan",
            "Delta Wisman",
            "Devisa Skenario (Juta USD)",
            "Delta Devisa (Miliar Rp)",
        ]
        st.dataframe(df_display_sim[show_cols], use_container_width=True, hide_index=True)

        csv_sim = df_sim.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh Data Hasil Simulasi Kebijakan (CSV)",
            data=csv_sim,
            file_name=f"simulasi_kebijakan_{chosen_preset[:12].strip().replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # Strategic take-away note
        st.markdown(
            '<div class="insight-card" style="margin-top:16px;">'
            ''
            '<div class="insight-text">'
            f'<b>Rekomendasi Kebijakan Strategis:</b><br>'
            f'Berdasarkan konfigurasi saat ini, intervensi menghasilkan net efek sebesar <b>{sim["net_growth_pct"]:+.1f}%</b> '
            f'dengan potensi perolehan devisa tambahan <b>{d_sign}Rp {sim["delta_devisa_idr"]:.2f} Triliun</b>. '
            f'Untuk mencapai pemerataan Beyond Bali yang optimal, alokasi subsidi slot penerbangan langsung '
            f'perlu dipadukan dengan kesiapan akomodasi dan promosi terpadu di 5 Destinasi Pariwisata Super Prioritas (Cluster 1).'
            '</div></div>',
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE: DETEKSI ANOMALI & ANALISIS KRISIS (ANOMALY DETECTION)
# =========================================================
elif page == "Deteksi Anomali":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Unsupervised Machine Learning • Crisis Analytics</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#133426;margin:0;">'
        "Deteksi Anomali & Analisis Krisis</h1>"
        '<p style="color:#2b3e2e;font-weight:500;margin-top:12px;max-width:70%;">'
        "Deteksi guncangan drastis (negative shocks) dan lonjakan tak biasa (positive surges) "
        "pada deret waktu pariwisata 2016–2026 menggunakan <b>Isolation Forest</b>, dilengkapi "
        "analisis kecepatan pemulihan (rebound velocity) dan catatan sejarah krisis pariwisata.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Mendeteksi titik guncangan krisis eksternal (negative shocks) dan lonjakan rebound (positive surges) sepanjang 127 bulan data historis (2016–2026).",
        metode="Unsupervised Machine Learning Isolation Forest (150 trees) dikombinasikan dengan deviasi musiman, analisis milestone kecepatan rebound, dan dossier krisis.",
        value="Mengevaluasi daya tahan (resiliensi) sistem pariwisata terhadap bencana/pandemi serta memahami durasi pemulihan sebagai panduan protokol mitigasi krisis masa depan.",
    )

    # ---------------------------------------------------------
    # FILTER KONTROL DETEKSI ANOMALI
    # ---------------------------------------------------------
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        scope_mode = st.radio(
            "Pilih Cakupan Analisis Anomali:",
            ["Tingkat Nasional (Agregat)", "Per Moda Transportasi", "Per Pintu Masuk Spesifik"],
            horizontal=True,
        )
    with col_a2:
        contamination_val = st.slider(
            "Tingkat Sensitivitas Anomali (Contamination Rate):",
            min_value=0.04,
            max_value=0.15,
            value=0.08,
            step=0.01,
            format="%.2f",
            help="Proporsi outlier yang diharapkan pada dataset deret waktu (default: 0.08 atau 8%).",
        )

    if scope_mode == "Tingkat Nasional (Agregat)":
        target_t = "nasional"
        target_n = "Nasional"
    elif scope_mode == "Per Moda Transportasi":
        target_t = "moda"
        target_n = st.selectbox("Pilih Moda Transportasi:", ["Udara", "Laut", "Darat"])
    else:
        target_t = "pintu"
        top_entries = [
            "Ngurah Rai",
            "Soekarno-Hatta",
            "Batam",
            "Juanda",
            "Kualanamu",
            "Tanjung Uban",
            "Entikong",
            "Bandara Int. Lombok",
            "Sam Ratulangi",
        ]
        target_n = st.selectbox("Pilih Pintu Masuk:", top_entries)

    # ---------------------------------------------------------
    # JALANKAN PIPELINE DETEKSI ANOMALI
    # ---------------------------------------------------------
    df_anomaly = detect_anomalies_pipeline(
        df=df,
        target_type=target_t,
        target_name=target_n,
        contamination=float(contamination_val),
    )
    rebound_stats = compute_rebound_analysis(df)

    # Ringkasan Anomali
    total_months = len(df_anomaly)
    anomalies_subset = df_anomaly[df_anomaly["Is_Anomaly"]]
    total_anomalies = len(anomalies_subset)
    shocks = df_anomaly[df_anomaly["Anomaly_Type"] == "Guncangan Negatif (Shock)"]
    surges = df_anomaly[df_anomaly["Anomaly_Type"] == "Lonjakan Positif (Surge)"]

    # Cari titik guncangan terdalam dan lonjakan tertinggi
    deepest_shock = df_anomaly.loc[df_anomaly["MoM_Pct"].idxmin()] if len(df_anomaly) > 0 else None
    highest_surge = df_anomaly.loc[df_anomaly["YoY_Pct"].idxmax()] if len(df_anomaly) > 0 else None

    # ---------------------------------------------------------
    # KPI ANOMALI
    # ---------------------------------------------------------
    st.markdown(
        f'<div class="section-title">Rangkuman Deteksi Anomali: {target_n}</div>',
        unsafe_allow_html=True,
    )

    ak1, ak2, ak3, ak4 = st.columns(4)
    with ak1:
        st.markdown(
            f'<div class="clay-card kpi">'
            f'<div class="label">PERIODE OBSERVASI</div>'
            f'<div class="value">{total_months} Bln</div>'
            f'<div class="detail">Januari 2016 – Juli 2026</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with ak2:
        st.markdown(
            f'<div class="clay-card kpi">'
            f'<div class="label">TOTAL ANOMALI TERDETEKSI</div>'
            f'<div class="value">{total_anomalies} Titik</div>'
            f'<div class="detail">{len(shocks)} Guncangan • {len(surges)} Lonjakan</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with ak3:
        if deepest_shock is not None:
            st.markdown(
                f'<div class="clay-card kpi">'
                f'<div class="label">GUNCANGAN TERDALAM</div>'
                f'<div class="value" style="color:#d32f2f;">{deepest_shock["MoM_Pct"]:.1f}%</div>'
                f'<div class="detail">{fmt_date_id(deepest_shock.name)} ({deepest_shock["Jumlah"]:,.0f})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with ak4:
        if highest_surge is not None:
            st.markdown(
                f'<div class="clay-card kpi">'
                f'<div class="label">LONJAKAN TERTINGGI (YoY)</div>'
                f'<div class="value" style="color:#27ae60;">+{highest_surge["YoY_Pct"]:.0f}%</div>'
                f'<div class="detail">{fmt_date_id(highest_surge.name)} (Rebound)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # GRAFIK DERET WAKTU INTERAKTIF DENGAN PENANDA ANOMALI
    # ---------------------------------------------------------
    st.markdown(
        '<div class="section-title">Grafik Deret Waktu & Penandaan Anomali Otomatis</div>'
        '<div class="section-subtitle">Titik merah menandakan guncangan krisis/bencana, sedangkan titik emas menandakan lonjakan/rebound super.</div>',
        unsafe_allow_html=True,
    )

    fig_anom = go.Figure()

    # Highlight area krisis COVID-19
    fig_anom.add_vrect(
        x0="2020-03-01",
        x1="2022-04-01",
        fillcolor="rgba(217, 4, 41, 0.07)",
        layer="below",
        line_width=0,
        annotation_text="Fase Krisis Pandemi COVID-19",
        annotation_position="top left",
        annotation_font=dict(color="#d90429", size=11, family="DM Sans"),
    )

    # Garis tren normal dengan smooth spline & fill lembut
    fig_anom.add_trace(
        go.Scatter(
            x=df_anomaly.index,
            y=df_anomaly["Jumlah"],
            mode="lines",
            name="Tren Kunjungan",
            line=dict(color="#00A86B", width=2.8, shape="spline", smoothing=0.8),
            fill="tozeroy",
            fillcolor="rgba(0, 168, 107, 0.07)",
            hoverinfo="skip",
        )
    )

    # Titik Normal
    df_normal = df_anomaly[~df_anomaly["Is_Anomaly"]]
    fig_anom.add_trace(
        go.Scatter(
            x=df_normal.index,
            y=df_normal["Jumlah"],
            mode="markers",
            name="Normal",
            marker=dict(size=4.5, color="#5a7d68", opacity=0.45),
            hovertemplate="<b>%{x|%B %Y}</b><br>Kunjungan: <b>%{y:,.0f} Wisman</b><extra></extra>",
        )
    )

    # Titik Guncangan Negatif (Severe Shock) dengan Jewel Crimson Markers
    if len(shocks) > 0:
        fig_anom.add_trace(
            go.Scatter(
                x=shocks.index,
                y=shocks["Jumlah"],
                mode="markers",
                name="Guncangan Negatif (Shock)",
                marker=dict(
                    size=12,
                    color="#d90429",
                    symbol="circle",
                    line=dict(width=2.5, color="#ffffff"),
                ),
                customdata=np.stack(
                    (shocks["MoM_Pct"], shocks["YoY_Pct"], shocks["Event_Name"]), axis=-1
                ),
                hovertemplate="<b>%{x|%B %Y}</b><br>Kunjungan: <b>%{y:,.0f} Wisman</b><br>MoM: <b>%{customdata[0]:+.1f}%</b><br>YoY: <b>%{customdata[1]:+.1f}%</b><br>Event: <i>%{customdata[2]}</i><extra></extra>",
            )
        )

    # Titik Lonjakan Positif (Surge) dengan Jewel Amber Diamond Markers
    if len(surges) > 0:
        fig_anom.add_trace(
            go.Scatter(
                x=surges.index,
                y=surges["Jumlah"],
                mode="markers",
                name="Lonjakan Positif (Surge)",
                marker=dict(
                    size=12,
                    color="#f77f00",
                    symbol="diamond",
                    line=dict(width=2.5, color="#ffffff"),
                ),
                customdata=np.stack(
                    (surges["MoM_Pct"], surges["YoY_Pct"], surges["Event_Name"]), axis=-1
                ),
                hovertemplate="<b>%{x|%B %Y}</b><br>Kunjungan: <b>%{y:,.0f} Wisman</b><br>MoM: <b>%{customdata[0]:+.1f}%</b><br>YoY: <b>%{customdata[1]:+.1f}%</b><br>Event: <i>%{customdata[2]}</i><extra></extra>",
            )
        )

    # Milestone Annotations
    fig_anom.add_annotation(
        x="2020-04-01",
        y=float(df_anomaly.loc["2020-04-01", "Jumlah"]) if "2020-04-01" in df_anomaly.index else 158000,
        text="<b>Lockdown Global</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#d90429",
        arrowsize=1,
        arrowwidth=1.5,
        ax=0,
        ay=-45,
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#d90429",
        borderwidth=1.5,
        borderpad=5,
        font=dict(size=11, color="#780016", family="DM Sans"),
    )

    fig_anom.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.40)",
        title=f"Deteksi Anomali Deret Waktu ({target_n} • 2016–2026)",
        xaxis=dict(title="Tahun", showgrid=True, gridcolor="rgba(31,81,55,0.08)"),
        yaxis=dict(title="Jumlah Wisatawan Mancanegara", showgrid=True, gridcolor="rgba(31,81,55,0.08)", ticksuffix=" "),
        height=450,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    style_plotly(fig_anom)
    render_chart(fig_anom)

    # ---------------------------------------------------------
    # KRONOLOGI KRISIS HISTORIS (EVENT DOSSIER)
    # ---------------------------------------------------------
    st.markdown(
        '<div class="section-title">Kronologi Krisis & Lonjakan Historis (Historical Event Dossier)</div>'
        '<div class="section-subtitle">Analisis kualitatif terhadap penyebab guncangan eksternal yang terekam pada data anomali.</div>',
        unsafe_allow_html=True,
    )

    dos_col1, dos_col2 = st.columns(2)
    for idx, ev in enumerate(HISTORICAL_EVENTS_DOSSIER):
        target_col = dos_col1 if idx % 2 == 0 else dos_col2
        with target_col:
            dossier_css = "crisis" if ev["tipe"] == "crisis" else ("rebound" if ev["tipe"] == "rebound" else "event")
            badge_css = "tag-negative" if ev["tipe"] == "crisis" else ("tag-positive" if ev["tipe"] == "rebound" else "tag-warning")
            st.markdown(
                f'<div class="dossier-card {dossier_css}">'
                f'<h4><span>{ev["judul"]}</span>'
                f'<span class="tag-badge {badge_css}">{ev["periode"]}</span></h4>'
                f'<p style="font-size:0.78rem;font-weight:700;color:#55755e;margin-bottom:6px;">{ev["lokasi"]} • {ev["tipe_label"]}</p>'
                f'<p>{ev["dampak"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # ANALISIS KECEPATAN REBOUND & RESILIENSI
    # ---------------------------------------------------------
    st.markdown(
        '<div class="section-title">Evaluasi Kecepatan Rebound Pasca-Krisis</div>'
        '<div class="section-subtitle">Durasi waktu yang dibutuhkan sektor pariwisata nasional untuk pulih dari titik nadir pandemi (Februari 2021).</div>',
        unsafe_allow_html=True,
    )

    m_cols = st.columns(4)
    milestone_data = [
        ("25% Pemulihan", rebound_stats["milestones"]["25%"], "#d4edda", "#155724"),
        ("50% Pemulihan", rebound_stats["milestones"]["50%"], "#cce5ff", "#004085"),
        ("75% Pemulihan", rebound_stats["milestones"]["75%"], "#fff3cd", "#856404"),
        ("100% Pemulihan Penuh", rebound_stats["milestones"]["100%"], "#d1e7dd", "#0f5132"),
    ]

    for col, (label, m_info, bg, fg) in zip(m_cols, milestone_data):
        with col:
            st.markdown(
                f'<div class="overview-card" style="border-top:4px solid {fg};">'
                f'<div class="ov-label">{label}</div>'
                f'<div class="ov-value" style="color:{fg};font-size:1.25rem;">{m_info["date_str"]}</div>'
                f'<div style="font-size:0.78rem;font-weight:700;color:#55755e;margin-top:6px;">'
                f'{m_info["months"]} Bulan Pasca-Nadir'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # Resiliensi Per Moda
    st.markdown("<br>", unsafe_allow_html=True)
    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        st.markdown(
            '<div class="clay-card">'
            '<div class="section-title" style="margin-top:0;font-size:1.25rem;">Komparasi Resiliensi Antar Moda</div>'
            '<p style="font-size:0.82rem;color:#55755e;margin-bottom:12px;">'
            'Tingkat pemulihan rata-rata bulanan tahun 2025 dibandingkan baseline pra-pandemi 2019.</p>',
            unsafe_allow_html=True,
        )
        modas = list(rebound_stats["moda_stats"].keys())
        rec_rates = [rebound_stats["moda_stats"][m]["recovery_rate"] for m in modas]
        moda_colors = ["#0077b6", "#00A86B", "#f77f00"]

        fig_moda_rec = go.Figure(
            go.Bar(
                x=modas,
                y=rec_rates,
                marker=dict(color=moda_colors),
                text=[f"<b>{v:.1f}%</b>" for v in rec_rates],
                textposition="outside",
                hovertemplate="<b>Jalur %{x}</b><br>Recovery Rate: <b>%{y:.1f}%</b><extra></extra>",
            )
        )
        fig_moda_rec.add_hline(
            y=100,
            line_dash="dash",
            line_color="#00A86B",
            line_width=1.5,
            annotation_text="100% Pra-Pandemi",
            annotation_position="top left",
            annotation_font=dict(color="#00A86B", size=10, family="DM Sans"),
        )
        fig_moda_rec.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.40)",
            yaxis=dict(title="Recovery Rate 2025 (%)", showgrid=True, gridcolor="rgba(31,81,55,0.08)", ticksuffix="%"),
            height=280,
            margin=dict(l=10, r=10, t=15, b=30),
        )
        style_plotly(fig_moda_rec)
        render_chart(fig_moda_rec)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res2:
        st.markdown(
            '<div class="insight-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">'
            ''
            '<div class="insight-text">'
            '<b>Temuan Analitis Resiliensi:</b><br><br>'
            '• <b>Pintu Udara</b> menunjukkan pemulihan paling konsisten seiring dibukanya rute penerbangan jarak jauh.<br>'
            '• <b>Pintu Laut (Batam/Kepri)</b> mencatatkan kecepatan rebound awal tercepat karena ketergantungan mobilitas harian/mingguan dengan Singapura dan Malaysia.<br>'
            '• <b>Pintu Darat (PLBN Kalimantan & Timor)</b> memiliki pemulihan yang sangat dipengaruhi oleh kebijakan pembukaan gerbang perbatasan darat antar-negara tetangga.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # TABEL ANOMALI & DOWNLOAD CSV
    # ---------------------------------------------------------
    st.markdown(
        '<div class="section-title">Tabel Rincian Titik Anomali Terdeteksi</div>',
        unsafe_allow_html=True,
    )

    anom_filter = st.radio(
        "Filter Tipe Anomali:",
        ["Semua Anomali", "Hanya Guncangan Negatif (Shock)", "Hanya Lonjakan Positif (Surge)"],
        horizontal=True,
    )

    if anom_filter == "Hanya Guncangan Negatif (Shock)":
        df_table_anom = shocks.copy()
    elif anom_filter == "Hanya Lonjakan Positif (Surge)":
        df_table_anom = surges.copy()
    else:
        df_table_anom = anomalies_subset.copy()

    df_table_anom["Tanggal_Str"] = [fmt_date_id(pd.to_datetime(d)) for d in df_table_anom.index]
    df_table_anom["Kunjungan"] = df_table_anom["Jumlah"].map(lambda x: f"{x:,.0f}")
    df_table_anom["MoM"] = df_table_anom["MoM_Pct"].map(lambda x: f"{x:+.1f}%")
    df_table_anom["YoY"] = df_table_anom["YoY_Pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    df_table_anom["Deviasi Musiman"] = df_table_anom["Seasonal_Dev_Pct"].map(lambda x: f"{x:+.1f}%")
    df_table_anom["Skor Anomali"] = df_table_anom["Anomaly_Score"].map(lambda x: f"{x:.3f}")
    df_table_anom["Peristiwa Historis"] = df_table_anom["Event_Name"].replace("", "—")

    table_cols = [
        "Tanggal_Str",
        "Kunjungan",
        "MoM",
        "YoY",
        "Deviasi Musiman",
        "Anomaly_Type",
        "Skor Anomali",
        "Peristiwa Historis",
    ]
    df_show = df_table_anom[table_cols].rename(
        columns={
            "Tanggal_Str": "Bulan",
            "Anomaly_Type": "Klasifikasi",
        }
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    csv_anom = df_table_anom.to_csv().encode("utf-8")
    st.download_button(
        label="Unduh Data Anomali (CSV)",
        data=csv_anom,
        file_name=f"deteksi_anomali_{target_n.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# =========================================================
# PAGE: DATA EXPLORER
# =========================================================
elif page == "Data Explorer":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Data Transparency • Auditability</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#133426;margin:0;">'
        "Jelajahi Data Kunjungan</h1>"
        '<p style="color:#2b3e2e;font-weight:500;margin-top:8px;">'
        "Eksplorasi dataset wisatawan mancanegara hasil standardisasi BPS yang telah "
        "melalui proses data cleaning bebas missing value.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Menyediakan transparansi dataset, keterbukaan informasi, dan fasilitas auditibilitas (reproducibility) bagi penguji dan peneliti.",
        metode="Dataset terstandarisasi BPS yang telah melalui pembersihan data (imputasi, unpivot tanggal, validasi tipe data) dengan fasilitas filter multi-dimensi.",
        value="Memastikan integritas seluruh pemodelan Machine Learning berdiri di atas data yang bersih, valid, dan dapat diunduh langsung untuk verifikasi independen.",
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        years = sorted(df["Tahun"].dropna().unique().tolist())
        year_sel = st.multiselect("Tahun", years,
                                  default=years[-3:] if len(years) >= 3 else years)
    with col_f2:
        entry_options = sorted(df["Pintu Masuk"].dropna().unique().tolist())
        entry_sel = st.multiselect("Pintu Masuk", entry_options, default=[])
    with col_f3:
        type_options = sorted(df["Jenis Data"].dropna().unique().tolist())
        type_sel = st.multiselect(
            "Jenis Data", type_options,
            default=["Pintu Masuk"] if "Pintu Masuk" in type_options else [],
        )

    filtered = df.copy()
    if year_sel:
        filtered = filtered[filtered["Tahun"].isin(year_sel)]
    if entry_sel:
        filtered = filtered[filtered["Pintu Masuk"].isin(entry_sel)]
    if type_sel:
        filtered = filtered[filtered["Jenis Data"].isin(type_sel)]

    kpi4 = st.columns(4)
    with kpi4[0]:
        metric_card("Baris hasil", f"{len(filtered):,}", "filtered data")
    with kpi4[1]:
        metric_card("Total kunjungan", fmt_number(filtered["Jumlah"].sum()), "filtered data")
    with kpi4[2]:
        metric_card("Missing value", f"{filtered['Jumlah'].isna().sum():,}", "nilai kosong")
    with kpi4[3]:
        metric_card("Pintu masuk unik", f"{filtered['Pintu Masuk'].nunique():,}", "filtered data")

    st.markdown('<div class="section-title">Tabel data</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered.sort_values(["Tahun","Bulan_Num","Pintu Masuk"]),
        width="stretch", hide_index=True,
    )

# =========================================================
# PAGE: TENTANG MODEL
# =========================================================
elif page == "Tentang Model":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Academic Methodology • Model Card</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#133426;margin:0;">'
        "K-Means Clustering + SARIMA Forecasting</h1>"
        '<p style="color:#2b3e2e;font-weight:500;margin-top:8px;">'
        "Dokumentasi ilmiah, formulasi matematis, dan pembuktian empiris seluruh model AI "
        "pada analitik pariwisata Indonesia.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Academic & Strategic Context Banner
    render_page_banner(
        tujuan="Mendokumentasikan kerangka ilmiah, metodologi pemodelan AI, formula matematis, serta bukti empiris performa model secara transparan.",
        metode="Model Cards standar industri: komparasi metrik clustering (Silhouette, Davies-Bouldin) dan peramalan (MAE, RMSE, MAPE, sMAPE) tanpa data leakage.",
        value="Memenuhi standar akademik pengujian proyek Machine Learning dengan pembuktian mengapa algoritma K-Means dan SARIMA terpilih sebagai solusi optimal.",
    )

    # Clustering
    st.markdown('<div class="section-title">Model Clustering</div>', unsafe_allow_html=True)
    if clustering_ok:
        c3 = st.columns(3)
        with c3[0]:
            metric_card("Algoritma", "K-Means", "unsupervised learning")
        with c3[1]:
            metric_card("k", "2", "jumlah cluster final")
        with c3[2]:
            metric_card("Objek", f"{cluster_features.shape[0]}", "pintu masuk")

    st.markdown(
        '<div class="clay-card">'
        '<h3 style="font-family:\'Baloo 2\',sans-serif;margin-top:0;color:#133426;">'
        "Penjelasan Clustering</h3>"
        '<ul style="color:#1f5137;font-size:.92rem;line-height:1.8;padding-left:20px;">'
        "<li><b>Metode:</b> K-Means adalah algoritma unsupervised learning yang "
        "mengelompokkan data tanpa label yang sudah ditentukan sebelumnya.</li>"
        "<li><b>Input:</b> 14 feature yang direkayasa dari data 2016–2019 "
        "(12 indeks musiman + log volume rata-rata + recovery 2025).</li>"
        "<li><b>PCA:</b> Digunakan hanya untuk visualisasi 2D, bukan input model clustering final.</li>"
        "<li><b>Hyperparameter:</b> n_clusters=2, random_state=42, n_init=10.</li>"
        "<li><b>Agglomerative Clustering</b> menghasilkan partisi identik dengan K-Means; "
        "K-Means dipilih sebagai model final.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    if clustering_ok:
        st.markdown('<div class="section-title">Feature Engineering (Clustering)</div>',
                    unsafe_allow_html=True)
        feature_desc = pd.DataFrame({
            "Feature": cluster_features.columns,
            "Makna": [
                "Indeks musiman Januari", "Indeks musiman Februari", "Indeks musiman Maret",
                "Indeks musiman April", "Indeks musiman Mei", "Indeks musiman Juni",
                "Indeks musiman Juli", "Indeks musiman Agustus", "Indeks musiman September",
                "Indeks musiman Oktober", "Indeks musiman November", "Indeks musiman Desember",
                "Log volume rata-rata 2016–2019",
                "Recovery 2025 terhadap baseline 2019",
            ],
        })
        st.dataframe(feature_desc, width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Evaluasi Clustering</div>', unsafe_allow_html=True)
    eval_df = pd.DataFrame({
        "Metrik": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
        "Nilai": [0.6295, 0.7270, 19.5933],
        "Interpretasi": [
            "Semakin tinggi semakin baik (maks 1.0)",
            "Semakin rendah semakin baik",
            "Semakin tinggi semakin baik",
        ],
    })
    st.dataframe(eval_df, width="stretch", hide_index=True)

    # Forecasting
    st.markdown('<div class="section-title">Model Forecasting</div>', unsafe_allow_html=True)
    fc4 = st.columns(4)
    with fc4[0]:
        metric_card("Algoritma", "SARIMA", "time-series forecasting")
    with fc4[1]:
        metric_card("Spesifikasi", "(0,1,1)(0,1,1,12)", "order + seasonal_order")
    with fc4[2]:
        metric_card("Data training", "108 bulan", "Jan 2016 – Des 2024")
    with fc4[3]:
        metric_card("Data testing", "12 bulan", "Jan – Des 2025")
    st.markdown(
        '<div class="clay-card">'
        '<h3 style="font-family:\'Baloo 2\',sans-serif;margin-top:0;color:#133426;">'
        "Penjelasan Forecasting</h3>"
        '<ul style="color:#1f5137;font-size:.92rem;line-height:1.8;padding-left:20px;">'
        "<li><b>SARIMA</b> (Seasonal ARIMA) menangkap ketergantungan temporal dan pola musiman.</li>"
        "<li><b>Target:</b> total wisatawan mancanegara bulanan (semua pintu masuk).</li>"
        "<li><b>Frekuensi:</b> bulanan (MS — Month Start).</li>"
        "<li><b>Periode musiman:</b> 12 bulan.</li>"
        "<li><b>Train/test:</b> data sebelum 2025 untuk training, data 2025 untuk testing. "
        "Tidak ada data leakage.</li>"
        "<li><b>Model final</b> difit menggunakan seluruh 127 data bulanan yang diketahui "
        "(Jan 2016 – Jul 2026).</li>"
        "<li><b>CI 95%</b> dihitung dari <code>get_forecast().conf_int()</code>.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Evaluasi Forecasting</div>', unsafe_allow_html=True)
    fc_eval_df = pd.DataFrame({
        "Metrik": ["MAE", "RMSE", "MAPE", "sMAPE"],
        "SARIMA (Best)": ["79,871.19", "88,965.73", "6.36%", "6.42%"],
        "XGBoost": ["85,525.98", "128,949.51", "7.58%", "6.98%"],
        "Random Forest": ["106,029.15", "125,554.62", "8.71%", "8.47%"],
        "Holt-Winters": ["175,262.03", "193,497.40", "13.18%", "14.18%"],
        "Prophet": ["615,747.89", "626,272.32", "47.69%", "62.81%"],
        "Keterangan": [
            "Semakin rendah semakin baik",
            "Semakin rendah semakin baik",
            "Semakin rendah semakin baik",
            "Semakin rendah semakin baik",
        ],
    })
    st.dataframe(fc_eval_df, width="stretch", hide_index=True)

    st.markdown(
        '<div class="note-box">'
        "<b>Catatan metodologis:</b> Model clustering menggunakan 14 feature yang sama "
        "dengan proses fitting. PCA hanya digunakan untuk visualisasi 2D. "
        "Model SARIMA tidak dilatih ulang setiap kali halaman dibuka; "
        "forecast dihitung dari model tersimpan di <code>model/tourism_sarima_model.pkl</code>."
        "</div>",
        unsafe_allow_html=True,
    )