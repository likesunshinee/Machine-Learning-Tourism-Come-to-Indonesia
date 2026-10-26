# =========================================================
# Nusantara Transit Intelligence
# Indonesian Tourism Analytics Dashboard
# =========================================================

from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

# =========================================================
# PAGE CONFIG (must be first Streamlit call)
# =========================================================
st.set_page_config(
    page_title="Nusantara Transit Intelligence",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "tourism_indonesia_2016-2026_clean.csv"
MODEL_PATH = BASE_DIR / "model" / "kmeans_clustering_final.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler_clustering.pkl"
SARIMA_PATH = BASE_DIR / "model" / "tourism_sarima_model.pkl"
# =========================================================
# GLOBAL CSS - Claymorphism Tropical
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Baloo+2:wght@500;600;700;800&display=swap');

:root {
    --bg: #e9f3dc;
    --bg-soft: #f5f8ea;
    --card: #f7f7e9;
    --card-2: #dff0ce;
    --text: #1F5137;
    --muted: #2D4D33;
    --green: #1F5137;
    --green-dark: #133426;
    --mint: #bfe0bd;
    --yellow: #f1cc6d;
    --orange: #e8995d;
    --blue: #8dc3d5;
    --sea: #8fbfc2;
    --shadow-dark: rgba(78,93,73,.24);
    --shadow-light: rgba(255,255,255,.85);
    --radius: 28px;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(241,204,109,.38), transparent 22%),
        radial-gradient(circle at 92% 14%, rgba(141,195,213,.34), transparent 23%),
        linear-gradient(135deg, #dceccf 0%, #eef4e1 50%, #dcefe2 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 18% 12%, rgba(241, 204, 109, 0.35), transparent 25%),
        radial-gradient(circle at 85% 88%, rgba(141, 195, 213, 0.30), transparent 25%),
        linear-gradient(180deg, #d3e8c5 0%, #dff0ce 60%, #cfe5c0 100%) !important;
    border-right: 1px solid rgba(31, 81, 55, 0.12) !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Sidebar Brand Card */
.sidebar-brand {
    background: linear-gradient(145deg, #ffffff 0%, #edf5e3 100%);
    padding: 22px 18px;
    border-radius: 24px;
    box-shadow: 8px 8px 18px var(--shadow-dark), -6px -6px 14px var(--shadow-light);
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
    background: linear-gradient(90deg, #1f5137 0%, #8dc3d5 50%, #f1cc6d 100%);
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
/* Completely hide widget label and stRadio header so no duplicate 'Navigasi Utama' appears */
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

/* Claymorphic Nav Pill Buttons - ONLY inside role="radiogroup" */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: linear-gradient(145deg, #ffffff 0%, #edf5e3 100%) !important;
    border-radius: 16px !important;
    padding: 12px 18px !important;
    margin: 0 !important;
    border: 1.5px solid rgba(255, 255, 255, 0.9) !important;
    box-shadow: 4px 4px 10px rgba(78, 93, 73, 0.12), -3px -3px 8px rgba(255, 255, 255, 0.9) !important;
    cursor: pointer !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

/* UNSELECTED TEXT: Selalu jelas, pekat, dan terbaca tanpa harus diklik */
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
    background: linear-gradient(145deg, #ffffff 0%, #dff0ce 100%) !important;
    transform: translateX(4px) translateY(-1px) !important;
    box-shadow: 6px 6px 14px rgba(78, 93, 73, 0.18), -4px -4px 10px rgba(255, 255, 255, 0.95) !important;
    border-color: rgba(31, 81, 55, 0.3) !important;
}

/* ACTIVE / SELECTED STATE: Forest Deep Green dengan teks putih tebal */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #1f5137 0%, #133426 100%) !important;
    border: 1.5px solid #0d251a !important;
    box-shadow: 4px 4px 12px rgba(19, 52, 38, 0.35), inset 1px 1px 2px rgba(255, 255, 255, 0.25) !important;
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
    background: linear-gradient(145deg, #f7f9ee, #e4ecda);
    border-radius: 20px;
    padding: 16px;
    box-shadow: 6px 6px 14px rgba(78, 93, 73, 0.12), -4px -4px 10px rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.6);
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
    box-shadow: 0 0 6px rgba(39, 174, 96, 0.6);
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
    border-bottom: 1px dashed rgba(31, 81, 55, 0.12);
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

.clay-card {
    background: linear-gradient(145deg, #fbfbef, #e7eadb);
    border-radius: var(--radius);
    box-shadow: 12px 12px 22px var(--shadow-dark), -9px -9px 18px var(--shadow-light);
    padding: 24px;
    border: 1px solid rgba(255,255,255,.5);
    color: var(--green);
    margin-bottom: 16px;
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

/* Light Clay Selectbox Control */
div[data-baseweb="select"] {
    background: transparent !important;
    border-radius: 16px !important;
}
div[data-baseweb="select"] > div {
    background: linear-gradient(145deg, #ffffff 0%, #edf5e3 100%) !important;
    border: 1.5px solid rgba(31, 81, 55, 0.25) !important;
    border-radius: 16px !important;
    box-shadow: 3px 3px 8px rgba(78, 93, 73, 0.08), -2px -2px 6px rgba(255, 255, 255, 0.9) !important;
    overflow: hidden !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(31, 81, 55, 0.5) !important;
    box-shadow: 4px 4px 12px rgba(78, 93, 73, 0.12), -3px -3px 8px rgba(255, 255, 255, 1) !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #1f5137 !important;
    box-shadow: 0 0 0 2px rgba(31, 81, 55, 0.2) !important;
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

/* Light Clay Dropdown Popover & Options Menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [role="listbox"],
ul[data-baseweb="menu"],
div[data-baseweb="menu"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-radius: 16px !important;
    border: 1.5px solid rgba(31, 81, 55, 0.2) !important;
    box-shadow: 0 12px 28px rgba(31, 81, 55, 0.18), 0 4px 10px rgba(0, 0, 0, 0.06) !important;
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

.hero {
    position: relative; overflow: hidden;
    background:
        radial-gradient(circle at 88% 28%, rgba(141,195,213,.46), transparent 22%),
        radial-gradient(circle at 17% 85%, rgba(241,204,109,.34), transparent 23%),
        linear-gradient(135deg, #f6f7ea, #dcefcf);
    min-height: 310px;
    padding: 36px;
    border-radius: var(--radius);
    box-shadow: 12px 12px 22px var(--shadow-dark), -9px -9px 18px var(--shadow-light);
    border: 1px solid rgba(255,255,255,.5);
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

.kpi { min-height: 140px; display: flex; flex-direction: column; justify-content: space-between; }
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
    background: #edf3df;
    box-shadow: inset 3px 3px 6px rgba(91,100,85,.14), inset -3px -3px 6px rgba(255,255,255,.65);
    font-size: .84rem; font-weight: 700; color: #1f5137;
}

.note-box {
    background: #fff7dc; border-radius: 20px; padding: 16px 18px;
    box-shadow: 7px 7px 12px rgba(116,99,45,.15), -5px -5px 10px rgba(255,255,255,.7);
    color: #3d3010; font-size: .88rem; font-weight: 500;
}

.cluster-chip {
    display: inline-block; padding: 8px 13px;
    border-radius: 999px; font-weight: 800; font-size: .8rem;
}
.cluster-0 { background: #c8b4e8; color: #2e1554; }
.cluster-1 { background: #a8d9b5; color: #0f3320; }
.insight-card {
    background: linear-gradient(145deg, #f0f7e8, #e0ecce);
    border-radius: 20px; padding: 18px 20px;
    box-shadow: 8px 8px 14px var(--shadow-dark), -6px -6px 12px var(--shadow-light);
    border-left: 4px solid #1f5137; color: #1f5137; margin-bottom: 10px;
}
.insight-card .insight-icon { font-size: 1.5rem; margin-bottom: 6px; }
.insight-card .insight-text { font-size: .9rem; font-weight: 600; line-height: 1.5; color: #1f5137; }

.overview-card {
    background: linear-gradient(145deg, #fbfbef, #e9eedd);
    border-radius: 22px; padding: 18px 20px;
    box-shadow: 8px 8px 14px var(--shadow-dark), -6px -6px 12px var(--shadow-light);
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
    background: linear-gradient(145deg, #f3f8ea, #e4ecd8);
    border-radius: 16px; padding: 14px 16px;
    box-shadow: 5px 5px 10px var(--shadow-dark), -4px -4px 8px var(--shadow-light);
    text-align: center; color: #1f5137; font-size: .82rem; font-weight: 700;
}

.best-badge {
    display: inline-block; background: #a8d9b5; color: #0f3320;
    padding: 3px 10px; border-radius: 999px;
    font-size: .72rem; font-weight: 800; letter-spacing: .04em;
}

footer { display: none; }
</style>
""",
    unsafe_allow_html=True,
)
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


def style_plotly(fig):
    """Apply consistent styling to all Plotly charts."""
    
    green = "#1f5137"
    white = "#ffffff"
    dark_bg = "#202521"

    layout_update = dict(
        font=dict(
            family="DM Sans, sans-serif",
            color=green
        ),
        legend=dict(
            font=dict(
                color=green
            )
        ),

        xaxis=dict(
            title_font=dict(
                color=green
            ),
            tickfont=dict(
                color=green
            )
        ),

        yaxis=dict(
            title_font=dict(
                color=green
            ),
            tickfont=dict(
                color=green
            )
        ),

        polar=dict(
            angularaxis=dict(
                tickfont=dict(
                    color=green
                )
            ),
            radialaxis=dict(
                tickfont=dict(
                    color=green
                ),
                title_font=dict(
                    color=green
                )
            )
        ),

        # ==========================================
        # HOVER TOOLTIP
        # ==========================================
        hoverlabel=dict(
            bgcolor=dark_bg,
            bordercolor=green,
            font=dict(
                family="DM Sans, sans-serif",
                color=white,
                size=12
            ),
            align="left"
        )
    )

    if fig.layout.title and fig.layout.title.text:
        layout_update["title_font"] = dict(family="Baloo 2, sans-serif", color=green)

    fig.update_layout(**layout_update)

    # Data label / text pada chart
    for trace in fig.data:
        if hasattr(trace, "textfont"):
            try:
                trace.textfont = dict(
                    color=green
                )
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
        <div class="brand-badge">🌴 INDONESIA TOURISM</div>
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
    "Data Explorer": "Data Explorer",
    "Transport Insights": "Transport Insights",
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
            <span class="status-dot"></span> Model Ready & Cached
        </div>
        <div class="meta-grid">
            <div class="meta-item">
                <span class="meta-label">Tema</span>
                <span class="meta-val">Claymorphism Tropical</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Moda</span>
                <span class="meta-val">Udara • Darat • Laut</span>
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
                <span class="meta-label">Cakupan</span>
                <span class="meta-val">26 Pintu (2016–2026)</span>
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
    # Hero section (text only)
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Indonesia Tourism Intelligence</div>'
        "<h1>Tourism Indonesia,<br>in Number.</h1>"
        "<p>Dashboard interaktif untuk mengeksplorasi kunjungan wisatawan "
        "mancanegara berdasarkan pintu masuk, waktu, moda transportasi, "
        "dan hasil clustering K-Means.</p>"
        '<div style="margin-top:16px;">'
        '<span class="transport-pill">✈ Udara</span>'
        '<span class="transport-pill">🚌 Darat</span>'
        '<span class="transport-pill">⛴ Laut</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    
    # Transport illustration via modern native st.html
    st.html(
        f'<div style="margin-top:-55px;text-align:right;pointer-events:none;padding-right:28px;">'
        f'<div style="display:inline-block;width:44%;min-width:280px;">'
        f'{get_transport_svg()}'
        f"</div></div>"
    )
    # KPI snapshot (cached)
    annual, latest_year, latest_total, total_period = get_dashboard_summary(df)
    n_entry = int(cluster_features.index.nunique()) if clustering_ok else "—"
    n_cluster = int(len(np.unique(cluster_labels))) if clustering_ok else "—"

    st.markdown('<div class="section-title">Snapshot</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total kunjungan 2016–2026", fmt_number(total_period), "data bersih")
    with cols[1]:
        metric_card(f"Kunjungan {latest_year}", fmt_number(latest_total), "tahun terakhir")
    with cols[2]:
        metric_card("Pintu masuk", str(n_entry), "objek clustering")
    with cols[3]:
        metric_card("Cluster final", str(n_cluster), "K-Means • k = 2")

    st.markdown(
        '<div class="note-box" style="margin-top:8px;">'
        "<b>Catatan data:</b> tahun 2026 tersedia di dataset hingga Juli 2026. "
        "Data Agustus 2026 ke atas merupakan hasil prediksi SARIMA, bukan data aktual."
        "</div>",
        unsafe_allow_html=True,
    )

    # Executive Insights
    st.markdown('<div class="section-title">Ringkasan Insight</div>', unsafe_allow_html=True)
    insights = []
    pre_covid = annual[annual["Tahun"] < 2020]
    if not pre_covid.empty:
        peak_row = pre_covid.loc[pre_covid["Jumlah"].idxmax()]
        insights.append(("📈",
            f"Puncak kunjungan pra-pandemi terjadi pada <b>{int(peak_row['Tahun'])}</b> "
            f"dengan total <b>{fmt_number(peak_row['Jumlah'])}</b> wisatawan mancanegara."))
    y2019 = annual.loc[annual["Tahun"] == 2019, "Jumlah"]
    y2025 = annual.loc[annual["Tahun"] == 2025, "Jumlah"]
    if not y2019.empty and not y2025.empty:
        rec = float(y2025.iloc[0]) / float(y2019.iloc[0]) * 100
        insights.append(("🔄",
            f"Tingkat pemulihan 2025 terhadap baseline 2019 mencapai "
            f"<b>{rec:.1f}%</b>, menandakan pemulihan pariwisata pasca-pandemi."))
    route_df = transport_mix(df)
    latest_route = route_df[route_df["Tahun"] == latest_year]
    if not latest_route.empty:
        dom = latest_route.loc[latest_route["Jumlah"].idxmax()]
        dom_share = dom["Jumlah"] / latest_route["Jumlah"].sum() * 100
        insights.append(("✈️",
            f"Jalur masuk dominan pada {latest_year} adalah <b>{dom['Rute']}</b> "
            f"dengan pangsa <b>{dom_share:.1f}%</b> dari total kunjungan."))
    if clustering_ok and len(cluster_labels) > 0:
        c0n = int(np.sum(cluster_labels == 0))
        c1n = int(np.sum(cluster_labels == 1))
        insights.append(("🗺️",
            f"Clustering K-Means mengidentifikasi <b>{c0n} pintu masuk</b> dengan "
            f"pola musiman khusus (Cluster 0) dan <b>{c1n} pintu masuk</b> pola umum (Cluster 1)."))

    cols_ins = st.columns(2)
    for i, (icon, text) in enumerate(insights):
        with cols_ins[i % 2]:
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-icon">{icon}</div>'
                f'<div class="insight-text">{text}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Annual trend
    st.markdown('<div class="section-title">Tren tahunan</div>', unsafe_allow_html=True)
    fig = px.area(annual, x="Tahun", y="Jumlah", markers=True,
                  labels={"Jumlah": "Jumlah wisatawan", "Tahun": ""})
    fig.update_traces(line_width=4, line_color="#1f5137", fillcolor="rgba(31,81,55,0.15)")
    fig.update_layout(margin=dict(l=0,r=0,t=20,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
                      hovermode="x unified")
    style_plotly(fig)
    render_chart(fig)
    # Transport mode
    st.markdown('<div class="section-title">Komposisi moda transportasi</div>', unsafe_allow_html=True)
    route_colors = {"Udara": "#8dc3d5", "Darat": "#e8995d", "Laut": "#8fbfc2"}
    fig_rt = px.area(route_df, x="Tahun", y="Jumlah", color="Rute", markers=True,
                     labels={"Jumlah": "Jumlah wisatawan", "Tahun": "", "Rute": "Moda"},
                     color_discrete_map=route_colors)
    fig_rt.update_layout(margin=dict(l=0,r=0,t=20,b=0),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
                         hovermode="x unified")
    style_plotly(fig_rt)
    render_chart(fig_rt)

    # Cluster summary
    if clustering_ok:
        st.markdown('<div class="section-title">Ringkasan cluster</div>', unsafe_allow_html=True)
        profile = cluster_features_with_label.copy()
        profile.index.name = None
        profile = profile.reset_index().rename(columns={"index": "Pintu Masuk"})
        c0_list = profile[profile["Cluster"] == 0]["Pintu Masuk"].tolist()
        c1_list = profile[profile["Cluster"] == 1]["Pintu Masuk"].tolist()
        col_a, col_b = st.columns(2)
        with col_a:
            c0_names = " • ".join(c0_list)
            st.markdown(
                f'<div class="clay-card">'
                f'<span class="cluster-chip cluster-0">CLUSTER 0</span>'
                f'<h3 style="font-family:\'Baloo 2\',sans-serif;margin:10px 0 4px;">Profil Khusus</h3>'
                f'<p style="font-size:.85rem;font-weight:600;">{len(c0_list)} pintu masuk</p>'
                f'<div class="cluster-members">{c0_names}</div>'
                f'<p style="font-size:.82rem;margin-top:12px;color:#2d4d33;">'
                f"Pola musiman sangat khas dengan karakteristik volume dan "
                f"recovery yang berbeda dari mayoritas pintu masuk.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            c1_preview = " • ".join(c1_list[:8])
            remainder = len(c1_list) - 8
            st.markdown(
                f'<div class="clay-card">'
                f'<span class="cluster-chip cluster-1">CLUSTER 1</span>'
                f'<h3 style="font-family:\'Baloo 2\',sans-serif;margin:10px 0 4px;">Profil Umum</h3>'
                f'<p style="font-size:.85rem;font-weight:600;">{len(c1_list)} pintu masuk</p>'
                f'<div class="cluster-members">{c1_preview} ... dan {remainder} lainnya</div>'
                f'<p style="font-size:.82rem;margin-top:12px;color:#2d4d33;">'
                f"Kelompok mayoritas dengan pola musiman yang lebih umum "
                f"dan volume relatif lebih tinggi.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Forecast teaser
    if sarima_ok and sarima_artifact is not None:
        try:
            _fc = sarima_artifact["model"].get_forecast(steps=12).predicted_mean
            avg_fc = _fc.mean()
            peak_fc = _fc.max()
            peak_m = _fc.idxmax()
            st.markdown('<div class="section-title">Teaser Forecast</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="note-box" style="margin-bottom:16px;">'
                "<b>SARIMA Forecast:</b> prediksi 12 bulan ke depan (Agustus 2026 – Juli 2027). "
                "Kunjungi halaman <b>Forecasting</b> untuk analisis lengkap."
                "</div>",
                unsafe_allow_html=True,
            )
            tc = st.columns(3)
            with tc[0]:
                metric_card("Rata-rata prediksi", fmt_number(avg_fc), "Agu 2026 – Jul 2027")
            with tc[1]:
                metric_card("Prediksi tertinggi", fmt_number(peak_fc), fmt_date_id(peak_m))
            with tc[2]:
                metric_card("Bulan puncak", fmt_date_id(peak_m), "puncak musiman")
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
        '<span class="transport-pill">🗺 26 Pintu Masuk</span>'
        '<span class="transport-pill">✈ Udara</span>'
        '<span class="transport-pill">⛴ Laut</span>'
        '<span class="transport-pill">🚌 Darat</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
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
        "Cluster 0 (Mega-Hub)": "#133426",
        "Cluster 1 (Regional Gateway)": "#e8995d",
        "Udara": "#2f6f4e",
        "Laut": "#008080",
        "Darat": "#e8995d",
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
            bgcolor="rgba(247,247,233,0.85)",
            bordercolor="rgba(31,81,55,0.2)",
            borderwidth=1,
        ),
    )
    style_plotly(fig_map)
    render_chart(fig_map)

    st.markdown(
        '<div class="note-box" style="margin-top:4px;margin-bottom:18px;">'
        "💡 <b>Panduan interaktif:</b> Ukuran lingkaran merefleksikan besaran volume kunjungan. "
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
            fig_p = px.line(
                annual_p,
                x="Tahun",
                y="Jumlah",
                markers=True,
                labels={"Jumlah": "Wisman", "Tahun": ""},
                title=f"Tren Kunjungan Wisman — {sel_pintu}"
            )
            fig_p.update_traces(line_width=3, line_color="#1f5137", marker=dict(size=7, color="#e8995d"))
            fig_p.update_layout(
                margin=dict(l=0, r=0, t=35, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.22)",
                height=260,
                hovermode="x unified",
            )
            style_plotly(fig_p)
            render_chart(fig_p)

    # Spatial Insights
    st.markdown('<div class="section-title">Wawasan Spasial & Geopolitik Pariwisata</div>', unsafe_allow_html=True)
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown(
            '<div class="insight-card">'
            '<div class="insight-icon">✈️</div>'
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
            '<div class="insight-icon">⛴️</div>'
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
            '<div class="insight-icon">🚌</div>'
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
        fig_season.add_trace(go.Scatterpolar(
            r=seasonal_values + [seasonal_values[0]],
            theta=MONTH_NAMES + [MONTH_NAMES[0]],
            fill="toself",
            name=selected_name,
            line=dict(color="#1f5137", width=2.5),
            fillcolor="rgba(31,81,55,0.18)",
        ))
        fig_season.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, tickfont=dict(color="#1e5c3a", size=11),
                                gridcolor="rgba(30,92,58,0.28)", linecolor="rgba(30,92,58,0.45)"),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=MONTH_NAMES,
                    ticktext=MONTH_NAMES,
                    tickfont=dict(color="#1e5c3a", size=12),
                    gridcolor="rgba(30,92,58,0.28)",
                    linecolor="rgba(30,92,58,0.45)",
                    direction="clockwise",
                    rotation=90,
                ),
            ),
            margin=dict(l=60, r=60, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e5c3a"),
            showlegend=False,
            title=dict(text="Profil Musiman",
                       font=dict(family="Baloo 2, sans-serif", color="#133426", size=16)),
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
            f'<div class="insight-icon">💡</div>'
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
        '<span class="transport-pill">🔮 12-Month Ahead Forecast</span>'
        '<span class="transport-pill">✈ Udara</span>'
        '<span class="transport-pill">🚌 Darat</span>'
        '<span class="transport-pill">⛴ Laut</span>'
        "</div></div>",
        unsafe_allow_html=True,
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
    target_badge = "🇮🇩 Nasional (Total Indonesia)"

    with fc_ctrl2:
        if fc_scope == "Per Moda Transportasi":
            selected_target_type = "moda"
            selected_target_name = st.selectbox("Pilih Moda Transportasi:", ["Udara", "Laut", "Darat"])
            icon_map = {"Udara": "✈️", "Laut": "⛴️", "Darat": "🚌"}
            target_badge = f"{icon_map.get(selected_target_name, '🚗')} Moda {selected_target_name}"
            st.caption(f"Memprediksi proyeksi pergerakan wisatawan untuk seluruh pintu {selected_target_name}.")
        elif fc_scope == "Per Pintu Masuk Spesifik":
            selected_target_type = "pintu_masuk"
            pintu_options = sorted(df[df["Jenis Data"] == "Pintu Masuk"]["Pintu Masuk"].unique().tolist())
            def_idx = pintu_options.index("Ngurah Rai") if "Ngurah Rai" in pintu_options else 0
            selected_target_name = st.selectbox("Pilih Pintu Masuk:", pintu_options, index=def_idx)
            target_badge = f"📍 Pintu Masuk: {selected_target_name}"
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
    fig_fc.add_trace(go.Scatter(
        x=series_known.index, y=series_known.values,
        name="Data Aktual (2016–Jul 2026)", line=dict(color="#1f5137", width=2.5), mode="lines",
    ))
    ci_x = list(future_forecast.index) + list(future_forecast.index[::-1])
    ci_y = list(conf_int[upper_col]) + list(conf_int[lower_col][::-1])
    fig_fc.add_trace(go.Scatter(
        x=ci_x, y=ci_y, fill="toself",
        fillcolor="rgba(141,195,213,0.25)",
        line=dict(color="rgba(141,195,213,0)"),
        name="Interval Kepercayaan 95% (CI)", showlegend=True,
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_forecast.index, y=future_forecast.values,
        name="Forecast SARIMA (Agu 2026–Jul 2027)",
        line=dict(color="#e8995d", width=2.8, dash="dash"),
        mode="lines+markers", marker=dict(size=6, color="#e8995d"),
    ))
    boundary_ts = series_known.index[-1]
    fig_fc.add_vline(
        x=boundary_ts.value / 1e6,
        line_dash="dot", line_color="#1f5137", line_width=1.5,
        annotation_text="Mulai Forecast (Agu 2026)", annotation_position="top",
        annotation_font=dict(color="#1f5137", size=12),
    )
    fig_fc.update_layout(
        title=dict(
            text=f"Proyeksi Deret Waktu — {selected_target_name}",
            font=dict(family="Baloo 2, sans-serif", color="#133426", size=15),
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Tanggal"), yaxis=dict(title="Jumlah wisatawan mancanegara"),
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
        f'<div class="insight-icon">📈</div>'
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
        label=f"📥 Download Data Forecast ({selected_target_name}) dalam CSV",
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
        badge = '<span class="best-badge">⭐ BEST</span>' if is_best else ""
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
    mape_colors = ["#1f5137" if b else "#bfe0bd" for b in comp_best]
    fig_mape = go.Figure(go.Bar(
        x=comp_names, y=comp_mapes,
        text=[f"{v:.2f}%" for v in comp_mapes],
        textposition="outside",
        marker_color=mape_colors,
    ))
    fig_mape.update_layout(
        margin=dict(l=0,r=0,t=40,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
        yaxis=dict(title="MAPE (%)", ticksuffix="%"), xaxis=dict(title=""),
        title=dict(text="MAPE per Model — Semakin rendah semakin baik",
                   font=dict(family="Baloo 2, sans-serif", color="#133426", size=16)),
    )
    style_plotly(fig_mape)
    render_chart(fig_mape)

    # 7. Methodology
    st.markdown('<div class="section-title">Metodologi Forecasting</div>', unsafe_allow_html=True)
    steps = [
        ("📊", "Data Historis", "Jan 2016 – Jul 2026"),
        ("📅", "Agregasi Bulanan", "MS frequency"),
        ("✂️", "Train / Test Split", "Train: s.d. 2024 | Test: 2025"),
        ("🤖", "5 Algoritma", "HW · SARIMA · Prophet · RF · XGB"),
        ("📐", "Evaluasi", "MAE · RMSE · MAPE · sMAPE"),
        ("🏆", "Model Terbaik", "SARIMA"),
        ("🔮", "Forecast Final", "12 bulan ke depan"),
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
        "<b>⚠️ Catatan penting tentang data 2026:</b><br>"
        "Dataset bersih hanya memuat data aktual hingga <b>Juli 2026</b>. "
        "Periode <b>Agustus 2026 – Juli 2027</b> sepenuhnya merupakan "
        "<b>hasil prediksi SARIMA</b> dan tidak boleh diperlakukan sebagai "
        "data historis yang terukur. Prediksi dibuat berdasarkan pola dari "
        "127 observasi bulanan (Januari 2016 – Juli 2026)."
        "</div>",
        unsafe_allow_html=True,
    )
# =========================================================
# PAGE: DATA EXPLORER
# =========================================================
elif page == "Data Explorer":
    st.markdown(
        '<div class="clay-card">'
        '<div class="eyebrow">Data Explorer</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#205438;margin:0;">'
        "Jelajahi data kunjungan</h1>"
        '<p style="color:#2b3e2e;font-weight:500;margin-top:8px;">'
        "Gunakan filter untuk mengeksplorasi dataset wisatawan yang telah "
        "melalui proses <i>data cleaning</i>.</p>"
        "</div>",
        unsafe_allow_html=True,
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
# PAGE: TRANSPORT INSIGHTS
# =========================================================
elif page == "Transport Insights":
    st.markdown(
        '<div class="clay-card">'
        '<div class="eyebrow">Transport Insights</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#205438;margin:0;">'
        "Udara • Darat • Laut</h1>"
        '<p style="color:#2b3e2e;font-weight:500;margin-top:8px;">'
        "Komposisi kunjungan menurut jalur masuk utama Indonesia.</p>"
        '<div style="margin-top:14px;">'
        '<span class="transport-pill">✈ Penerbangan</span>'
        '<span class="transport-pill">🚌 Kendaraan darat</span>'
        '<span class="transport-pill">⛴ Kapal / perahu</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    route = transport_mix(df)
    latest_year_rt = int(route["Tahun"].max())
    latest_route = route[route["Tahun"] == latest_year_rt].copy()
    route_colors = {"Udara": "#8dc3d5", "Darat": "#e8995d", "Laut": "#8fbfc2"}

    if not latest_route.empty:
        total_rt = latest_route["Jumlah"].sum()
        rt_kpi = st.columns(3)
        for i, rute in enumerate(["Udara", "Darat", "Laut"]):
            row_r = latest_route[latest_route["Rute"] == rute]
            val = float(row_r["Jumlah"].iloc[0]) if not row_r.empty else 0.0
            with rt_kpi[i]:
                metric_card(
                    f"Jalur {rute} ({latest_year_rt})", fmt_number(val),
                    f"{val/total_rt*100:.1f}% dari total" if total_rt > 0 else "",
                )

    st.markdown('<div class="section-title">Tren moda transportasi</div>', unsafe_allow_html=True)
    fig_tr = px.area(route, x="Tahun", y="Jumlah", color="Rute", markers=True,
                     labels={"Jumlah": "Jumlah wisatawan", "Tahun": "", "Rute": "Moda"},
                     color_discrete_map=route_colors)
    fig_tr.update_layout(margin=dict(l=0,r=0,t=20,b=0),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
                         hovermode="x unified")
    style_plotly(fig_tr)
    render_chart(fig_tr)
    st.markdown(f'<div class="section-title">Komposisi {latest_year_rt}</div>', unsafe_allow_html=True)
    fig_bar = px.bar(latest_route.sort_values("Jumlah", ascending=False),
                     x="Rute", y="Jumlah", text="Jumlah", color="Rute",
                     labels={"Jumlah": "Jumlah wisatawan", "Rute": ""},
                     color_discrete_map=route_colors)
    fig_bar.update_traces(texttemplate="%{text:,.0f}")
    fig_bar.update_layout(margin=dict(l=0,r=0,t=20,b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
                          showlegend=False)
    style_plotly(fig_bar)
    render_chart(fig_bar)

    pivot = route[route["Tahun"].isin([2019, 2025])].pivot_table(
        index="Rute", columns="Tahun", values="Jumlah", aggfunc="sum"
    )
    if 2019 in pivot.columns and 2025 in pivot.columns:
        pivot["Recovery_2025"] = pivot[2025] / pivot[2019] * 100
        st.markdown(
            '<div class="section-title">Recovery 2025 terhadap 2019</div>',
            unsafe_allow_html=True,
        )
        recovery = pivot.reset_index()[["Rute", "Recovery_2025"]]
        fig_rec = px.bar(recovery.sort_values("Recovery_2025", ascending=False),
                         x="Rute", y="Recovery_2025", text="Recovery_2025", color="Rute",
                         labels={"Recovery_2025": "Recovery (%)", "Rute": ""},
                         color_discrete_map=route_colors)
        fig_rec.add_hline(y=100, line_dash="dash", line_color="#133426",
                           annotation_text="Baseline 2019 (100%)",
                           annotation_font=dict(color="#133426"))
        fig_rec.update_traces(texttemplate="%{text:.1f}%")
        fig_rec.update_layout(margin=dict(l=0,r=0,t=20,b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
                               showlegend=False)
        style_plotly(fig_rec)
        render_chart(fig_rec)

        best_rec = recovery.loc[recovery["Recovery_2025"].idxmax()]
        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-icon">🔄</div>'
            f'<div class="insight-text">'
            f"Jalur masuk <b>{best_rec['Rute']}</b> menunjukkan pemulihan terbaik "
            f"pada 2025 dibandingkan baseline 2019, mencapai "
            f"<b>{best_rec['Recovery_2025']:.1f}%</b>."
            f"</div></div>",
            unsafe_allow_html=True,
        )
# =========================================================
# PAGE: TENTANG MODEL
# =========================================================
elif page == "Tentang Model":
    st.markdown(
        '<div class="clay-card">'
        '<div class="eyebrow">Model Card</div>'
        '<h1 style="font-family:\'Baloo 2\',sans-serif;color:#205438;margin:0;">'
        "K-Means Clustering + SARIMA Forecasting</h1>"
        '<p style="color:#2b3e2e;font-weight:500;margin-top:8px;">'
        "Model card komprehensif untuk proyek machine learning analitik "
        "pariwisata Indonesia.</p>"
        "</div>",
        unsafe_allow_html=True,
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