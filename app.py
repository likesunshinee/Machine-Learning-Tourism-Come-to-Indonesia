# =========================================================
# Nusantara Transit Intelligence
# Indonesian Tourism Analytics Dashboard
# =========================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
        radial-gradient(circle at 25% 15%, rgba(241,204,109,.28), transparent 22%),
        linear-gradient(180deg, #cfe6bf 0%, #dff0ce 100%);
    border-right: 1px solid rgba(32,49,39,.06);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] p { color: #1f5137 !important; }
section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }

.sidebar-brand {
    background: #f4f4e7;
    padding: 20px;
    border-radius: 24px;
    box-shadow: 10px 10px 18px var(--shadow-dark), -8px -8px 16px var(--shadow-light);
    margin-bottom: 18px;
}
.sidebar-brand .brand-title {
    font-family: "Baloo 2", sans-serif;
    font-size: 1.45rem; font-weight: 800;
    color: var(--green-dark); line-height: 1.0;
}
.sidebar-brand .brand-sub {
    font-size: .78rem; color: #2d4d33;
    font-weight: 600; margin-top: 4px;
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

input, textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}
input::placeholder, textarea::placeholder {
    color: #d7e6dc !important;
    -webkit-text-fill-color: #d7e6dc !important;
    opacity: 1;
}

div[data-baseweb="select"] { color: #1f5137 !important; }
div[data-baseweb="select"] > div {
    background-color: #24262d !important;
    border-color: rgba(31,81,55,.20) !important;
}
div[data-baseweb="select"] [data-baseweb="select"] *,
div[data-baseweb="select"] input,
div[data-baseweb="select"] [role="button"],
div[data-baseweb="select"] [aria-selected="true"] {
    color: #e6f4ec !important;
    -webkit-text-fill-color: #e6f4ec !important;
}
ul[data-baseweb="menu"] li, div[role="option"] {
    color: #1f5137 !important;
    background-color: #f7f7e9 !important;
}
div[role="option"] span { color: #1f5137 !important; }

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


def make_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create 14-feature table for clustering (matches notebook logic)"""
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
def transport_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract transportation mode data"""
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

    fig.update_layout(
        font=dict(
            family="DM Sans, sans-serif",
            color=green
        ),

        title_font=dict(
            family="Baloo 2, sans-serif",
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
def build_monthly_series(df: pd.DataFrame) -> pd.Series:
    """Build monthly tourism series — sama dengan logic forecasting.ipynb."""
    
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


# =========================================================
# DATA & MODEL LOADING
# =========================================================
if not DATA_PATH.exists():
    st.error(f"Dataset tidak ditemukan.\n`{DATA_PATH}`")
    st.stop()

df = load_data(str(DATA_PATH))

# Initialize clustering components
clustering_ok = MODEL_PATH.exists() and SCALER_PATH.exists()
cluster_features = pd.DataFrame()
cluster_labels = np.array([])
cluster_features_with_label = pd.DataFrame()
if clustering_ok:
    try:
        model, scaler = load_clustering_models(str(MODEL_PATH), str(SCALER_PATH))
        cluster_features = make_feature_table(df)
        X_scaled = scaler.transform(cluster_features)
        cluster_labels = model.predict(X_scaled)
        cluster_features_with_label = cluster_features.copy()
        cluster_features_with_label["Cluster"] = cluster_labels
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
    '<div class="sidebar-brand">'
    '<div class="brand-title">Nusantara<br>Transit Intelligence</div>'
    '<div class="brand-sub">Tourism mobility analytics • 2016–2026</div>'
    "</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigasi",
    [
        "Dashboard",
        "Cluster Explorer", 
        "Forecasting",
        "Data Explorer",
        "Transport Insights",
        "Tentang Model",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:.78rem;color:#2d4d33;font-weight:600;line-height:1.55;">'
    "<b>Tema:</b> Claymorphism Tropical<br>"
    "<b>Moda:</b> Udara • Darat • Laut<br>"
    "<b>Clustering:</b> K-Means (k=2)<br>"
    "<b>Forecasting:</b> SARIMA(0,1,1)(0,1,1,12)"
    "</div>",
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
    
    # Transport illustration via safe HTML component
    components.html(
        f'<div style="margin-top:-55px;text-align:right;pointer-events:none;padding-right:28px;">'
        f'<div style="display:inline-block;width:44%;min-width:280px;">'
        f'{get_transport_svg()}'
        f"</div></div>",
        height=190,
        scrolling=False,
    )
    # KPI snapshot
    individual = df[df["Jenis Data"].isin(["Pintu Masuk", "Pintu Masuk Lainnya"])].copy()
    annual = individual.groupby("Tahun")["Jumlah"].sum(min_count=1).reset_index()
    latest_year = int(annual["Tahun"].max())
    latest_total = float(annual.loc[annual["Tahun"] == latest_year, "Jumlah"].iloc[0])
    total_period = float(individual["Jumlah"].sum())
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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig_rt, use_container_width=True)

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
        st.plotly_chart(fig_season, use_container_width=True)
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
            use_container_width=True, hide_index=True,
        )


# =========================================================
# PAGE: FORECASTING
# =========================================================
elif page == "Forecasting":
    st.markdown(
        '<div class="clay-card hero">'
        '<div class="eyebrow">Forecasting Intelligence</div>'
        "<h1>Melihat Arah Perjalanan<br>Wisata Indonesia</h1>"
        "<p>Prediksi jumlah kunjungan wisatawan mancanegara menggunakan "
        "model time-series terbaik.</p>"
        '<div style="margin-top:16px;">'
        '<span class="transport-pill">✈ Udara</span>'
        '<span class="transport-pill">🚌 Darat</span>'
        '<span class="transport-pill">⛴ Laut</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    if not sarima_ok or sarima_artifact is None:
        st.warning(
            "**Model forecasting belum ditemukan.**\n\n"
            "Export `tourism_sarima_model.pkl` dari `forecasting.ipynb` "
            "dan simpan di folder `model/`."
        )
        st.stop()

    try:
        final_fit = sarima_artifact["model"]
        model_name_str = sarima_artifact.get("model_name", "SARIMA(0,1,1)(0,1,1,12)")
        training_start = sarima_artifact.get("training_start", "2016-01-01")
        training_end = sarima_artifact.get("training_end", "2026-07-01")
        forecast_result = final_fit.get_forecast(steps=12)
        future_forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        lower_col, upper_col = conf_int.columns[0], conf_int.columns[1]
    except Exception as e:
        st.error(f"Gagal menjalankan forecast: {e}")
        st.stop()

    monthly_known = build_monthly_series(df)

    # 1. Overview cards
    st.markdown('<div class="section-title">Ringkasan Model</div>', unsafe_allow_html=True)
    ts_start = pd.Timestamp(training_start)
    ts_end = pd.Timestamp(training_end)
    ov_items = [
        ("Forecast Model", "SARIMA"),
        ("Spesifikasi", model_name_str),
        ("Horizon Forecast", "12 bulan"),
        ("Periode Prediksi", "Agu 2026 – Jul 2027"),
        ("Data Historis", f"{fmt_date_id(ts_start)} – {fmt_date_id(ts_end)}"),
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
    # 2. KPI cards
    st.markdown('<div class="section-title">Key Forecast Metrics</div>', unsafe_allow_html=True)
    avg_fc = future_forecast.mean()
    max_fc = future_forecast.max()
    min_fc = future_forecast.min()
    peak_m_fc = future_forecast.idxmax()
    low_m_fc = future_forecast.idxmin()

    kpi5 = st.columns(5)
    with kpi5[0]:
        metric_card("Rata-rata Prediksi", fmt_number(avg_fc), "per bulan")
    with kpi5[1]:
        metric_card("Prediksi Tertinggi", fmt_number(max_fc), fmt_date_id(peak_m_fc))
    with kpi5[2]:
        metric_card("Prediksi Terendah", fmt_number(min_fc), fmt_date_id(low_m_fc))
    with kpi5[3]:
        metric_card("Bulan Puncak", fmt_date_id(peak_m_fc), "puncak musiman")
    with kpi5[4]:
        metric_card("Horizon Prediksi", "12 bulan", "Agu 2026 – Jul 2027")

    # 3. Main chart
    st.markdown('<div class="section-title">Grafik Forecast SARIMA</div>', unsafe_allow_html=True)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=monthly_known.index, y=monthly_known.values,
        name="Data Aktual", line=dict(color="#1f5137", width=2.5), mode="lines",
    ))
    ci_x = list(future_forecast.index) + list(future_forecast.index[::-1])
    ci_y = list(conf_int[upper_col]) + list(conf_int[lower_col][::-1])
    fig_fc.add_trace(go.Scatter(
        x=ci_x, y=ci_y, fill="toself",
        fillcolor="rgba(141,195,213,0.22)",
        line=dict(color="rgba(141,195,213,0)"),
        name="CI 95%", showlegend=True,
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_forecast.index, y=future_forecast.values,
        name="Forecast SARIMA",
        line=dict(color="#e8995d", width=2.5, dash="dash"),
        mode="lines+markers", marker=dict(size=6, color="#e8995d"),
    ))
    boundary_ts = monthly_known.index[-1]
    fig_fc.add_vline(
        x=boundary_ts.value / 1e6,
        line_dash="dot", line_color="#1f5137", line_width=1.5,
        annotation_text="Mulai Forecast", annotation_position="top",
        annotation_font=dict(color="#1f5137", size=12),
    )
    fig_fc.update_layout(
        margin=dict(l=0,r=0,t=30,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.22)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Tanggal"), yaxis=dict(title="Jumlah wisatawan"),
    )
    style_plotly(fig_fc)
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown(
        '<div class="note-box" style="margin-bottom:16px;">'
        "<b>Catatan:</b> Data aktual tersedia sampai Juli 2026. "
        "Agustus 2026 – Juli 2027 merupakan hasil prediksi SARIMA, "
        "<i>bukan</i> data aktual yang terukur."
        "</div>",
        unsafe_allow_html=True,
    )

    # 4. Forecast table
    st.markdown('<div class="section-title">Tabel Hasil Forecast</div>', unsafe_allow_html=True)
    forecast_df = pd.DataFrame({
        "Tanggal": [fmt_date_id(ts) for ts in future_forecast.index],
        "Forecast": [fmt_number_id(v) for v in future_forecast.values],
        "Lower Bound (95%)": [fmt_number_id(v) for v in conf_int[lower_col].values],
        "Upper Bound (95%)": [fmt_number_id(v) for v in conf_int[upper_col].values],
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    # 5. Model comparison table
    st.markdown('<div class="section-title">Perbandingan Model</div>', unsafe_allow_html=True)
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
    st.plotly_chart(fig_mape, use_container_width=True)

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
        use_container_width=True, hide_index=True,
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
    st.plotly_chart(fig_tr, use_container_width=True)
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
    st.plotly_chart(fig_bar, use_container_width=True)

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
        st.plotly_chart(fig_rec, use_container_width=True)

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
        st.dataframe(feature_desc, use_container_width=True, hide_index=True)

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
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

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
    st.dataframe(fc_eval_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="note-box">'
        "<b>Catatan metodologis:</b> Model clustering menggunakan 14 feature yang sama "
        "dengan proses fitting. PCA hanya digunakan untuk visualisasi 2D. "
        "Model SARIMA tidak dilatih ulang setiap kali halaman dibuka; "
        "forecast dihitung dari model tersimpan di <code>model/tourism_sarima_model.pkl</code>."
        "</div>",
        unsafe_allow_html=True,
    )