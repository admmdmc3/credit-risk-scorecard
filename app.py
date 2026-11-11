"""
Credit Risk Scorecard — Kurumsal Bankacılık Karar Destek Sistemi
================================================================
Streamlit Uygulaması | Frost Glass Banking Teması | Türkçe Arayüz
Çalıştırma: streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit as st
import streamlit.components.v1 as components

# ──────────────────────────────────────────────────────────────────────
# SAYFA YAPISI & TEMA
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kredi Risk Karar Motoru",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Frost Glass Banking CSS Teması (Açık Tema, Net Renk Hiyerarşisi)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #111928;
}

/* Ana Sayfa ve Üst Bar */
[data-testid="stHeader"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
}

div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.08) !important;
    border-left: 4px solid #3B82F6 !important;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #64748B !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* Sekmeler */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    background-color: transparent;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 0 4px 0;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0px !important;
    padding: 8px 14px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #3B82F6 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #3B82F6 !important;
}

h1, h2, h3, h4 {
    color: #0F172A !important;
    font-weight: 600 !important;
}
p, .stMarkdown p {
    color: #374151 !important;
}
label {
    color: #374151 !important;
}

/* Bölüm Başlıkları */
.section-header {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #64748B !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin-top: 14px;
    margin-bottom: 8px;
}

/* Executive Top Bar */
.exec-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #0F172A !important;
    color: #F1F5F9 !important;
    padding: 12px 24px !important;
    border-radius: 0 0 12px 12px !important;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
.exec-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #F1F5F9 !important;
    text-transform: uppercase;
}
.exec-badges {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}
.exec-badge {
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    display: inline-flex;
    align-items: center;
}
.exec-badge-green {
    border: 1px solid rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.15);
    color: #10B981 !important;
}
.exec-badge-blue {
    border: 1px solid rgba(59, 130, 246, 0.4);
    background: rgba(59, 130, 246, 0.15);
    color: #60A5FA !important;
}
.exec-badge-gray {
    border: 1px solid rgba(148, 163, 184, 0.4);
    background: rgba(148, 163, 184, 0.15);
    color: #CBD5E1 !important;
}

/* Metrik Kartları */
.kpi-card-custom {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 8px rgba(0, 0, 0, 0.08) !important;
    margin-bottom: 12px !important;
}
.kpi-custom-label {
    font-size: 11px !important;
    color: #64748B !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}
.kpi-custom-value {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    line-height: 1.1 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.kpi-custom-sub {
    font-size: 12px !important;
    color: #64748B !important;
    margin-top: 4px !important;
}

/* Cut-off Metrik Kartları */
.cutoff-metric-tile {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border-left: 4px solid #3B82F6 !important;
    padding: 16px 20px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 8px rgba(0, 0, 0, 0.06) !important;
}
.cutoff-metric-label {
    font-size: 11px !important;
    color: #64748B !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}
.cutoff-metric-val {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    line-height: 1.1 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.cutoff-metric-sub {
    font-size: 12px !important;
    color: #64748B !important;
    margin-top: 4px !important;
}

/* ── GLIDE DATA GRID (st.dataframe) & KURUMSAL HTML TABLOLAR ── */
:root,
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {
    --gdg-bg-cell: #FFFFFF !important;
    --gdg-bg-cell-medium: #F8FAFC !important;
    --gdg-bg-header: #F1F5F9 !important;
    --gdg-bg-header-has-focus: #E2E8F0 !important;
    --gdg-bg-header-hovered: #E2E8F0 !important;
    --gdg-text-dark: #0F172A !important;
    --gdg-text-medium: #334155 !important;
    --gdg-text-light: #475569 !important;
    --gdg-text-header: #0F172A !important;
    --gdg-text-group-header: #0F172A !important;
    --gdg-border-color: #E2E8F0 !important;
    --gdg-horizontal-border-color: #E2E8F0 !important;
    --gdg-accent-color: #2563EB !important;
}
[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
[data-testid="stDataFrame"] td {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    font-size: 13px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
iframe[title="streamlit_aggrid"] * {
    color: #0F172A !important;
    background-color: #FFFFFF !important;
}

.corp-table-wrap {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow-x: auto !important;
    overflow-y: auto !important;
    max-height: 440px !important;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04) !important;
    margin-bottom: 8px !important;
}
.corp-table,
[data-testid="stExpander"] .corp-table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    background-color: #FFFFFF !important;
    font-family: 'Inter', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    color: #0F172A !important;
    margin: 0 !important;
    table-layout: auto !important;
}
.corp-table thead,
.corp-table tbody,
.corp-table tr {
    height: auto !important;
}
.corp-table thead th,
[data-testid="stExpander"] .corp-table thead th {
    position: static !important;
    line-height: 1.4 !important;
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    text-align: left !important;
    padding: 11px 14px !important;
    border-bottom: 1px solid #E2E8F0 !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    white-space: nowrap !important;
}
.corp-table tbody td,
[data-testid="stExpander"] .corp-table tbody td {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    padding: 10px 14px !important;
    line-height: 1.4 !important;
    border-bottom: 1px solid #E2E8F0 !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    vertical-align: middle !important;
}
.corp-table tbody tr:last-child td,
[data-testid="stExpander"] .corp-table tbody tr:last-child td {
    border-bottom: none !important;
}
.corp-table tbody tr:hover td,
[data-testid="stExpander"] .corp-table tbody tr:hover td {
    background-color: #F8FAFC !important;
}
.corp-table th.num-col,
.corp-table td.num-col,
[data-testid="stExpander"] .corp-table th.num-col,
[data-testid="stExpander"] .corp-table td.num-col {
    text-align: right !important;
}
.corp-table td.num-col,
[data-testid="stExpander"] .corp-table td.num-col {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}

/* ── DROPDOWN / SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] * {
    color: #0F172A !important;
}
[data-baseweb="select"] * {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
[data-baseweb="popover"] * {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

/* ── GRAFİK ALTI YAZILARI VE GENEL YAZI RENKLERİ ── */
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
    font-size: 12px !important;
}
.element-container p,
.stMarkdown p,
.stText,
.stWrite,
[data-testid="stMarkdownContainer"] {
    color: #374151 !important;
}
small {
    color: #64748B !important;
}

/* ── KOMİTE DEĞERLENDİRME NOTU KARTI & ALERT KUTULARI ── */
.committee-note-card {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #2563EB !important;
    border-radius: 12px !important;
    padding: 18px 22px 22px 22px !important;
    margin-top: 8px !important;
    margin-bottom: 12px !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
    box-sizing: border-box !important;
    box-shadow: 0 1px 8px rgba(0, 0, 0, 0.06) !important;
}
.committee-note-title {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin-bottom: 10px !important;
    letter-spacing: 0.02em !important;
}
.committee-note-list {
    margin: 0 !important;
    padding-left: 18px !important;
    color: #0F172A !important;
}
.committee-note-list li {
    color: #0F172A !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
    margin-bottom: 4px !important;
}
[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #2563EB !important;
    color: #0F172A !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] li,
[data-testid="stAlert"] span {
    color: #0F172A !important;
}

/* ── PLOTLY GRAFİK YAZILARI (axis, tick, legend) ── */
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle,
.js-plotly-plot .plotly .legendtext {
    fill: #374151 !important;
}
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text {
    fill: #6B7280 !important;
}

/* ── EXPANDER (Tam IV, Dağılım & Kesme Eşiği Tabloları) ── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: #0F172A !important;
    fill: #0F172A !important;
}

/* ── "PARAMETRELERİ DEĞİŞTİR" & ANA EKRAN BUTONLARI ── */
[data-testid="stButton"] > button {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stButton"] > button p,
[data-testid="stButton"] > button span,
[data-testid="stButton"] > button svg,
[data-testid="stButton"] > button * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
[data-testid="stButton"] > button:hover {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
}

/* ── SIDEBAR: KOYU LACİVERT TEMA & TAM KONTRAST (#F8FAFC) ── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #0F172A !important;
    min-width: 220px !important;
    max-width: 220px !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] .element-container p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio p,
[data-testid="stSidebar"] .stSlider p,
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {
    color: #F8FAFC !important;
    fill: #F8FAFC !important;
}
[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] > div {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border-color: #334155 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
    fill: #F8FAFC !important;
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] .stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35) !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] button[kind="primary"] *,
[data-testid="stSidebar"] .stButton > button * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SABITLER & YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
DATA_PATH = Path("data/cs-training.csv")

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
SYSTEMIC_ERROR_CODES = {96, 98}


@st.cache_resource(show_spinner="Model yükleniyor…")
def load_artifacts():
    lgbm   = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    lr_pip = joblib.load(MODEL_DIR / "lr_pipeline.pkl")
    meta   = joblib.load(MODEL_DIR / "metadata.pkl")
    iv_df  = pd.read_csv(MODEL_DIR / "iv_table.csv")

    with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(MODEL_DIR / "validation_curves.json", "r", encoding="utf-8") as f:
        val_curves = json.load(f)

    shap_bg = pd.read_csv(MODEL_DIR / "shap_background.csv")
    return lgbm, lr_pip, meta, iv_df, metrics, val_curves, shap_bg


lgbm_model, lr_pipeline, metadata, iv_df, metrics, val_curves, shap_bg = load_artifacts()

SELECTED_FEATURES = metadata["selected_features"]
CLEANING_PARAMS   = metadata["cleaning_params"]
PDO_PARAMS        = metadata["pdo_params"]
FACTOR            = float(PDO_PARAMS["factor"])
OFFSET            = float(PDO_PARAMS["offset"])


# ── Temizlik & Kalibrasyon (train_model.py ile birebir aynı) ─────────
def apply_cleaning_single(row: dict) -> dict:
    """Tek satırlık dict girdisine deterministik cleaning uygula."""
    r = dict(row)

    # Age
    if r.get("age", 40) <= 18 or r.get("age", 40) > 95:
        r["age"] = CLEANING_PARAMS["age_median"]

    # Past-due systemic codes
    for col in PAST_DUE_COLS:
        if col in r and r[col] in SYSTEMIC_ERROR_CODES:
            r[col] = CLEANING_PARAMS[f"{col}_median"]

    # Revolving clip
    if "RevolvingUtilizationOfUnsecuredLines" in r:
        r["RevolvingUtilizationOfUnsecuredLines"] = min(
            r["RevolvingUtilizationOfUnsecuredLines"],
            CLEANING_PARAMS["revolving_clip_upper"],
        )

    # Defaults for missing
    if "MonthlyIncome" not in r or r["MonthlyIncome"] is None:
        r["MonthlyIncome"] = CLEANING_PARAMS["MonthlyIncome_median"]
    if "NumberOfDependents" not in r or r["NumberOfDependents"] is None:
        r["NumberOfDependents"] = CLEANING_PARAMS["NumberOfDependents_median"]

    return r


def calibrate_prob(p_raw: float, train_base_rate: float = 0.0668,
                   balanced_rate: float = 0.5) -> float:
    r = train_base_rate / balanced_rate
    return (p_raw * r) / (p_raw * r + (1 - p_raw) * (1 - r))


def pd_to_score(pd_cal: float) -> int:
    odds = (1 - pd_cal) / max(pd_cal, 1e-15)
    score = OFFSET + FACTOR * np.log(odds)
    return int(np.clip(score, 300, 850))


# ── Optimum cut-off (maks. P&L) — Sekme 1 ve Sekme 4 ortak kullanır ──
_cutoff_all_df = pd.DataFrame(val_curves["cutoff_analysis"])
_opt_cut_row = _cutoff_all_df.loc[_cutoff_all_df["expected_pl_units"].idxmax()]
OPT_CUTOFF_PD = float(_opt_cut_row["pd_threshold"])
OPT_CUTOFF_SCORE = int(pd_to_score(OPT_CUTOFF_PD))


def score_to_risk_group(score: int) -> tuple[str, str]:
    """(Risk Grubu, Renk kodu) döndür."""
    if score >= 750:
        return "A+", "#10B981"
    elif score >= 680:
        return "A",  "#10B981"
    elif score >= 600:
        return "B",  "#3B82F6"
    elif score >= 520:
        return "C",  "#F59E0B"
    return "D", "#EF4444"


def score_to_karar(score: int) -> tuple[str, str]:
    """(Karar metni, Renk kodu) döndür. Eşikler: optimum cut-off skoru ve 600."""
    approve_thr = max(600, OPT_CUTOFF_SCORE)
    if score >= approve_thr:
        return "ONAY", "#10B981"
    elif score >= OPT_CUTOFF_SCORE:
        return "TEMİNATLI İNCELEME", "#F59E0B"
    return "RET", "#EF4444"


def predict_customer(input_dict: dict):
    """Temizle → tahmin → kalibre et → PDO skoru üret."""
    cleaned = apply_cleaning_single(input_dict)
    X = pd.DataFrame([cleaned])[SELECTED_FEATURES].astype(float)
    p_raw   = float(lgbm_model.predict_proba(X)[0, 1])
    p_cal   = calibrate_prob(p_raw)
    score   = pd_to_score(p_cal)
    return X, p_cal, score


# ── SHAP açıklayıcısı (önbelleğe al) ─────────────────────────────────
@st.cache_resource(show_spinner="SHAP açıklayıcısı hazırlanıyor…")
def get_shap_explainer():
    return shap.TreeExplainer(lgbm_model, shap_bg)


# ── Türkçe değişken etiketleri ────────────────────────────────────────
TR_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines" : "Kredi Kullanım Oranı",
    "NumberOfTimes90DaysLate"              : "90+ Gün Gecikmeli Ödeme Sayısı",
    "NumberOfTime30-59DaysPastDueNotWorse" : "30-59 Gün Gecikmeli Ödeme Sayısı",
    "NumberOfTime60-89DaysPastDueNotWorse" : "60-89 Gün Gecikmeli Ödeme Sayısı",
    "age"                                  : "Yaş",
    "DebtRatio"                            : "Borç Oranı",
    "MonthlyIncome"                        : "Aylık Gelir ($)",
    "NumberOfOpenCreditLinesAndLoans"      : "Açık Kredi Hattı / Kredi Sayısı",
    "NumberOfDependents"                   : "Bakmakla Yükümlü Kişi Sayısı",
    "NumberRealEstateLoansOrLines"         : "Konut / Gayrimenkul Kredi Sayısı",
}
FEATURE_LABELS = TR_LABELS

# ── Hazır müşteri profilleri ──────────────────────────────────────────
PROFILES = {
    "Düşük Riskli VIP": {
        "RevolvingUtilizationOfUnsecuredLines" : 0.08,
        "NumberOfTimes90DaysLate"              : 0,
        "NumberOfTime30-59DaysPastDueNotWorse" : 0,
        "NumberOfTime60-89DaysPastDueNotWorse" : 0,
        "age"                                  : 48,
        "DebtRatio"                            : 0.18,
        "MonthlyIncome"                        : 9500,
        "NumberOfOpenCreditLinesAndLoans"      : 7,
        "NumberOfDependents"                   : 1,
    },
    "Gri Bölge / İnceleme": {
        "RevolvingUtilizationOfUnsecuredLines" : 0.08,
        "NumberOfTimes90DaysLate"              : 0,
        "NumberOfTime30-59DaysPastDueNotWorse" : 1,
        "NumberOfTime60-89DaysPastDueNotWorse" : 0,
        "age"                                  : 48,
        "DebtRatio"                            : 0.18,
        "MonthlyIncome"                        : 9500,
        "NumberOfOpenCreditLinesAndLoans"      : 7,
        "NumberOfDependents"                   : 1,
    },
    "Yüksek Riskli": {
        "RevolvingUtilizationOfUnsecuredLines" : 0.95,
        "NumberOfTimes90DaysLate"              : 5,
        "NumberOfTime30-59DaysPastDueNotWorse" : 7,
        "NumberOfTime60-89DaysPastDueNotWorse" : 4,
        "age"                                  : 27,
        "DebtRatio"                            : 1.20,
        "MonthlyIncome"                        : 1800,
        "NumberOfOpenCreditLinesAndLoans"      : 14,
        "NumberOfDependents"                   : 4,
    },
}

LIGHT_TEMPLATE = "plotly_white"
PLOTLY_LAYOUT = dict(
    template=LIGHT_TEMPLATE,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    font=dict(color="#374151", size=12, family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def render_corporate_table(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> None:
    """DataFrame verisini açık gri başlık, beyaz satır ve #0F172A koyu metinli HTML tablosu olarak çizer."""
    num_set = set(numeric_cols or [])
    headers = "".join(
        f'<th class="{"num-col" if col in num_set else ""}">{html.escape(str(col))}</th>'
        for col in df.columns
    )
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(
            f'<td class="{"num-col" if col in num_set else ""}">{html.escape(str(row[col]))}</td>'
            for col in df.columns
        )
        rows_html.append(f"<tr>{cells}</tr>")
    body = "".join(rows_html)
    st.markdown(
        f'<div class="corp-table-wrap"><table class="corp-table"><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — Müşteri Giriş Formu
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-size:14px; font-weight:700; color:#F8FAFC !important; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:12px;">KREDİ BAŞVURU FORMU</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Hızlı profil seçimi
    st.markdown('<div style="font-size:12px; font-weight:600; color:#F8FAFC !important; margin-bottom:6px;">Hızlı Profil Seç:</div>', unsafe_allow_html=True)

    def _profile_value(feat: str, prof: dict):
        v = prof.get(feat, 0)
        if feat in ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio"):
            return float(round(v, 2))
        if feat == "MonthlyIncome":
            return int(v)
        if feat in ("age", "NumberOfOpenCreditLinesAndLoans", "NumberOfDependents") or feat in PAST_DUE_COLS:
            return int(v)
        return float(v)

    def _apply_profile():
        """Profil değişince tüm giriş widget state'lerini profille senkronize et."""
        prof = PROFILES[st.session_state["profile_sel"]]
        for f in SELECTED_FEATURES:
            st.session_state[f"inp_{f}"] = _profile_value(f, prof)
        st.session_state["profile_changed"] = True

    def _matches_profile(cur_inputs: dict, prof: dict) -> bool:
        """Mevcut giriş değerlerinin seçili hazır profil ile birebir eşleşip eşleşmediğini kontrol et."""
        for f in SELECTED_FEATURES:
            if f not in cur_inputs:
                return False
            cur_val = cur_inputs[f]
            prof_val = _profile_value(f, prof)
            if isinstance(prof_val, int):
                if int(round(float(cur_val))) != prof_val:
                    return False
            else:
                if abs(float(cur_val) - float(prof_val)) > 1e-6:
                    return False
        return True

    if "profile_sel" not in st.session_state:
        st.session_state["profile_sel"] = list(PROFILES.keys())[0]
    for _f in SELECTED_FEATURES:
        if f"inp_{_f}" not in st.session_state:
            st.session_state[f"inp_{_f}"] = _profile_value(_f, PROFILES[st.session_state["profile_sel"]])

    selected_profile = st.radio(
        "Profil",
        list(PROFILES.keys()),
        key="profile_sel",
        on_change=_apply_profile,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Giriş alanları (selected_features sırası)
    inputs: dict = {}
    for feat in SELECTED_FEATURES:
        label = TR_LABELS.get(feat, feat)
        wkey = f"inp_{feat}"

        if feat == "age":
            inputs[feat] = st.slider(label, 18, 95, key=wkey)
        elif feat in PAST_DUE_COLS or feat in (
            "NumberOfOpenCreditLinesAndLoans", "NumberOfDependents"
        ):
            inputs[feat] = st.slider(label, 0, 20, key=wkey)
        elif feat == "MonthlyIncome":
            inputs[feat] = st.number_input(label, min_value=0, max_value=1000000,
                                            step=100, key=wkey)
        elif feat == "RevolvingUtilizationOfUnsecuredLines":
            inputs[feat] = st.slider(label, 0.0, 1.0, step=0.01, key=wkey)
        elif feat == "DebtRatio":
            inputs[feat] = st.slider(label, 0.0, 2.0, step=0.01, key=wkey)
        else:
            inputs[feat] = st.number_input(label, key=wkey)

    st.markdown("---")
    predict_btn = st.button("Kredi Skoru Hesapla", type="primary", use_container_width=True)

# ── KALICI HAFIZA (st.session_state) & OTOMATİK HESAPLAMA ─────────────
_profile_changed = st.session_state.pop("profile_changed", False)
_inputs_changed = st.session_state.get("current_inputs") != inputs
_matched_profile_label = (
    selected_profile
    if _matches_profile(inputs, PROFILES[selected_profile])
    else "Özel Senaryo (Manuel)"
)
if "score_calculated" not in st.session_state or predict_btn or _profile_changed or _inputs_changed:
    X_calc, pd_calc, score_calc = predict_customer(inputs)
    st.session_state.score_calculated = True
    st.session_state.X_input = X_calc
    st.session_state.pd_cal = pd_calc
    st.session_state.credit_score = score_calc
    st.session_state.current_inputs = dict(inputs)
st.session_state.active_profile = _matched_profile_label

if predict_btn:
        components.html(
            """
            <script>
            window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"] button, [data-testid="collapsedControl"] button')?.click();
            </script>
            """,
            height=0,
        )

# ── EXECUTIVE TOP BAR (Header Bar) ───────────────────────────────────
st.markdown(f"""
<div class="exec-top-bar">
    <div class="exec-title">
        FINRISK PRO // KREDİ RİSK SKOR KARTI &amp; VALİDASYON MOTORU
    </div>
    <div class="exec-badges">
        <span class="exec-badge exec-badge-green">Basel III / IFRS 9 Metodolojisi</span>
        <span class="exec-badge exec-badge-blue">Motor: LightGBM v2.1 (GINI: {metrics['LightGBM']['Test']['GINI']:.3f})</span>
        <span class="exec-badge exec-badge-gray">PSI: {metrics['PSI']['LightGBM']:.4f} (Stable)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SEKMELERİ TANIMLA (Underline Stil)
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Canlı Kredi Karar Motoru",
    "EDA & WoE / IV Raporu",
    "Model Validasyonu & Kalibrasyon",
    "Cut-off Optimizasyonu & P&L",
])

# ═══════════════════════════════════════════════════════════════════════
# SEKME 1 — CANLI KREDİ KARAR MOTORU
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    head_col1, head_col2 = st.columns([1.3, 1.7])
    with head_col1:
        st.markdown('<div class="section-header">CANLI KREDİ KARAR MOTORU &amp; PDO SKOR KARTI</div>', unsafe_allow_html=True)
    with head_col2:
        _active_prof = html.escape(str(st.session_state.get("active_profile", selected_profile)))
        st.markdown(
            f'<div style="display:flex; justify-content:flex-end;">'
            f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #2563EB; border-radius:10px; '
            f'padding:8px 16px; font-size:12px; line-height:1.5; color:#0F172A; width:fit-content; max-width:100%; '
            f'box-shadow:0 1px 4px rgba(15,23,42,0.05);">'
            f'<div style="font-weight:700; color:#0F172A; white-space:nowrap;">Aktif Profil: {_active_prof}</div>'
            f'<div style="font-family:\'JetBrains Mono\', monospace; color:#475569; white-space:nowrap;">'
            f'Optimum Cut-off Skoru: {OPT_CUTOFF_SCORE} | PD Eşiği: %{int(round(OPT_CUTOFF_PD * 100))}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    X_input = st.session_state.X_input
    pd_cal = st.session_state.pd_cal
    credit_score = st.session_state.credit_score

    risk_group, rg_color = score_to_risk_group(credit_score)
    karar_text, karar_color = score_to_karar(credit_score)

    # ── 3-Column Grid Metrik Kartları ─────────────────────────────────
    c1, c2, c3 = st.columns(3)

    # PDO kartı: #3B82F6
    pdo_border = "#3B82F6"

    # PD kartı: #F59E0B
    pd_border = "#F59E0B"
    pd_pct = pd_cal * 100

    # Karar kartı: yeşil ise #10B981, kırmızı ise #EF4444, inceleme ise #F59E0B
    if "ONAY" in karar_text and "TEMİNAT" not in karar_text:
        karar_border = "#10B981"
        karar_val_color = "#10B981"
    elif "İNCELEME" in karar_text:
        karar_border = "#F59E0B"
        karar_val_color = "#F59E0B"
    else:
        karar_border = "#EF4444"
        karar_val_color = "#EF4444"

    with c1:
        st.markdown(f"""
        <div class="kpi-card-custom" style="border-left: 4px solid {pdo_border};">
            <div class="kpi-custom-label">PDO KREDİ SKORU</div>
            <div class="kpi-custom-value" style="color: #0F172A;">{credit_score}</div>
            <div class="kpi-custom-sub">300 – 850 Bandı</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card-custom" style="border-left: 4px solid {pd_border};">
            <div class="kpi-custom-label">KALİBRE EDİLMİŞ PD (%)</div>
            <div class="kpi-custom-value" style="color: #0F172A;">{pd_pct:.2f}%</div>
            <div class="kpi-custom-sub">24 Aylık (2 Yıllık) Temerrüt Olasılığı</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card-custom" style="border-left: 4px solid {karar_border};">
            <div class="kpi-custom-label">KOMİTE KARARI &amp; RİSK GRUBU</div>
            <div class="kpi-custom-value" style="color: {karar_val_color}; font-size: 26px;">{karar_text}</div>
            <div class="kpi-custom-sub">Risk Grubu: <strong>{risk_group}</strong> • LightGBM + PDO Motoru</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge (İbre) Grafiği ─────────────────────────────────────
    col_gauge, col_shap = st.columns([1, 1.6])
    with col_gauge:
        # Skora göre dinamik bar rengi: RET < cut-off <= İnceleme < 600 <= ONAY
        if credit_score < OPT_CUTOFF_SCORE:
            gauge_bar_color = "#EF4444"
        elif credit_score < 600:
            gauge_bar_color = "#F59E0B"
        else:
            gauge_bar_color = "#10B981"

        # İbre renk skalası: ['#E02424','#F05252','#FACA15','#84E1BC','#057A55']
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=credit_score,
            delta={
                "reference": OPT_CUTOFF_SCORE,
                "increasing": {"color": "#10B981"},
                "decreasing": {"color": "#EF4444"},
            },
            title={"text": "<span style='font-size:11px;font-weight:600;color:#64748B;letter-spacing:0.05em;'>PDO KREDİ SKORU</span><br><span style='font-size:11px;color:#64748B'>300 (Yüksek Risk) → 850 (Düşük Risk)</span>",
                   "font": dict(size=14, color="#374151", family="Inter, sans-serif")},
            gauge={
                "axis": dict(
                    range=[300, 850],
                    tickvals=[300, 400, 500, 600, 700, 800, 850],
                    tickcolor="#6B7280",
                    tickfont=dict(color="#6B7280", size=11),
                    tickwidth=1,
                ),
                "bar": {"color": gauge_bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,1)",
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [300, 410], "color": "#E02424"},
                    {"range": [410, 520], "color": "#F05252"},
                    {"range": [520, 630], "color": "#FACA15"},
                    {"range": [630, 740], "color": "#84E1BC"},
                    {"range": [740, 850], "color": "#057A55"},
                ],
                "threshold": {
                    "line": {"color": "#3B82F6", "width": 3},
                    "thickness": 0.85,
                    "value": OPT_CUTOFF_SCORE,
                },
            },
            number={"font": {"size": 46, "color": gauge_bar_color, "family": "JetBrains Mono"}},
        ))
        fig_gauge.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            height=320,
            margin=dict(t=75, b=20, l=30, r=30),
            font=dict(color="#374151", family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Otomatik Onay Önerisi Metin Kutusu (Yeşil / Kırmızı / Sarı)
        if "ONAY" in karar_text and "TEMİNAT" not in karar_text:
            box_bg = "#ECFDF5"
            box_border = "#10B981"
            box_color = "#065F46"
            box_title = "Otomatik Onay Önerisi:"
            box_msg = "Müşteri profili düşük risk bantta yer almaktadır. Standart fiyatlandırma ile onaylanabilir."
        elif "İNCELEME" in karar_text:
            box_bg = "#FFFBEB"
            box_border = "#F59E0B"
            box_color = "#92400E"
            box_title = "Teminatlı İnceleme Önerisi:"
            box_msg = "Müşteri profili gri bölgede yer almaktadır. Ek teminat veya gelir belgesi talep edilmelidir."
        else:
            box_bg = "#FEF2F2"
            box_border = "#EF4444"
            box_color = "#991B1B"
            box_title = "Ret Önerisi:"
            box_msg = "Müşteri profili yüksek risk bantta yer almaktadır. Mevcut kredi politikası çerçevesinde tahsis önerilmez."

        st.markdown(f"""
        <div style="background: {box_bg}; border-left: 4px solid {box_border}; color: {box_color}; border-radius: 8px; padding: 16px; margin-top: 14px; font-size: 13.5px; line-height: 1.5;">
            <strong>{box_title}</strong> {box_msg}
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP Waterfall ───────────────────────────────────────────
    with col_shap:
        st.markdown('<div class="section-header">SKOR KATKI ANALİZİ (SHAP)</div>', unsafe_allow_html=True)
        with st.spinner("SHAP değerleri hesaplanıyor…"):
            explainer = get_shap_explainer()
            shap_vals = explainer.shap_values(X_input)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            else:
                sv = shap_vals[0]

            feat_names = SELECTED_FEATURES
            sv_df = pd.DataFrame({
                "Değişken": [TR_LABELS.get(f, f) for f in feat_names],
                "SHAP": sv,
            }).sort_values("SHAP")

            # Hedef = Temerrüt (1): pozitif SHAP riski artırır (kırmızı), negatif azaltır (yeşil)
            colors = ["#EF4444" if v > 0 else "#10B981" for v in sv_df["SHAP"]]

            shap_lo = float(min(0.0, sv_df["SHAP"].min()))
            shap_hi = float(max(0.0, sv_df["SHAP"].max()))
            shap_span = (shap_hi - shap_lo) or 1.0
            shap_range = [shap_lo - 0.25 * shap_span, shap_hi + 0.25 * shap_span]

            fig_shap = go.Figure(go.Bar(
                x=sv_df["SHAP"],
                y=sv_df["Değişken"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in sv_df["SHAP"]],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(color="#374151", size=11, family="JetBrains Mono"),
            ))
            fig_shap.update_layout(
                template="plotly_white",
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
                title=dict(text="🔴 Kırmızı: Risk Artırıcı Etki (+ Log-Odds / PD)  |  🟢 Yeşil: Risk Azaltıcı Etki (- Log-Odds / PD)",
                           font=dict(size=11, color="#64748B", family="Inter, sans-serif")),
                xaxis_title="SHAP Katkısı (log-odds)",
                xaxis=dict(range=shap_range, automargin=True, color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
                yaxis=dict(automargin=True, color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(size=11, color="#374151")),
                font=dict(color="#374151", family="Inter, sans-serif"),
                height=360,
                margin=dict(l=20, r=60, t=50, b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            top4_df = (
                sv_df.assign(abs_val=sv_df["SHAP"].abs())
                     .sort_values("abs_val", ascending=False)
                     .head(4)
                     .reset_index(drop=True)
            )
            note_items_html = []
            for rank, (_, row_shap) in enumerate(top4_df.iterrows(), start=1):
                is_pos = row_shap["SHAP"] > 0
                direction = "Risk Artırıcı (+)" if is_pos else "Risk Azaltıcı (-)"
                badge_color = "#DC2626" if is_pos else "#059669"
                var_name = html.escape(str(row_shap["Değişken"]))
                note_items_html.append(
                    f'<li><strong style="color:#0F172A !important;">{rank}. {var_name}:</strong> '
                    f'<span style="color:{badge_color} !important; font-weight:600;">{direction}</span> '
                    f'yönde etki (<code style="font-family:\'JetBrains Mono\', monospace; color:#0F172A !important; background:#F1F5F9; padding:1px 6px; border-radius:4px;">{row_shap["SHAP"]:+.4f}</code> log-odds)</li>'
                )
            items_str = "".join(note_items_html)
            st.markdown(f"""
            <div class="committee-note-card">
                <div class="committee-note-title">Kredi Tahsis Komitesi Otomatik Değerlendirme Notu</div>
                <ul class="committee-note-list" style="list-style:none; padding-left:0 !important;">{items_str}</ul>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SEKME 2 — EDA & WoE / IV RAPORU
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">KEŞİFSEL VERİ ANALİZİ &amp; WOE / IV GÜÇ RAPORU</div>', unsafe_allow_html=True)

    # ── IV Bar Grafiği ───────────────────────────────────────────────
    st.markdown('<div class="section-header">INFORMATION VALUE (IV) — DEĞİŞKEN TAHMİN GÜCÜ</div>', unsafe_allow_html=True)

    iv_sorted = iv_df.sort_values("IV")
    bar_colors = []
    for iv_val in iv_sorted["IV"]:
        if iv_val >= 0.30:
            bar_colors.append("#3B82F6")
        elif iv_val >= 0.10:
            bar_colors.append("#10B981")
        elif iv_val >= 0.02:
            bar_colors.append("#F59E0B")
        else:
            bar_colors.append("#EF4444")

    fig_iv = go.Figure()
    fig_iv.add_trace(go.Bar(
        x=iv_sorted["IV"],
        y=[TR_LABELS.get(f, f) for f in iv_sorted["Feature"]],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f} — {s}" for v, s in zip(iv_sorted["IV"], iv_sorted["Strength"])],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color="#374151", size=11, family="JetBrains Mono"),
    ))
    # Eşik çizgileri
    for x_val, label, color in [
        (0.02, "Zayıf Eşiği (0.02)", "#64748B"),
        (0.10, "Orta Eşiği (0.10)", "#3B82F6"),
        (0.30, "Güçlü Eşiği (0.30)", "#10B981"),
    ]:
        fig_iv.add_vline(x=x_val, line_dash="dash", line_color=color,
                         layer="below",
                         annotation_text=label,
                         annotation_font=dict(color=color, size=11))

    fig_iv.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        height=420,
        xaxis_title="Information Value (IV)",
        xaxis=dict(range=[0, float(iv_sorted["IV"].max()) * 1.22], color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
        yaxis=dict(color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
        font=dict(color="#374151", family="Inter, sans-serif"),
        margin=dict(l=10, r=180, t=20, b=10),
    )
    st.plotly_chart(fig_iv, use_container_width=True)

    # IV Tablosu
    with st.expander("Tam IV Tablosu"):
        iv_display_df = (
            iv_df.assign(
                Feature=iv_df["Feature"].map(lambda f: TR_LABELS.get(f, f)),
                IV=iv_df["IV"].map(lambda v: f"{float(v):.4f}"),
            )
            .rename(columns={"Feature": "Değişken", "IV": "IV Değeri", "Strength": "Güç Seviyesi"})
        )
        render_corporate_table(iv_display_df, numeric_cols=["IV Değeri"])

    st.markdown("---")

    # ── EDA: Değişken Dağılımı & Batık Oranı ─────────────────────────
    st.markdown('<div class="section-header">DEĞİŞKEN BAZLI DAĞILIM &amp; BATIK ORANI</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner="Veri okunuyor…")
    def load_training_data():
        df = pd.read_csv(DATA_PATH)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        return df

    @st.fragment
    def render_eda_fragment():
        raw_df = load_training_data()

        feat_sel = st.selectbox(
            "Analiz Edilecek Değişken:",
            options=SELECTED_FEATURES,
            format_func=lambda f: TR_LABELS.get(f, f),
            key="eda_feat_sel",
        )

        col_eda1, col_eda2 = st.columns(2)
        rate_df = None

        with col_eda1:
            if feat_sel == "DebtRatio":
                s_hist = raw_df.loc[raw_df[feat_sel] <= 2.0, feat_sel].dropna()
                hist_suffix = "≤ 2.0 Filtreli"
            else:
                s_hist = raw_df[feat_sel].clip(
                    raw_df[feat_sel].quantile(0.01),
                    raw_df[feat_sel].quantile(0.99),
                )
                hist_suffix = "%1-%99 Kırpılmış"

            tr_name = FEATURE_LABELS.get(feat_sel, feat_sel)
            fig_hist = px.histogram(
                x=s_hist, nbins=40,
                title=f"{tr_name} — Dağılım ({hist_suffix})",
                color_discrete_sequence=["#3B82F6"],
                template="plotly_white",
                labels={"x": tr_name, "y": "Gözlem Sayısı"},
            )
            discrete_int_cols = set(PAST_DUE_COLS) | {
                "NumberOfOpenCreditLinesAndLoans",
                "NumberOfDependents",
                "NumberRealEstateLoansOrLines",
            }
            hist_xaxis = dict(
                color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0",
                tickfont=dict(color="#374151"),
            )
            if feat_sel in discrete_int_cols:
                hist_xaxis["dtick"] = 1

            fig_hist.update_layout(
                template="plotly_white",
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
                height=320,
                xaxis_title=tr_name,
                yaxis_title="Gözlem Sayısı",
                xaxis=hist_xaxis,
                yaxis=dict(color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
                font=dict(color="#374151", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            fig_hist.update_traces(marker_line_color="#E2E8F0", marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_eda2:
            try:
                tmp_df = raw_df[[feat_sel, "SeriousDlqin2yrs"]].copy()
                # 96/98 sistemik hata kodları bant analizini bozmasın
                if feat_sel in PAST_DUE_COLS:
                    tmp_df = tmp_df[~tmp_df[feat_sel].isin(SYSTEMIC_ERROR_CODES)].copy()
                tmp_df = tmp_df.dropna(subset=[feat_sel])
                tmp_df["bin"] = pd.qcut(tmp_df[feat_sel], q=10, duplicates="drop")
                if tmp_df["bin"].nunique() < 3:
                    # Sıfır ağırlıklı değişkenler için kesikli bantlara geç
                    tmp_df["bin"] = pd.cut(
                        tmp_df[feat_sel],
                        bins=[-0.001, 0, 1, 2, 3, np.inf],
                        labels=["0 (Gecikmesiz)", "1 Kez", "2 Kez", "3 Kez", "4+ Kez"],
                    )
                rate_df = (tmp_df.groupby("bin", observed=False)["SeriousDlqin2yrs"]
                           .agg(["mean", "count"])
                           .reset_index())
                rate_df.columns = ["Bant", "Batık Oranı", "Gözlem Sayısı"]
                rate_df["Bant"] = rate_df["Bant"].astype(str)

                fig_dr = go.Figure()
                fig_dr.add_trace(go.Bar(
                    x=rate_df["Bant"], y=rate_df["Gözlem Sayısı"],
                    name="Gözlem", marker_color="rgba(59, 130, 246, 0.2)",
                    yaxis="y2",
                ))
                fig_dr.add_trace(go.Scatter(
                    x=rate_df["Bant"], y=rate_df["Batık Oranı"] * 100,
                    name="Batık Oranı (%)", mode="lines+markers",
                    marker=dict(color="#EF4444", size=8),
                    line=dict(color="#EF4444", width=2),
                ))
                max_dr = float((rate_df["Batık Oranı"] * 100).max()) if not rate_df.empty else 1.0
                max_cnt = float(rate_df["Gözlem Sayısı"].max()) if not rate_df.empty else 1.0
                max_dr_top = max(max_dr * 1.22, 1.0) if np.isfinite(max_dr) else 1.0
                max_cnt_top = max(max_cnt * 1.22, 1.0) if np.isfinite(max_cnt) else 1.0

                fig_dr.update_layout(
                    template="plotly_white",
                    paper_bgcolor='rgba(255,255,255,1)',
                    plot_bgcolor='rgba(255,255,255,1)',
                    title=f"{TR_LABELS.get(feat_sel, feat_sel)} — Bant Bazlı Batık Oranı",
                    height=320,
                    yaxis=dict(title="Batık Oranı (%)", tickformat=".1f", range=[0, max_dr_top], color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
                    yaxis2=dict(title="Gözlem Sayısı", overlaying="y", side="right", showgrid=False, range=[0, max_cnt_top], color="#6B7280", linecolor="#E2E8F0", tickfont=dict(color="#64748B")),
                    xaxis=dict(tickangle=-30, tickfont=dict(size=10, color="#374151"), automargin=True, color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0"),
                    font=dict(color="#374151", family="Inter, sans-serif"),
                    legend=dict(x=0, y=1.1, orientation="h"),
                    margin=dict(l=10, r=10, t=70, b=65),
                )
                st.plotly_chart(fig_dr, use_container_width=True)
            except Exception:
                st.warning("Bu değişken için bant analizi oluşturulamadı.")

        if rate_df is not None:
            with st.expander("Değişken Bazlı Dağılım & Batık Oranı Tablosu", expanded=True):
                eda_tbl_df = rate_df.copy()
                eda_tbl_df["Batık Oranı"] = eda_tbl_df["Batık Oranı"].map(lambda v: f"{float(v):.2%}")
                eda_tbl_df["Gözlem Sayısı"] = eda_tbl_df["Gözlem Sayısı"].map(lambda v: f"{int(v):,}")
                render_corporate_table(eda_tbl_df, numeric_cols=["Batık Oranı", "Gözlem Sayısı"])

    render_eda_fragment()


# ═══════════════════════════════════════════════════════════════════════
# SEKME 3 — MODEL VALİDASYON & KALİBRASYON
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">MODEL VALİDASYONU, KALİBRASYON &amp; PSI RAPORU</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; color:#64748B; margin-top:-6px; margin-bottom:14px;">BDDK / Basel III Model Risk Yönetimi Standartları</div>', unsafe_allow_html=True)

    # ── Metrik Karşılaştırma Tablosu ─────────────────────────────────
    st.markdown('<div class="section-header">TRAIN VS TEST PERFORMANS METRİKLERİ</div>', unsafe_allow_html=True)

    lgbm_m  = metrics["LightGBM"]
    lr_m    = metrics["LogisticRegression"]
    psi_m   = metrics["PSI"]

    rows = []
    for model_name, m in [("LightGBM", lgbm_m), ("Lojistik Regresyon", lr_m)]:
        for split, vals in m.items():
            rows.append({
                "Model": model_name,
                "Set": split,
                "AUC": vals["AUC"],
                "GINI": vals["GINI"],
                "KS": vals["KS"],
                "Brier Score": vals["BRIER"],
            })

    met_df = pd.DataFrame(rows)
    met_display_df = met_df.copy()
    met_display_df["AUC"] = met_display_df["AUC"].map("{:.4f}".format)
    met_display_df["GINI"] = met_display_df["GINI"].map("{:.4f}".format)
    met_display_df["KS"] = met_display_df["KS"].map("{:.4f}".format)
    met_display_df["Brier Score"] = met_display_df["Brier Score"].map("{:.6f}".format)
    render_corporate_table(
        met_display_df,
        numeric_cols=["AUC", "GINI", "KS", "Brier Score"],
    )

    # ── Overfitting & Stability Validation Kartları ───────────────────
    st.markdown('<div class="section-header">OVERFITTING &amp; STABİLİTE KONTROLLERİ</div>', unsafe_allow_html=True)

    # Karttaki gap, tabloda gösterilen (4 hane yuvarlanmış) Train - Test GINI değerlerinden hesaplanır
    lgbm_gap = round(lgbm_m["Train"]["GINI"], 4) - round(lgbm_m["Test"]["GINI"], 4)
    lr_gap   = round(lr_m["Train"]["GINI"], 4)   - round(lr_m["Test"]["GINI"], 4)
    psi_lgbm = psi_m["LightGBM"]
    psi_lr   = psi_m["LogisticRegression"]

    def _badge(condition: bool, pass_text: str, fail_text: str) -> str:
        border_color = "#10B981" if condition else "#EF4444"
        bg_color = "#ECFDF5" if condition else "#FEF2F2"
        txt_color = "#065F46" if condition else "#991B1B"
        text = pass_text if condition else fail_text
        return f'<div style="background:{bg_color}; border:1px solid {border_color}; border-left:4px solid {border_color}; border-radius:8px; padding:10px 14px; color:{txt_color}; font-family:\'JetBrains Mono\', monospace; font-size:12px; font-weight:700; box-shadow:0 1px 4px rgba(0,0,0,0.05);">{text}</div>'

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(
            _badge(abs(lgbm_gap) <= 0.05,
                   f"LightGBM GINI Gap: {lgbm_gap:+.4f} [PASS]",
                   f"LightGBM GINI Gap: {lgbm_gap:+.4f} [WARN]"),
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            _badge(abs(lr_gap) <= 0.05,
                   f"LogReg GINI Gap: {lr_gap:+.4f} [PASS]",
                   f"LogReg GINI Gap: {lr_gap:+.4f} [WARN]"),
            unsafe_allow_html=True,
        )
    with b3:
        st.markdown(
            _badge(psi_lgbm < 0.10,
                   f"LightGBM PSI: {psi_lgbm:.6f} [Stable]",
                   f"LightGBM PSI: {psi_lgbm:.6f} [Shift]"),
            unsafe_allow_html=True,
        )
    with b4:
        st.markdown(
            _badge(psi_lr < 0.10,
                   f"LogReg PSI: {psi_lr:.6f} [Stable]",
                   f"LogReg PSI: {psi_lr:.6f} [Shift]"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GINI / KS Bar Karşılaştırma Grafiği ──────────────────────────
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_gini = go.Figure()
        for model_name, m_data, color_pair in [
            ("LightGBM", lgbm_m, ["#3B82F6", "#60A5FA"]),
            ("LogReg",   lr_m,   ["#10B981", "#34D399"]),
        ]:
            fig_gini.add_trace(go.Bar(
                name=f"{model_name} Train",
                x=["GINI", "KS", "AUC"],
                y=[m_data["Train"]["GINI"], m_data["Train"]["KS"], m_data["Train"]["AUC"]],
                marker_color=color_pair[0],
            ))
            fig_gini.add_trace(go.Bar(
                name=f"{model_name} Test",
                x=["GINI", "KS", "AUC"],
                y=[m_data["Test"]["GINI"], m_data["Test"]["KS"], m_data["Test"]["AUC"]],
                marker_color=color_pair[1],
            ))
        fig_gini.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            title="Train vs Test: GINI / KS / AUC",
            barmode="group",
            height=340,
            yaxis=dict(range=[0, 1], dtick=0.1, color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
            xaxis=dict(color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
            font=dict(color="#374151", family="Inter, sans-serif"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_gini, use_container_width=True)

    # ── Kalibrasyon Eğrisi ────────────────────────────────────────────
    with col_v2:
        cal_tbl = pd.DataFrame(val_curves["calibration_table"])
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=cal_tbl["mean_predicted_pd"],
            y=cal_tbl["actual_default_rate"],
            mode="markers+lines",
            name="LightGBM Kalibrasyon",
            marker=dict(color="#3B82F6", size=9,
                        line=dict(color="#fff", width=1)),
            line=dict(color="#3B82F6", width=2),
        ))
        # Mükemmel kalibrasyon çizgisi
        cal_max = max(cal_tbl["mean_predicted_pd"].max(),
                      cal_tbl["actual_default_rate"].max()) * 1.1
        fig_cal.add_trace(go.Scatter(
            x=[0, cal_max], y=[0, cal_max],
            mode="lines", name="Mükemmel Kalibrasyon",
            line=dict(color="#64748B", dash="dash", width=1.5),
        ))
        fig_cal.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            title="Kalibrasyon Eğrisi (Reliability Curve)",
            xaxis_title="Tahmin Edilen PD (Ortalama)",
            yaxis_title="Gerçekleşen Batık Oranı",
            xaxis=dict(color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
            yaxis=dict(color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
            font=dict(color="#374151", family="Inter, sans-serif"),
            height=340,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # ── Brier Score ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">BRIER SCORE — KALİBRASYON KALİTESİ</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    for col, model_name, m_data in [
        (bc1, "LightGBM", lgbm_m),
        (bc2, "Lojistik Regresyon", lr_m),
    ]:
        with col:
            fig_br = go.Figure(go.Bar(
                x=["Train", "Test"],
                y=[m_data["Train"]["BRIER"], m_data["Test"]["BRIER"]],
                marker_color=["#3B82F6", "#10B981"],
                text=[f'{v:.6f}' for v in [m_data["Train"]["BRIER"], m_data["Test"]["BRIER"]]],
                textposition="outside",
                textfont=dict(color="#374151", family="JetBrains Mono"),
            ))
            fig_br.update_layout(
                template="plotly_white",
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
                title=f"{model_name} — Brier Score",
                height=260,
                yaxis=dict(range=[0, max(m_data["Train"]["BRIER"],
                                          m_data["Test"]["BRIER"]) * 1.25],
                           color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
                xaxis=dict(color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
                font=dict(color="#374151", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_br, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# SEKME 4 — CUT-OFF OPTİMİZASYONU & P&L SİMÜLATÖRÜ
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">CUT-OFF OPTİMİZASYONU &amp; FİNANSAL KAR/ZARAR SİMÜLATÖRÜ</div>', unsafe_allow_html=True)
    st.caption("Finansal Varsayım: Her sağlam kredi +$1,000 faiz geliri, her batık kredi -$10,000 anapara zararı olarak modellenmiştir.")

    cutoff_df = pd.DataFrame(val_curves["cutoff_analysis"])

    # ── Optimal Cut-off (Maks P&L) ───────────────────────────────────
    opt_idx  = cutoff_df["expected_pl_units"].idxmax()
    opt_row  = cutoff_df.loc[opt_idx]
    opt_score = pd_to_score(opt_row["pd_threshold"])

    st.markdown(f"""
    <div style="background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-left: 4px solid #10B981;
                border-radius: 12px;
                padding: 16px 24px;
                box-shadow: 0 1px 8px rgba(0,0,0,0.06);
                margin-bottom: 20px;">
        <div style="color: #64748B; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
            OPTİMUM KESME NOKTASI ÖNERİSİ
        </div>
        <div style="color: #0F172A; font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
            PD Eşiği: <span style="color:#10B981;">{opt_row['pd_threshold']:.0%}</span> &nbsp;|&nbsp;
            PDO Skoru: <span style="color:#3B82F6;">≥ {opt_score}</span> &nbsp;|&nbsp;
            Onay Oranı: {opt_row['approval_rate']:.1%} &nbsp;|&nbsp;
            Yakalanan Batık: {opt_row['bad_capture_rate']:.1%} &nbsp;|&nbsp;
            Net P&amp;L: <span style="color:#10B981;">${opt_row['expected_pl_units']:,.0f}K</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── İnteraktif Eşik Seçimi ────────────────────────────────────────
    st.markdown('<div class="section-header">EŞİK SİMÜLASYONU</div>', unsafe_allow_html=True)

    @st.fragment
    def render_cutoff_simulation():
        sim_col1, sim_col2 = st.columns([1, 2])

        with sim_col1:
            sim_threshold = st.slider(
                "PD Kesme Eşiği Seç:",
                min_value=0.02, max_value=0.50, value=0.18,
                step=0.02, format="%.2f", key="cutoff_slider_pd",
            )
            selected_cutoff = float(sim_threshold)
            sim_row = cutoff_df[
                (cutoff_df["pd_threshold"] - sim_threshold).abs() < 0.001
            ]
            if sim_row.empty:
                sim_row = cutoff_df.iloc[(cutoff_df["pd_threshold"] - sim_threshold)
                                          .abs().argsort()[:1]]

            sim_data = sim_row.iloc[0]
            sim_score_thresh = pd_to_score(float(sim_data["pd_threshold"]))

            st.markdown(f"""
            <div class="cutoff-metric-tile">
                <div class="cutoff-metric-label">SEÇİLEN EŞİK</div>
                <div class="cutoff-metric-val">{sim_data['pd_threshold']:.0%}</div>
                <div class="cutoff-metric-sub">PDO Skoru ≥ {sim_score_thresh}</div>
            </div>
            <div class="cutoff-metric-tile">
                <div class="cutoff-metric-label">ONAY ORANI</div>
                <div class="cutoff-metric-val">{sim_data['approval_rate']:.1%}</div>
                <div class="cutoff-metric-sub">Portföy Kabul Oranı</div>
            </div>
            <div class="cutoff-metric-tile">
                <div class="cutoff-metric-label">YAKALANAN BATIK ORANI</div>
                <div class="cutoff-metric-val">{sim_data['bad_capture_rate']:.1%}</div>
                <div class="cutoff-metric-sub">Elenen Batık Kredi Oranı</div>
            </div>
            <div class="cutoff-metric-tile">
                <div class="cutoff-metric-label">BEKLENEN NET P&amp;L</div>
                <div class="cutoff-metric-val" style="color: {'#10B981' if sim_data['expected_pl_units']>0 else '#EF4444'};">${float(sim_data['expected_pl_units']):,.0f}K</div>
                <div class="cutoff-metric-sub">Net Finansal Etki (+$1,000 Sağlam, -$10,000 Batık)</div>
            </div>
            """, unsafe_allow_html=True)

        with sim_col2:
            # Çift eksenli Plotly grafiği
            fig_cutoff = go.Figure()

            # Approval Rate & Bad Capture
            fig_cutoff.add_trace(go.Scatter(
                x=cutoff_df["pd_threshold"],
                y=cutoff_df["approval_rate"] * 100,
                name="Onay Oranı (%)",
                mode="lines+markers",
                line=dict(color="#3B82F6", width=2.5),
                marker=dict(size=5),
            ))
            fig_cutoff.add_trace(go.Scatter(
                x=cutoff_df["pd_threshold"],
                y=cutoff_df["bad_capture_rate"] * 100,
                name="Yakalanan Batık (%)",
                mode="lines+markers",
                line=dict(color="#EF4444", width=2.5),
                marker=dict(size=5),
            ))

            # P&L — sağ eksen
            fig_cutoff.add_trace(go.Scatter(
                x=cutoff_df["pd_threshold"],
                y=cutoff_df["expected_pl_units"],
                name="Net P&L ($K)",
                mode="lines+markers",
                line=dict(color="#10B981", width=2.5, dash="dot"),
                marker=dict(size=5),
                yaxis="y2",
            ))

            # Optimal eşik dikey çizgisi ve üst kutucuk etiketi
            opt_pd_val = float(opt_row["pd_threshold"])
            opt_near_right = opt_pd_val >= 0.40
            fig_cutoff.add_vline(x=opt_pd_val, line_dash="dot", line_color="#10B981")
            fig_cutoff.add_annotation(
                x=opt_pd_val,
                y=0.92,
                xref="x",
                yref="paper",
                text=f"<b>Optimum: {opt_pd_val:.0%}</b>",
                showarrow=False,
                xanchor="right" if opt_near_right else "left",
                xshift=-6 if opt_near_right else 6,
                font=dict(color="#047857", size=11, family="Inter, sans-serif"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#10B981",
                borderwidth=1,
                borderpad=4,
            )

            # Seçili eşik dikey çizgisi ve ikinci seviye kutucuk etiketi
            sel_near_right = selected_cutoff >= 0.40
            fig_cutoff.add_vline(x=selected_cutoff, line_dash="dash", line_color="#F59E0B")
            fig_cutoff.add_annotation(
                x=selected_cutoff,
                y=0.82,
                xref="x",
                yref="paper",
                text=f"<b>Seçili: {int(round(selected_cutoff * 100))}%</b>",
                showarrow=False,
                xanchor="right" if sel_near_right else "left",
                xshift=-6 if sel_near_right else 6,
                font=dict(color="#B45309", size=11, family="Inter, sans-serif"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#F59E0B",
                borderwidth=1,
                borderpad=4,
            )

            fig_cutoff.update_layout(
                template="plotly_white",
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
                title="PD Kesme Eşiği vs Onay Oranı / Batık Yakalama / Net P&L",
                xaxis=dict(title="PD Kesme Eşiği", tickformat=".0%", color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
                yaxis=dict(title="Oran (%)", range=[0, 105], color="#6B7280", gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#374151")),
                yaxis2=dict(title="Net P&L ($K)", overlaying="y", side="right", showgrid=False, color="#6B7280", linecolor="#E2E8F0", tickfont=dict(color="#10B981")),
                height=420,
                font=dict(color="#374151", family="Inter, sans-serif"),
                legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cutoff, use_container_width=True)

    render_cutoff_simulation()

    # ── Tam Cut-off Tablosu ───────────────────────────────────────────
    with st.expander("Tüm Kesme Eşiği Analiz Tablosu"):
        display_df = cutoff_df.copy()
        display_df["pd_score_threshold"] = display_df["pd_threshold"].apply(pd_to_score)
        display_df["expected_pl_fmt"] = display_df["expected_pl_units"].apply(lambda v: f"${v:,.0f}K")
        display_df = display_df[[
            "pd_threshold", "n_approved", "approval_rate",
            "bad_capture_rate", "expected_pl_fmt", "pd_score_threshold"
        ]]
        display_df.columns = [
            "PD Eşiği", "Onaylanan Müşteri", "Onay Oranı",
            "Yakalanan Batık Oranı", "Beklenen Net P&L", "PDO Skor Eşiği"
        ]
        display_df["Onaylanan Müşteri"] = display_df["Onaylanan Müşteri"].map(lambda v: f"{int(v):,}")
        display_df["Onay Oranı"] = display_df["Onay Oranı"].map("{:.1%}".format)
        display_df["Yakalanan Batık Oranı"] = display_df["Yakalanan Batık Oranı"].map("{:.1%}".format)
        display_df["PD Eşiği"] = display_df["PD Eşiği"].map("{:.0%}".format)
        render_corporate_table(
            display_df,
            numeric_cols=[
                "PD Eşiği",
                "Onaylanan Müşteri",
                "Onay Oranı",
                "Yakalanan Batık Oranı",
                "Beklenen Net P&L",
                "PDO Skor Eşiği",
            ],
        )
