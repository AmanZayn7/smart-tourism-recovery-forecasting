# app/app.py
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from components import metric_card, model_badge, info_banner
from stis.config import CITIES, METRICS_PATH, EVAL_PATH, OUTLOOK_PATH

# ---------------------------------------------
# Page config
# ---------------------------------------------
st.set_page_config(
    page_title="Smart Traveler Insights System",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------
# Inject CSS (dark theme)
# ---------------------------------------------
def inject_css():
    css_path = Path(__file__).parent / "theme.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_css()

# ---------------------------------------------
# Cached loaders
# ---------------------------------------------
@st.cache_data
def load_metrics(city: str) -> dict:
    p = METRICS_PATH(city)
    with open(p, "r") as f:
        return json.load(f)

@st.cache_data
def load_eval(city: str) -> pd.DataFrame:
    p = EVAL_PATH(city)
    return pd.read_csv(p, parse_dates=["date"])

@st.cache_data
def load_outlook(city: str) -> pd.DataFrame:
    p = OUTLOOK_PATH(city)
    return pd.read_csv(p, parse_dates=["date"])

# ---------------------------------------------
# Small helpers
# ---------------------------------------------
def confidence_from_mape(m: float) -> str:
    if m <= 3:
        return "★★★★★"
    if m <= 5:
        return "★★★★"
    if m <= 10:
        return "★★★"
    return "★★"

def section_title(txt: str):
    st.markdown(
        f"<h2 style='margin-top:0.6rem;margin-bottom:0.4rem;color:#a2a8b3'>{txt}</h2>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------
# Sidebar
# ---------------------------------------------
st.sidebar.title("🧭 Traveler Insights")
city = st.sidebar.selectbox("Choose a city", list(CITIES.keys()), index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Artifacts are precomputed. No training occurs in-app.")
st.sidebar.caption("12-month outlook is a scenario (not evaluated).")

# ---------------------------------------------
# Title / Model badge
# ---------------------------------------------
st.markdown(f"<h1 style='margin:0 0 .25rem 0'>{city}</h1>", unsafe_allow_html=True)
m = load_metrics(city)
st.markdown(model_badge(m["model"]), unsafe_allow_html=True)

# ---------------------------------------------
# Metric cards
# ---------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Model", m["model"].upper())
with c2:
    metric_card("MAPE (2024)", f"{m['mape']:.2f}%")
with c3:
    metric_card("RMSE (2024)", f"{int(m['rmse']):,}")
with c4:
    metric_card("Confidence", confidence_from_mape(m["mape"]))

# ---------------------------------------------
# Charts: 2024 Actuals + Predictions + 12M Outlook
# ---------------------------------------------
eval_df = load_eval(city)
outlook_df = load_outlook(city)

hist_df = eval_df[["date", "actual"]].copy().rename(columns={"actual": "value"})
hist_df["series"] = "Actual (2024 actual only here)"

pred_df = eval_df[["date", "pred"]].copy().rename(columns={"pred": "value"})
pred_df["series"] = "Prediction (2024 test)"

future_df = outlook_df[["date", "forecast"]].copy().rename(columns={"forecast": "value"})
future_df["series"] = "12-month outlook (scenario)"

plot_df = pd.concat([hist_df, pred_df, future_df], ignore_index=True)

fig = px.line(
    plot_df,
    x="date",
    y="value",
    color="series",
    title="Actuals, 2024 Predictions, and 12-Month Outlook",
    labels={"value": "Arrivals", "date": "Month"},
    template="plotly_dark",  # << dark theme
)
fig.update_traces(mode="lines+markers")
st.plotly_chart(fig, use_container_width=True)

info_banner(
    "The outlook is a scenario generated from the best model per city. "
    "It is <b>not evaluated</b> and excludes COVID years (2020–2022) in training."
)

# ---------------------------------------------
# Tables
# ---------------------------------------------
section_title("2024 Evaluation (Actual vs Pred)")
st.dataframe(
    eval_df.style.format(
        {"actual": "{:,}", "pred": "{:,}", "abs_err": "{:,}", "ape_pct": "{:.2f}"}
    ),
    use_container_width=True,
)

section_title("Next 12 Months — Outlook (Scenario)")
st.dataframe(
    outlook_df.style.format({"forecast": "{:,}"}),
    use_container_width=True,
)

# ---------------------------------------------
# Downloads
# ---------------------------------------------
section_title("Downloads")
d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "⬇️ 2024 Evaluation CSV",
        data=eval_df.to_csv(index=False),
        file_name=f"{city.lower().replace(' ','')}_eval_2024.csv",
        mime="text/csv",
    )
with d2:
    st.download_button(
        "⬇️ Outlook (12M) CSV",
        data=outlook_df.to_csv(index=False),
        file_name=f"{city.lower().replace(' ','')}_outlook_next12.csv",
        mime="text/csv",
    )

# ---------------------------------------------
# Model Card (from metrics.json)
# ---------------------------------------------
section_title("Model Card")
st.json(m["model_card"])
st.caption(
    f"Train: {m['train_period']} | Test: {m['test_period']} | "
    f"Rows: train={m['rows_train']}, test={m['rows_test']}"
)
