import streamlit as st

def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{title}</div>
          <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def model_badge(model_code: str) -> str:
    label = {
        "ridge": "Ridge (log-arrivals)",
        "rf": "Random Forest (log-arrivals)",
        "snaive": "Seasonal-Naive (lag-12)",
    }.get(model_code, model_code.upper())
    return f"""<span class="badge">{label}</span>"""

def info_banner(text: str):
    st.markdown(f"""<div class="info-banner">{text}</div>""", unsafe_allow_html=True)
