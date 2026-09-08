import streamlit as st
import plotly.graph_objects as go

from data.loader import load_stock_data
from features.engineer import create_features
from drift.kswin_detector import run_kswin
from evaluation.ground_truth_external import get_earnings_ground_truth

st.set_page_config(
    page_title="NVDA Concept Drift Monitor",
    layout="wide"
)

ticker = st.sidebar.text_input("Ticker", value="NVDA")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this dashboard**\n\n"
    "KSWIN runs directly on the 20-day rolling volatility series. "
    "Red dashed lines mark real, SEC-verified NVDA earnings dates. "
    "Orange dotted lines mark KSWIN's drift alarms."
)

df = load_stock_data(ticker)
df = create_features(df)
df = df.reset_index(drop=True)

st.title("Concept Drift Monitoring Dashboard")
st.caption(
    "Detector: KSWIN on raw Volatility_20. Ground truth: SEC EDGAR-verified "
    "earnings dates (see data/earnings_dates_FINAL.csv for provenance)."
)

volatility_stream = df["Volatility_20"].values
kswin_alarms = run_kswin(volatility_stream)

earnings_ground_truth = []
if ticker == "NVDA":
    try:
        earnings_ground_truth = get_earnings_ground_truth(df)
    except FileNotFoundError:
        st.sidebar.warning(
            "Ground truth file not found -- earnings markers will not "
            "be shown. Run the data/fetch_earnings_ground_truth.py -> "
            "classify -> finalize pipeline first."
        )

from evaluation.detector_metrics import evaluate_detector

tolerance_days = st.sidebar.slider(
    "Match tolerance (trading days)", min_value=5, max_value=30, value=15, step=5
)

metrics = evaluate_detector(kswin_alarms, earnings_ground_truth, tolerance=tolerance_days)

col1, col2, col3, col4 = st.columns(4)
col1.metric("KSWIN Alarms Raised", len(kswin_alarms))
col2.metric("Confirmed Earnings Dates", len(earnings_ground_truth))
col3.metric("Precision", f"{metrics['precision']:.2f}")
col4.metric("Recall", f"{metrics['recall']:.2f}")

if metrics["avg_delay"] is not None:
    st.caption(
        f"Average detection delay: {metrics['avg_delay']:.1f} trading days "
        f"(matched {metrics['matched_truth']} of {len(earnings_ground_truth)} "
        f"earnings events within {tolerance_days}-day tolerance)."
    )
else:
    st.caption(f"No KSWIN alarms matched an earnings date within {tolerance_days}-day tolerance.")


def add_event_lines(fig, positions, dates_series, color, label, dash="dash"):
    shown_label = False
    for pos in positions:
        if 0 <= pos < len(dates_series):
            event_date = dates_series.iloc[pos]
            fig.add_vline(
                x=event_date,
                line_width=1.5,
                line_dash=dash,
                line_color=color,
            )
    return fig


# --- Price chart with both overlays ---
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=df["Date"], y=df["Close"], mode="lines", name="Close Price",
    line=dict(color="#1f77b4")
))
fig_price = add_event_lines(fig_price, earnings_ground_truth, df["Date"], "red", "Earnings")
fig_price = add_event_lines(fig_price, kswin_alarms, df["Date"], "orange", "KSWIN Alarm", dash="dot")
fig_price.update_layout(title="Price, with Earnings Dates (red) and KSWIN Alarms (orange)",
                         xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_price, use_container_width=True)

# --- Volatility chart, the actual detector input ---
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=df["Date"], y=df["Volatility_20"], mode="lines", name="20-Day Volatility",
    line=dict(color="#2ca02c")
))
fig_vol = add_event_lines(fig_vol, earnings_ground_truth, df["Date"], "red", "Earnings")
fig_vol = add_event_lines(fig_vol, kswin_alarms, df["Date"], "orange", "KSWIN Alarm", dash="dot")
fig_vol.update_layout(title="20-Day Rolling Volatility (the signal KSWIN actually monitors)",
                       xaxis_title="Date", yaxis_title="Volatility")
st.plotly_chart(fig_vol, use_container_width=True)

# --- Returns chart, shown for context (KSWIN found nothing here) ---
fig_ret = go.Figure()
fig_ret.add_trace(go.Scatter(
    x=df["Date"], y=df["Log_Return"], mode="lines", name="Log Returns",
    line=dict(color="#7f7f7f")
))
fig_ret.update_layout(title="Log Returns (for reference -- KSWIN detected no drift on this series)",
                       xaxis_title="Date", yaxis_title="Log Return")
st.plotly_chart(fig_ret, use_container_width=True)

with st.expander("Alarm and ground-truth positions (row indices)"):
    st.write("KSWIN alarm positions:", kswin_alarms)
    st.write("Earnings ground-truth positions:", earnings_ground_truth)