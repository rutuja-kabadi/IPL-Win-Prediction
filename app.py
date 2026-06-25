import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Probability",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — dark cricket aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #8B949E;
    font-size: 13px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Selectbox / number input */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background-color: #21262D !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 8px !important;
}

/* Slider */
[data-testid="stSlider"] .stSlider > div > div > div {
    background-color: #1D9E75 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #8B949E !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* Divider */
hr {
    border-color: #21262D !important;
    margin: 1rem 0 !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E6EDF3 !important;
}

/* Tab styling */
[data-testid="stTab"] {
    background-color: transparent;
    color: #8B949E;
    font-size: 13px;
    font-weight: 500;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1D9E75 !important;
    border-bottom: 2px solid #1D9E75 !important;
}

/* Info/warning boxes */
[data-testid="stInfo"] {
    background-color: #1C2A1E;
    border: 1px solid #1D9E75;
    border-radius: 8px;
    color: #7EE8B8;
}

/* Plot backgrounds match app */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model (with graceful fallback demo mode)
# ─────────────────────────────────────────────
MODEL_PATH  = Path("model/ipl_model.pkl")
BAT_PATH    = Path("model/le_bat.pkl")
BOWL_PATH   = Path("model/le_bowl.pkl")

DEMO_MODE = not (MODEL_PATH.exists() and BAT_PATH.exists() and BOWL_PATH.exists())

if not DEMO_MODE:
    model   = pickle.load(open(MODEL_PATH, "rb"))
    le_bat  = pickle.load(open(BAT_PATH,  "rb"))
    le_bowl = pickle.load(open(BOWL_PATH, "rb"))
    TEAMS   = sorted(le_bat.classes_)
else:
    model, le_bat, le_bowl = None, None, None
    TEAMS = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad",
    ]


# ─────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────
def predict_win_prob(runs, wickets, balls_done, target, batting, bowling):
    """Return win probability for batting team."""
    balls_remaining = 120 - balls_done
    runs_remaining  = target - runs
    eps = 1e-5
    crr    = runs / (balls_done / 6 + eps)
    rrr    = runs_remaining / (balls_remaining / 6 + eps)
    rr_diff = crr - rrr

    if DEMO_MODE:
        # Logistic approximation — good enough to demo the UI
        score  = (
             0.04 * runs
           - 0.12 * wickets
           + 0.10 * rr_diff
           - 0.03 * rrr
           + 0.02 * (balls_done / 120)
           - 3.5
        )
        return float(1 / (1 + np.exp(-score)))

    known_bat  = set(le_bat.classes_)
    known_bowl = set(le_bowl.classes_)
    bat_enc    = le_bat.transform([batting])[0]  if batting  in known_bat  else 0
    bowl_enc   = le_bowl.transform([bowling])[0] if bowling  in known_bowl else 0

    row = pd.DataFrame([{
        "runs_scored":     runs,
        "wickets_fallen":  wickets,
        "balls_remaining": balls_remaining,
        "runs_remaining":  runs_remaining,
        "wickets_in_hand": 10 - wickets,
        "over":            balls_done // 6,
        "crr":             crr,
        "rrr":             rrr,
        "rr_diff":         rr_diff,
        "target":          target,
        "momentum":        runs / max(balls_done, 6) * 6,
        "is_powerplay":    int(balls_done // 6 < 6),
        "is_death":        int(balls_done // 6 >= 15),
        "bat_enc":         bat_enc,
        "bowl_enc":        bowl_enc,
    }])
    return float(model.predict_proba(row)[0][1])


# ─────────────────────────────────────────────
# Build probability curve across all overs
# ─────────────────────────────────────────────
def build_curve(runs, wickets, balls_done, target, batting, bowling):
    """Simulate probability from ball 1 to current ball."""
    points = []
    for b in range(1, balls_done + 1):
        frac       = b / balls_done
        r_at_b     = int(runs * frac)
        w_at_b     = max(0, int(wickets * (b / balls_done) ** 1.5))
        prob       = predict_win_prob(r_at_b, w_at_b, b, target, batting, bowling)
        points.append({"ball": b, "over": b / 6, "prob": prob * 100})
    return pd.DataFrame(points)


# ─────────────────────────────────────────────
# Phase label helper
# ─────────────────────────────────────────────
def get_phase(over_num):
    if over_num < 6:   return "Powerplay", "#378ADD"
    if over_num < 15:  return "Middle overs", "#EF9F27"
    return "Death overs", "#E24B4A"


# ─────────────────────────────────────────────
# Sidebar — match setup
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏏 Match setup")
    st.markdown("---")

    batting_team = st.selectbox("Batting team", TEAMS, index=5)
    bowling_team = st.selectbox(
        "Bowling team",
        [t for t in TEAMS if t != batting_team],
        index=0,
    )

    st.markdown("---")
    target = st.number_input(
        "Target (runs to win)", min_value=50, max_value=300, value=175, step=1
    )

    st.markdown("---")
    st.markdown("##### Current state")

    over_num  = st.slider("Current over",   0, 19, 10)
    ball_num  = st.slider("Ball in over",   1,  6,  3)
    balls_done = over_num * 6 + ball_num

    runs     = st.slider("Runs scored",    0, target - 1, 95)
    wickets  = st.slider("Wickets fallen", 0, 9, 3)

    if DEMO_MODE:
        st.info("**Demo mode** — train and save your model to `model/` for real predictions.")


# ─────────────────────────────────────────────
# Compute current prediction
# ─────────────────────────────────────────────
win_prob  = predict_win_prob(runs, wickets, balls_done, target, batting_team, bowling_team)
lose_prob = 1 - win_prob

balls_remaining = 120 - balls_done
runs_remaining  = target - runs
eps = 1e-5
crr  = runs / (balls_done / 6 + eps)
rrr  = runs_remaining / (balls_remaining / 6 + eps)
phase_label, phase_color = get_phase(over_num)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        f"<h1 style='margin-bottom:0; font-size:28px'>IPL Win Probability</h1>"
        f"<p style='color:#8B949E; font-size:14px; margin-top:4px'>"
        f"{batting_team} chasing {target} vs {bowling_team}"
        f"</p>",
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        f"<div style='text-align:right; padding-top:8px'>"
        f"<span style='background:{phase_color}22; color:{phase_color}; "
        f"font-size:12px; font-weight:500; padding:4px 10px; border-radius:99px; "
        f"border:1px solid {phase_color}44'>{phase_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ─────────────────────────────────────────────
# Big probability display
# ─────────────────────────────────────────────
prob_col1, prob_col2 = st.columns(2)

with prob_col1:
    prob_pct = win_prob * 100
    bar_color = "#1D9E75" if prob_pct >= 50 else "#E24B4A"
    st.markdown(
        f"""<div style='background:#161B22; border:1px solid #21262D;
            border-radius:16px; padding:24px 28px; text-align:center'>
            <p style='color:#8B949E; font-size:12px; font-weight:500;
               text-transform:uppercase; letter-spacing:0.06em; margin:0 0 8px'>
               {batting_team}</p>
            <p style='font-family:Space Grotesk,sans-serif; font-size:56px;
               font-weight:700; margin:0; color:{bar_color}; line-height:1'>
               {prob_pct:.1f}<span style='font-size:28px'>%</span></p>
            <p style='color:#8B949E; font-size:13px; margin:8px 0 0'>Win probability</p>
            <div style='background:#21262D; border-radius:4px; height:6px; margin-top:16px'>
              <div style='background:{bar_color}; height:6px; border-radius:4px;
                          width:{prob_pct:.1f}%'></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

with prob_col2:
    lose_pct = lose_prob * 100
    bar_color2 = "#E24B4A" if prob_pct >= 50 else "#1D9E75"
    st.markdown(
        f"""<div style='background:#161B22; border:1px solid #21262D;
            border-radius:16px; padding:24px 28px; text-align:center'>
            <p style='color:#8B949E; font-size:12px; font-weight:500;
               text-transform:uppercase; letter-spacing:0.06em; margin:0 0 8px'>
               {bowling_team}</p>
            <p style='font-family:Space Grotesk,sans-serif; font-size:56px;
               font-weight:700; margin:0; color:{bar_color2}; line-height:1'>
               {lose_pct:.1f}<span style='font-size:28px'>%</span></p>
            <p style='color:#8B949E; font-size:13px; margin:8px 0 0'>Win probability</p>
            <div style='background:#21262D; border-radius:4px; height:6px; margin-top:16px'>
              <div style='background:{bar_color2}; height:6px; border-radius:4px;
                          width:{lose_pct:.1f}%'></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ─────────────────────────────────────────────
# Match snapshot metrics
# ─────────────────────────────────────────────
st.markdown("##### Match snapshot")
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Runs scored",    f"{runs}",
          delta=f"{runs_remaining} needed")
m2.metric("Wickets fallen", f"{wickets}",
          delta=f"{10 - wickets} in hand",
          delta_color="inverse")
m3.metric("Balls remaining", f"{balls_remaining}",
          delta=f"Over {over_num}.{ball_num}")
m4.metric("Current RR",     f"{crr:.2f}",
          delta=f"Req: {rrr:.2f}",
          delta_color="normal" if crr >= rrr else "inverse")
m5.metric("Run rate gap",   f"{crr - rrr:+.2f}",
          delta="ahead" if crr >= rrr else "behind",
          delta_color="normal" if crr >= rrr else "inverse")

st.markdown("---")


# ─────────────────────────────────────────────
# Tabs — Chart / Scenario / Guide
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈  Probability curve", "🔮  Scenario explorer", "📋  Feature guide"])


# ── Tab 1: Probability curve ──────────────────
with tab1:
    curve_df = build_curve(runs, wickets, balls_done, target, batting_team, bowling_team)

    fig = go.Figure()

    # Phase background bands
    fig.add_vrect(x0=0, x1=6,  fillcolor="#378ADD", opacity=0.05, line_width=0)
    fig.add_vrect(x0=6, x1=15, fillcolor="#EF9F27", opacity=0.05, line_width=0)
    fig.add_vrect(x0=15, x1=20, fillcolor="#E24B4A", opacity=0.05, line_width=0)

    # Phase labels
    for x, label in [(3, "Powerplay"), (10.5, "Middle"), (17.5, "Death")]:
        if x <= curve_df["over"].max():
            fig.add_annotation(
                x=x, y=96, text=label,
                showarrow=False,
                font=dict(size=10, color="#8B949E"),
                xanchor="center",
            )

    # 50% line
    fig.add_hline(
        y=50, line_dash="dot", line_color="#30363D", line_width=1,
        annotation_text="50%", annotation_position="right",
        annotation_font_color="#8B949E", annotation_font_size=11,
    )

    # Main probability line
    fig.add_trace(go.Scatter(
        x=curve_df["over"],
        y=curve_df["prob"].round(1),
        mode="lines",
        name=batting_team,
        line=dict(color="#1D9E75", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(29,158,117,0.08)",
        hovertemplate="Over %{x:.1f}<br>Win prob: <b>%{y:.1f}%</b><extra></extra>",
    ))

    # Current position marker
    fig.add_trace(go.Scatter(
        x=[curve_df["over"].iloc[-1]],
        y=[curve_df["prob"].iloc[-1].round(1)],
        mode="markers",
        marker=dict(color="#1D9E75", size=10, line=dict(color="#0D1117", width=2)),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        plot_bgcolor="#0D1117",
        paper_bgcolor="#0D1117",
        font=dict(family="Inter", color="#8B949E", size=12),
        margin=dict(l=48, r=24, t=24, b=48),
        height=340,
        xaxis=dict(
            title="Over", range=[0, 20], dtick=2,
            gridcolor="#21262D", linecolor="#21262D",
            tickcolor="#21262D", tickfont=dict(color="#8B949E"),
        ),
        yaxis=dict(
            title="Win probability (%)", range=[0, 100], dtick=25,
            gridcolor="#21262D", linecolor="#21262D",
            tickcolor="#21262D", tickfont=dict(color="#8B949E"),
            ticksuffix="%",
        ),
        showlegend=False,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"<p style='color:#8B949E; font-size:12px; text-align:center'>"
        f"Simulated trajectory — adjust sliders in the sidebar to update</p>",
        unsafe_allow_html=True,
    )


# ── Tab 2: Scenario explorer ──────────────────
with tab2:
    st.markdown("##### What if…")
    st.markdown(
        "<p style='color:#8B949E; font-size:13px'>"
        "Explore how the win probability shifts under different scenarios "
        "from the current match state.</p>",
        unsafe_allow_html=True,
    )

    scenarios = {
        "Current state": (runs, wickets, balls_done),
        "Wicket next ball": (runs, min(wickets + 1, 9), balls_done + 1),
        "Boundary next ball": (runs + 4, wickets, balls_done + 1),
        "Six next ball": (runs + 6, wickets, balls_done + 1),
        "Dot ball": (runs, wickets, balls_done + 1),
        "Two wickets in 2 balls": (runs, min(wickets + 2, 9), balls_done + 2),
    }

    scen_data = []
    for label, (r, w, b) in scenarios.items():
        if b <= 120:
            p = predict_win_prob(r, w, b, target, batting_team, bowling_team)
            scen_data.append({
                "Scenario": label,
                "Win %": round(p * 100, 1),
                "Δ from current": round((p - win_prob) * 100, 1),
            })

    scen_df = pd.DataFrame(scen_data)

    fig2 = go.Figure()
    colors = [
        "#1D9E75" if d >= 0 else "#E24B4A"
        for d in scen_df["Δ from current"]
    ]
    fig2.add_trace(go.Bar(
        x=scen_df["Scenario"],
        y=scen_df["Win %"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in scen_df["Win %"]],
        textposition="outside",
        textfont=dict(color="#E6EDF3", size=11),
        hovertemplate="%{x}<br>Win prob: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig2.add_hline(y=50, line_dash="dot", line_color="#30363D", line_width=1)
    fig2.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(family="Inter", color="#8B949E", size=12),
        margin=dict(l=32, r=24, t=24, b=80),
        height=320,
        xaxis=dict(
            gridcolor="#21262D", linecolor="#21262D",
            tickcolor="#21262D", tickfont=dict(color="#8B949E", size=11),
        ),
        yaxis=dict(
            title="Win probability (%)", range=[0, 110],
            gridcolor="#21262D", linecolor="#21262D",
            tickcolor="#21262D", tickfont=dict(color="#8B949E"),
            ticksuffix="%",
        ),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Delta table
    st.markdown("##### Impact summary")
    for _, row in scen_df.iterrows():
        delta = row["Δ from current"]
        delta_color = "#1D9E75" if delta >= 0 else "#E24B4A"
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        st.markdown(
            f"""<div style='display:flex; justify-content:space-between;
                align-items:center; padding:8px 12px; background:#161B22;
                border:1px solid #21262D; border-radius:8px; margin-bottom:6px'>
                <span style='font-size:13px; color:#E6EDF3'>{row["Scenario"]}</span>
                <span style='font-family:Space Grotesk,sans-serif; font-size:14px;
                    font-weight:600; color:{delta_color}'>
                    {arrow} {abs(delta):.1f}pp &nbsp;
                    <span style='color:#8B949E; font-weight:400; font-size:12px'>
                    → {row["Win %"]:.1f}%</span>
                </span>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Tab 3: Feature guide ──────────────────────
with tab3:
    st.markdown("##### Features used by the model")
    st.markdown(
        "<p style='color:#8B949E; font-size:13px'>A reference for your "
        "CV conversations — what each feature means and why it matters.</p>",
        unsafe_allow_html=True,
    )

    features = [
        ("runs_scored", "Tier 1", "Total runs in current innings (cumulative)", crr),
        ("wickets_in_hand", "Tier 1", "10 − wickets fallen", 10 - wickets),
        ("balls_remaining", "Tier 1", "120 − balls faced", balls_remaining),
        ("rrr", "Tier 2", "Runs remaining ÷ overs remaining", round(rrr, 2)),
        ("crr", "Tier 2", "Runs scored ÷ overs faced", round(crr, 2)),
        ("rr_diff", "Tier 2", "CRR − RRR (positive = ahead)", round(crr - rrr, 2)),
        ("target", "Tier 3", "1st innings total + 1", target),
        ("momentum", "Bonus", "Runs in last 6 balls", "rolling"),
        ("is_powerplay", "Bonus", "Over < 6", int(over_num < 6)),
        ("is_death", "Bonus", "Over ≥ 15", int(over_num >= 15)),
    ]

    tier_colors = {
        "Tier 1": ("#E1F5EE", "#0F6E56"),
        "Tier 2": ("#EEEDFE", "#3C3489"),
        "Tier 3": ("#FAEEDA", "#633806"),
        "Bonus":  ("#F1EFE8", "#444441"),
    }

    for fname, tier, desc, val in features:
        bg, fg = tier_colors[tier]
        st.markdown(
            f"""<div style='display:flex; gap:10px; align-items:flex-start;
                padding:10px 12px; background:#161B22; border:1px solid #21262D;
                border-radius:8px; margin-bottom:6px'>
                <span style='background:{bg}; color:{fg}; font-size:10px;
                    font-weight:600; padding:2px 7px; border-radius:99px;
                    flex-shrink:0; margin-top:2px; white-space:nowrap'>{tier}</span>
                <div style='flex:1; min-width:0'>
                    <p style='font-family:monospace; font-size:12px; color:#79C0FF;
                       margin:0 0 2px'>{fname}</p>
                    <p style='font-size:12px; color:#8B949E; margin:0'>{desc}</p>
                </div>
                <span style='font-family:Space Grotesk,sans-serif; font-size:13px;
                    font-weight:600; color:#E6EDF3; white-space:nowrap'>{val}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#30363D; font-size:11px; text-align:center'>"
    "IPL Win Probability · XGBoost · Streamlit · "
    "Data: Cricsheet / Kaggle · Built for CDC internship season"
    "</p>",
    unsafe_allow_html=True,
)
