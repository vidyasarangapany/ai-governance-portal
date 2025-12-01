import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI Agent Governance Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Sidebar – Load Data
# =====================================================================

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json",
    type=["json"]
)

if uploaded_file:
    df = pd.read_json(uploaded_file)

    # Normalize column names (protect against wrong keys)
    df.columns = [c.lower().strip() for c in df.columns]

    # Expected columns
    required_cols = [
        "agent_name", "owner", "created_by",
        "risk_level", "autonomy_level",
        "review_cadence", "lifecycle_state"
    ]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    st.sidebar.success(f"Loaded {len(df)} agents")
else:
    df = pd.DataFrame()

# =====================================================================
# Sidebar – Navigation (emoji-proof)
# =====================================================================

page = st.sidebar.radio(
    "Navigate",
    {
        "🏠 Overview": "overview",
        "📉 Insights": "insights",
        "📋 Agents Table": "table",
        "🔎 Agent Detail": "detail",
        "🕒 Lifecycle Timeline": "timeline",
        "⚙️ Policy Simulator": "policy"
    }
)

# =====================================================================
# Helper Functions
# =====================================================================

def compute_next_review(row):
    cadence = row["review_cadence"].lower()
    try:
        last_date = pd.to_datetime(row.get("last_reviewed", datetime.now()))
    except:
        last_date = datetime.now()

    if cadence == "immediate":
        return last_date + timedelta(days=1)
    if cadence == "monthly":
        return last_date + timedelta(days=30)
    if cadence == "quarterly":
        return last_date + timedelta(days=90)
    if cadence == "semi-annual":
        return last_date + timedelta(days=180)
    if cadence == "annual":
        return last_date + timedelta(days=365)

    return last_date + timedelta(days=30)

# Ensure needed derived fields exist
if not df.empty:
    df["last_reviewed"] = datetime.now() - pd.to_timedelta(
        (df.index + 1) * 10, unit="d"
    )
    df["next_review_due"] = df.apply(compute_next_review, axis=1)
    df["days_to_next"] = (df["next_review_due"] - datetime.now()).dt.days

# =====================================================================
# PAGE: Overview
# =====================================================================

if page == "overview":

    st.title("AI Agent Governance Portal")
    st.caption("Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture.")

    if df.empty:
        st.info("Upload a JSON file to begin.")
        st.stop()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Agents", len(df))
    c2.metric("High Risk", sum(df["risk_level"] == "HIGH RISK"))
    c3.metric("Medium Risk", sum(df["risk_level"] == "MEDIUM RISK"))
    c4.metric("Low Risk", sum(df["risk_level"] == "LOW RISK"))
    c5.metric("No Autonomy", sum(df["autonomy_level"] == "NO_AUTONOMY"))

    st.divider()

    # Overdue banner
    overdue = df[df["days_to_next"] < 0]
    if len(overdue) > 0:
        owners = ", ".join(overdue["owner"].unique())
        st.error(
            f"⚠️ {len(overdue)} agents are overdue for review across {owners} — this represents cross-functional governance risk."
        )

    st.subheader("🔔 Upcoming Reviews (next 30 days)")

    upcoming = df[(df["days_to_next"] >= 0) & (df["days_to_next"] <= 30)]

    st.dataframe(
        upcoming[
            [
                "agent_name", "owner", "risk_level",
                "review_cadence", "last_reviewed",
                "next_review_due", "days_to_next"
            ]
        ],
        use_container_width=True
    )

    cA, cB = st.columns(2)
    cA.download_button(
        "📥 Export upcoming reviews (CSV)",
        upcoming.to_csv(index=False),
        "upcoming_reviews.csv"
    )
    cB.download_button(
        "📥 Export upcoming reviews (JSON)",
        upcoming.to_json(orient="records"),
        "upcoming_reviews.json"
    )

    st.divider()

    # Heatmap — FIXED version (no narwhals DuplicateError)
    st.subheader("Risk vs Autonomy Heatmap")

    heat_df = (
        df.groupby(["risk_level", "autonomy_level"])
        .size()
        .reset_index(name="count")
    )

    fig_heat = px.density_heatmap(
        heat_df,
        x="autonomy_level",
        y="risk_level",
        z="count",
        color_continuous_scale="Blues",
        title="Where autonomy and risk intersect",
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    # Pie Chart — FIXED version
    st.subheader("Risk Breakdown")

    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    fig_pie = px.pie(
        risk_counts,
        names="risk_level",
        values="count",
        hole=0.4,
        title="Risk Distribution"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# =====================================================================
# PAGE: Insights
# =====================================================================

elif page == "insights":

    st.title("📉 Insights & Governance Lens")

    if df.empty:
        st.info("Upload a JSON file to begin.")
        st.stop()

    # Insight: Risk Mix
    st.subheader("1️⃣ Portfolio Risk Mix")

    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    fig_riskbar = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        color="risk_level",
        title="Risk Distribution",
    )

    st.plotly_chart(fig_riskbar, use_container_width=True)

    # Insight: Autonomy vs Risk
    st.subheader("2️⃣ Autonomy vs Risk Lens")

    risk_auto = (
        df.groupby(["risk_level", "autonomy_level"])
        .size()
        .reset_index(name="count")
    )

    fig_heat2 = px.density_heatmap(
        risk_auto,
        x="autonomy_level",
        y="risk_level",
        z="count",
        color_continuous_scale="Blues",
        title="Where autonomy and risk intersect",
    )

    st.plotly_chart(fig_heat2, use_container_width=True)

# =====================================================================
# PAGE: Agents Table
# =====================================================================

elif page == "table":

    st.title("📋 All Agents")

    if df.empty:
        st.info("Upload a JSON to view table.")
        st.stop()

    st.dataframe(df, use_container_width=True)

# =====================================================================
# PAGE: Agent Detail
# =====================================================================

elif page == "detail":

    st.title("🔎 Agent Detail")

    if df.empty:
        st.info("Upload a JSON file.")
        st.stop()

    agent_list = df["agent_name"].tolist()

    selected = st.selectbox("Select an agent", agent_list)

    row = df[df["agent_name"] == selected].iloc[0]

    st.subheader(selected)
    st.markdown(f"**Owner:** {row['owner']}")
    st.markdown(f"**Risk Level:** {row['risk_level']}")
    st.markdown(f"**Autonomy:** {row['autonomy_level']}")
    st.markdown(f"**Lifecycle:** {row['lifecycle_state']}")

# =====================================================================
# PAGE: Lifecycle Timeline
# =====================================================================

elif page == "timeline":

    st.title("🕒 Lifecycle Timeline")

    if df.empty:
        st.info("Upload JSON to view lifecycle timeline.")
        st.stop()

    st.markdown("Coming soon — Gantt-style lifecycle map.")

# =====================================================================
# PAGE: Policy Simulator
# =====================================================================

elif page == "policy":

    st.title("⚙️ Policy Simulator")

    if df.empty:
        st.info("Upload JSON.")
        st.stop()

    st.markdown("Coming soon — rule simulation engine.")

