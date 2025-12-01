import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Agent Governance Portal", layout="wide")

st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json", type=["json"]
)

if uploaded_file:
    df = pd.read_json(uploaded_file)
    st.sidebar.success(f"Loaded {len(df)} agents")
else:
    st.sidebar.info("Waiting for file...")
    st.stop()

# ---- FIX FIELD NAMES ----
df.rename(
    columns={
        "risk": "risk_level",
        "autonomy": "autonomy_level"
    },
    inplace=True,
)

# Sidebar filters
risk_filter = st.sidebar.selectbox(
    "Filter by Risk Level",
    ["All"] + sorted(df["risk_level"].unique())
)

aut_filter = st.sidebar.selectbox(
    "Filter by Autonomy Level",
    ["All"] + sorted(df["autonomy_level"].unique())
)

life_filter = st.sidebar.selectbox(
    "Filter by Lifecycle State",
    ["All"] + sorted(df["lifecycle_state"].unique())
)

filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if aut_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == aut_filter]
if life_filter != "All":
    filtered = filtered[filtered["lifecycle_state"] == life_filter]

# Page navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📆 Lifecycle Timeline", "📋 Agents Table", "🔍 Agent Detail", "💡 Insights"]
)

# ===================================================
# PAGE 1 — OVERVIEW
# ===================================================
if page == "🏠 Overview":

    st.title("🛡️ AI Agent Governance Portal")
    st.markdown("Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Agents", len(df))
    col2.metric("High Risk", sum(df["risk_level"] == "HIGH RISK"))
    col3.metric("Medium Risk", sum(df["risk_level"] == "MEDIUM RISK"))
    col4.metric("Low Risk", sum(df["risk_level"] == "LOW RISK"))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("No Autonomy", sum(df["autonomy_level"] == "NO_AUTONOMY"))
    col6.metric("Human-in-Loop", sum(df["autonomy_level"] == "HUMAN_IN_LOOP"))
    col7.metric("Limited Autonomy", sum(df["autonomy_level"] == "LIMITED_AUTONOMY"))
    col8.metric("Auto Allowed", sum(df["autonomy_level"] == "AUTO_ALLOWED"))

    # Synthetic review dates
    today = datetime.today()
    df["last_reviewed"] = today - pd.to_timedelta((df.index % 90) + 5, unit="d")
    df["next_review_due"] = df["last_reviewed"] + pd.to_timedelta(
        df["review_cadence"].map({
            "Immediate": 1,
            "Monthly": 30,
            "Quarterly": 90,
            "Semi-Annual": 180,
            "Annual": 365
        }).fillna(30),
        unit="d"
    )
    df["days_to_next"] = (df["next_review_due"] - today).dt.days

    st.markdown("### 🔔 Upcoming Reviews (next 30 days)")
    upcoming = df[df["days_to_next"] <= 30].sort_values("days_to_next")

    st.dataframe(
        upcoming[[
            "agent_name", "owner", "risk_level",
            "review_cadence", "last_reviewed",
            "next_review_due", "days_to_next"
        ]]
    )

    # Export buttons
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "📄 Export upcoming reviews (CSV)",
            upcoming.to_csv(index=False),
            file_name="upcoming_reviews.csv"
        )
    with colB:
        st.download_button(
            "📄 Export upcoming reviews (JSON)",
            upcoming.to_json(orient="records", indent=2),
            file_name="upcoming_reviews.json"
        )

    # Executive guidance
    st.markdown("""
    ### How to read this section
    - Focus on agents with reviews due in the next 0–7 days for immediate attention.  
    - 8–30 day horizon is your planning runway – where you can batch reviews by owner or business unit.  
    - If you see **no upcoming reviews**, your cadence may be generous or you’ve recently completed a cycle.
    """)

    # Highlight high-risk agents
    with st.expander("🔥 High-Risk Agent Spotlight"):
        hr = df[df["risk_level"] == "HIGH RISK"]

        for _, row in hr.iterrows():
            st.markdown(f"### {row['agent_name']}")
            st.markdown(f"- **Owner:** `{row['owner']}`")
            st.markdown(f"- **Risk Level:** `{row['risk_level']}`")
            st.markdown(f"- **Autonomy:** `{row['autonomy_level']}`")
            st.markdown(f"- **Lifecycle:** `{row['lifecycle_state']}`")
            st.markdown("---")
# ===================================================
# PAGE 2 — LIFECYCLE TIMELINE
# ===================================================
elif page == "📆 Lifecycle Timeline":
    st.title("📆 Lifecycle Timeline")

    df["age_days"] = (datetime.today() - df["last_reviewed"]).dt.days

    fig_timeline = px.scatter(
        df,
        x="age_days",
        y="agent_name",
        color="risk_level",
        size="age_days",
        title="Agent Review Aging"
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

# ===================================================
# PAGE 3 — AGENTS TABLE
# ===================================================
elif page == "📋 Agents Table":
    st.title("📋 All Agents")

    st.dataframe(
        filtered[[
            "agent_name", "owner", "created_by",
            "risk_level", "autonomy_level",
            "review_cadence", "lifecycle_state"
        ]]
    )

# ===================================================
# PAGE 4 — AGENT DETAIL
# ===================================================
elif page == "🔍 Agent Detail":

    st.title("🔍 Agent Detail Viewer")

    agent_list = sorted(df["agent_name"].unique())
    choice = st.selectbox("Select an agent", agent_list)

    row = df[df["agent_name"] == choice].iloc[0]

    st.markdown(f"## {row['agent_name']}")
    st.markdown(f"- **Owner:** `{row['owner']}`")
    st.markdown(f"- **Created By:** `{row['created_by']}`")
    st.markdown(f"- **Risk Level:** `{row['risk_level']}`")
    st.markdown(f"- **Autonomy:** `{row['autonomy_level']}`")
    st.markdown(f"- **Review Cadence:** `{row['review_cadence']}`")
    st.markdown(f"- **Lifecycle:** `{row['lifecycle_state']}`")
    st.markdown("---")

    st.download_button(
        "📄 Export agent record (JSON)",
        row.to_json(indent=2),
        file_name=f"{choice}_record.json"
    )
# ===================================================
# PAGE 5 — INSIGHTS
# ===================================================
elif page == "💡 Insights":

    st.title("💡 Insights & Governance Lens")

    # -----------------------------------------------
    # Insight 1 — Risk Mix
    # -----------------------------------------------
    st.subheader("📊 Portfolio Risk Mix")
    st.markdown("""
    - High risk agents require focused control and predictable review cycles.  
    - Medium risk agents often drive operational efficiency but carry dependencies.  
    - Low risk agents typically represent automation of low-impact workflows.
    """)

    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    fig_risk = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        color="risk_level",
        title="Risk Distribution"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # -----------------------------------------------
    # Insight 2 — Autonomy vs Risk
    # -----------------------------------------------
    st.subheader("🧠 Autonomy vs Risk Lens")

    risk_auto = (
        df.groupby(["risk_level", "autonomy_level"])
        .size()
        .reset_index(name="count")
    )

    if not risk_auto.empty:
        fig_heat = px.density_heatmap(
            risk_auto,
            x="autonomy_level",
            y="risk_level",
            z="count",
            color_continuous_scale="Blues",
            title="Where autonomy and risk intersect"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    **Executive takeaway:**  
    Focus on **HIGH RISK + higher autonomy** agents first — these require strongest guardrails and oversight.
    """)

    # -----------------------------------------------
    # Insight 3 — Architecture Diagram
    # -----------------------------------------------
    st.subheader("🧩 High-Level Architecture (Mermaid)")

    mermaid_snippet = """
    flowchart LR
        subgraph DataLayer[Data Sources]
            JSON[(governance_decisions.json)]
        end

        JSON --> Portal

        subgraph Portal[Governance Portal]
            Filters --> Metrics
            Metrics --> Reviews
            Reviews --> Insights
        end
    """

    st.code(mermaid_snippet, language="mermaid")

