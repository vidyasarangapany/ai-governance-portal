import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI Agent Governance Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .small-metric { font-size: 32px !important; font-weight: 600; }
    .metric-label { font-size: 16px !important; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR – FILE UPLOAD + STATE
# ---------------------------------------------------------
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json", type=["json"]
)

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if uploaded_file:
    st.session_state.df = pd.read_json(uploaded_file)

df = st.session_state.df

if not df.empty:
    st.sidebar.success(f"Loaded {len(df)} agents")
else:
    st.sidebar.info("Upload a JSON file to begin.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.subheader("Filter by Risk Level")
risk_filter = st.sidebar.selectbox(
    "Risk Level", ["All"] + sorted(df["risk_level"].unique())
)

st.sidebar.subheader("Filter by Autonomy Level")
auto_filter = st.sidebar.selectbox(
    "Autonomy Level", ["All"] + sorted(df["autonomy_level"].unique())
)

st.sidebar.subheader("Filter by Lifecycle State")
lifecycle_filter = st.sidebar.selectbox(
    "Lifecycle State", ["All"] + sorted(df["lifecycle_state"].unique())
)

# Apply filters
df_filtered = df.copy()

if risk_filter != "All":
    df_filtered = df_filtered[df_filtered["risk_level"] == risk_filter]

if auto_filter != "All":
    df_filtered = df_filtered[df_filtered["autonomy_level"] == auto_filter]

if lifecycle_filter != "All":
    df_filtered = df_filtered[df_filtered["lifecycle_state"] == lifecycle_filter]

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Insights",
        "📋 Agents Table",
        "🔍 Agent Detail",
        "⏳ Lifecycle Timeline",
        "⚙ Policy Simulator"
    ]
)
# ---------------------------------------------------------
# HELPER – ENRICH WITH SYNTHETIC REVIEW SCHEDULE
# ---------------------------------------------------------
def enrich_with_schedule(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic last_reviewed / next_review_due / days_to_next columns."""
    df_work = df_in.copy().reset_index(drop=True)
    today = datetime.today().date()

    cadence_days = {
        "Immediate": 7,
        "Monthly": 30,
        "Quarterly": 90,
        "Semi-Annual": 180,
        "Semi Annual": 180,
        "Semiannual": 180,
        "Annual": 365,
        "Annually": 365,
    }

    last_dates = []
    next_dates = []
    days_to_next_list = []

    for i, row in df_work.iterrows():
        cadence = row.get("review_cadence", "Quarterly")
        step = cadence_days.get(str(cadence), 90)

        # Stagger last review dates in the past
        last = today - timedelta(days=(i + 1) * 7)
        next_d = last + timedelta(days=step)
        days_to_next = (next_d - today).days

        last_dates.append(last)
        next_dates.append(next_d)
        days_to_next_list.append(days_to_next)

    df_work["last_reviewed"] = last_dates
    df_work["next_review_due"] = next_dates
    df_work["days_to_next"] = days_to_next_list
    return df_work


# Use the enriched frame for any schedule-related views
df_sched = enrich_with_schedule(df_filtered)


# ---------------------------------------------------------
# PAGE 1 – OVERVIEW
# ---------------------------------------------------------
if page == "🏠 Overview":
    st.title("AI Agent Governance Portal")
    st.caption(
        "Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture."
    )

    # ---- Top metrics row ----
    total_agents = len(df_filtered)
    high_risk = (df_filtered["risk_level"] == "HIGH RISK").sum()
    medium_risk = (df_filtered["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (df_filtered["risk_level"] == "LOW RISK").sum()
    no_autonomy = (df_filtered["autonomy_level"] == "NO_AUTONOMY").sum()
    auto_allowed = (df_filtered["autonomy_level"] == "AUTO_ALLOWED").sum()
    human_in_loop = (df_filtered["autonomy_level"] == "HUMAN_IN_LOOP").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-label">Total Agents</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-metric">{total_agents}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-label">High Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-metric">{high_risk}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-label">Medium Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-metric">{medium_risk}</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-label">Low Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-metric">{low_risk}</div>', unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown('<div class="metric-label">No Autonomy</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-metric">{no_autonomy}</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="metric-label">Human-in-Loop</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-metric">{human_in_loop}</div>', unsafe_allow_html=True)
    with c7:
        st.markdown('<div class="metric-label">Auto Allowed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-metric">{auto_allowed}</div>', unsafe_allow_html=True)

    # ---- Governance posture banner ----
    overdue_df = df_sched[df_sched["days_to_next"] < 0]
    num_overdue = len(overdue_df)
    if num_overdue > 0:
        overdue_owners = sorted(overdue_df["owner"].unique())
        owners_str = ", ".join(overdue_owners)
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6;border-left:4px solid #e55353;
                        padding:10px 16px;margin-top:24px;margin-bottom:12px;">
                <b>⚠ {num_overdue} agents are overdue for review</b>
                across <b>{owners_str}</b> — this represents cross-functional governance risk.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Upcoming reviews (next 30 days) ----
    st.subheader("Upcoming Reviews (next 30 days)")
    horizon_df = df_sched[df_sched["days_to_next"] <= 30].copy()
    horizon_df = horizon_df.sort_values("days_to_next")

    if horizon_df.empty:
        st.info("No upcoming reviews in the next 30 days under current filters.")
    else:
        display_cols = [
            "agent_name",
            "owner",
            "risk_level",
            "review_cadence",
            "last_reviewed",
            "next_review_due",
            "days_to_next",
        ]
        st.dataframe(horizon_df[display_cols], use_container_width=True)

        # Export buttons
        csv_all = horizon_df.to_csv(index=False).encode("utf-8")
        json_all = horizon_df.to_json(orient="records", indent=2).encode("utf-8")

        cexp1, cexp2 = st.columns(2)
        with cexp1:
            st.download_button(
                "📥 Export upcoming reviews (CSV)",
                data=csv_all,
                file_name="upcoming_reviews.csv",
                mime="text/csv",
            )
        with cexp2:
            st.download_button(
                "📥 Export upcoming reviews (JSON)",
                data=json_all,
                file_name="upcoming_reviews.json",
                mime="application/json",
            )

    # ---- Mini-guide on how to read the section ----
    st.markdown(
        """
        ### How to read this section
        - Focus on agents with reviews due in the next **0–7 days** for immediate attention.  
        - **8–30 day horizon** is your planning runway — where you can batch reviews by owner or business unit.  
        - Use these metrics to drive **review SLAs**, escalation rules, and dashboards for Security, HR, and IT.
        """
    )

    # ---- Risk vs Autonomy heatmap + Risk breakdown pie ----
    st.subheader("Governance posture at a glance")
    col_heat, col_pie = st.columns(2)

    # Heatmap data
    risk_auto = (
        df_filtered.groupby(["risk_level", "autonomy_level"])
        .size()
        .reset_index(name="count")
    )

    with col_heat:
        st.markdown("**Risk vs Autonomy Heatmap**")
        if risk_auto.empty:
            st.info("No data available for the current filter selection.")
        else:
            fig_heat = px.density_heatmap(
                risk_auto,
                x="autonomy_level",
                y="risk_level",
                z="count",
                color_continuous_scale="Blues",
                title="Where autonomy and risk intersect",
            )
            fig_heat.update_layout(margin=dict(l=40, r=10, t=40, b=40))
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_pie:
        st.markdown("**Risk Breakdown**")
        risk_counts = (
            df_filtered["risk_level"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "risk_level", "risk_level": "count"})
        )

        if risk_counts.empty:
            st.info("No data available for the current filter selection.")
        else:
            # IMPORTANT: no .copy() here – avoids DuplicateError with narwhals
           risk_counts = (
    df_filtered["risk_level"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "risk_level", "risk_level": "count"})
)

fig_pie = px.bar(
    risk_counts,
    x="risk_level",
    y="count",
    color="risk_level",
    title="Portfolio Risk Mix",
)

fig_pie.update_layout(
    bargap=0.4,
    showlegend=False,
)

fig_pie.update_traces(
    text=risk_counts["count"],
    textposition="outside"
)

st.plotly_chart(fig_pie, use_container_width=True)



# ---------------------------------------------------------
# PAGE 2 – INSIGHTS (EXECUTIVE LENS)
# ---------------------------------------------------------
elif page == "📊 Insights":
    st.title("📊 Insights & Governance Lens")

    st.subheader("① Portfolio Risk Mix – What story do the numbers tell?")
    st.markdown(
        """
        - **High risk agents** require focused control and predictable review cycles.  
        - **Medium risk agents** often drive efficiency but carry moderate dependencies.  
        - **Low risk agents** typically represent automation of low-impact workflows.  
        """
    )

    risk_counts = (
        df_filtered["risk_level"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "risk_level", "risk_level": "count"})
    )

    if risk_counts.empty:
        st.info("No data available under current filters.")
    else:
        fig_insight_pie = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            hole=0.5,
            title="Risk distribution across the portfolio",
        )
        fig_insight_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_insight_pie, use_container_width=True)

    st.markdown(
        """
        **Executive takeaway:**  
        - Ask: *Is the proportion of high-risk agents consistent with our risk appetite?*  
        - If high-risk share is growing, you may need to **tighten onboarding controls** or **raise review frequency**.
        """
    )

    # ---- Insight 2 – Autonomy vs Risk ----
    st.subheader("② Autonomy vs Risk Lens – Where should governance focus first?")
    risk_auto = (
        df_filtered.groupby(["risk_level", "autonomy_level"])
        .size()
        .reset_index(name="count")
    )

    if risk_auto.empty:
        st.info("No data available under current filters.")
    else:
        fig_insight_heat = px.density_heatmap(
            risk_auto,
            x="autonomy_level",
            y="risk_level",
            z="count",
            color_continuous_scale="Blues",
            title="Hot spots where autonomy and risk intersect",
        )
        fig_insight_heat.update_layout(margin=dict(l=40, r=10, t=40, b=40))
        st.plotly_chart(fig_insight_heat, use_container_width=True)

    st.markdown(
        """
        **Executive takeaway:**  
        - Focus governance attention on the **top-right quadrant** (HIGH RISK + higher autonomy).  
        - These agents should have: clear owners, **runbooks**, and **strong guardrails** (approvals, monitoring, rollbacks).
        """
    )

    # ---- Insight 3 – Overdue risk and cross-functional implication ----
    st.subheader("③ Overdue Reviews – Translating gaps into business impact")
    overdue_df = df_sched[df_sched["days_to_next"] < 0].copy()
    num_overdue = len(overdue_df)

    if num_overdue == 0:
        st.success("All agents are within their configured review windows under current filters.")
    else:
        overdue_owners = sorted(overdue_df["owner"].unique())
        owners_str = ", ".join(overdue_owners)

        st.markdown(
            f"""
            - **{num_overdue} agents are overdue for review.**  
            - They span **{owners_str}**, indicating a **cross-functional governance gap** rather than a local issue.  
            - In a real deployment, this would translate into:  
              - Potential **compliance findings** if audited.  
              - Need for **automated escalation workflows** (e.g., Slack/JIRA/ServiceNow) triggered by overdue status.  
            """
        )

    st.markdown(
        """
        ---
        These insights set you up for the remaining pages:
        - **Agents Table** – operational slice and dice.  
        - **Agent Detail** – drill-down into a single high-risk agent.  
        - **Lifecycle Timeline** – how agents move from request → deploy → retire.  
        - **Policy Simulator** – “what-if” adjustments to cadence, autonomy, and risk.
        """
    )
