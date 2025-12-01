import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime, timedelta

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Agent Governance Portal",
    layout="wide",
)

# -------------------------------------------------------
# Constants
# -------------------------------------------------------

RISK_SCORE_MAP = {
    "HIGH RISK": 3,
    "MEDIUM RISK": 2,
    "LOW RISK": 1,
}

AUTONOMY_WEIGHT = {
    "AUTO_ALLOWED": 2.0,
    "LIMITED_AUTONOMY": 1.6,
    "HUMAN_IN_LOOP": 1.3,
    "NO_AUTONOMY": 1.0,
}

AUTONOMY_ORDER = [
    "AUTO_ALLOWED",
    "LIMITED_AUTONOMY",
    "HUMAN_IN_LOOP",
    "NO_AUTONOMY",
]

LIFECYCLE_COLORS = {
    "DEPLOYED": "#2ca02c",      # green
    "TESTING": "#1f77b4",       # blue
    "PILOT": "#17becf",         # teal
    "RETIRED": "#7f7f7f",       # grey
    "DEPRECATED": "#d62728",    # red
    "ARCHIVED": "#9467bd",      # purple
}

REVIEW_OFFSETS_DAYS = {
    "Immediate": 1,
    "IMMEDIATE": 1,
    "Monthly": 30,
    "MONTHLY": 30,
    "Quarterly": 90,
    "QUARTERLY": 90,
    "Semi-Annual": 180,
    "SEMI-ANNUAL": 180,
    "Annual": 365,
    "ANNUAL": 365,
}

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------


def compute_risk_score(row) -> float:
    """
    Composite governance risk score (1–10) combining:
    - risk_level (HIGH / MEDIUM / LOW)
    - autonomy_level (AUTO / LIMITED / HUMAN / NO)
    - how overdue the review is (negative days_to_next_review)
    """
    base = RISK_SCORE_MAP.get(str(row.get("risk_level", "")).upper(), 0)

    auto_factor = AUTONOMY_WEIGHT.get(
        str(row.get("autonomy_level", "")).upper(), 1.0
    )
    days_to_next = row.get("days_to_next_review", 0)

    overdue_factor = 1.0
    if pd.notna(days_to_next) and days_to_next < 0:
        # up to +100% boost if very overdue
        overdue_factor = 1.0 + min(abs(int(days_to_next)) / 60.0, 1.0)

    score = base * auto_factor * overdue_factor
    return float(round(min(score, 10.0), 1))


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None) -> pd.DataFrame:
    """
    Load governance_decisions.json from either:
    - User upload (Streamlit sidebar), or
    - Local file in the repo.
    Ensures required columns exist and are normalized.
    """
    try:
        if uploaded_file is not None:
            data = json.load(uploaded_file)
        else:
            with open("governance_decisions.json", "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Ensure expected columns exist
    required_cols = [
        "agent_name",
        "owner",
        "created_by",
        "risk_level",
        "autonomy_level",
        "review_cadence",
        "lifecycle_state",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Optional text columns for governance reasoning
    optional_cols = ["reasoning", "autonomy_reasoning", "action"]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = ""

    # Normalize fields
    df["agent_name"] = df["agent_name"].astype(str)
    df["owner"] = df["owner"].astype(str)
    df["created_by"] = df["created_by"].astype(str)

    df["risk_level"] = df["risk_level"].astype(str).str.upper()
    df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()
    df["review_cadence"] = df["review_cadence"].astype(str)
    df["lifecycle_state"] = df["lifecycle_state"].astype(str).str.upper()

    # Risk score base (without overdue)
    df["risk_score_base"] = df["risk_level"].map(RISK_SCORE_MAP).fillna(0)

    # Autonomy sort order for visuals/tables
    df["autonomy_sort"] = df["autonomy_level"].apply(
        lambda x: AUTONOMY_ORDER.index(x) if x in AUTONOMY_ORDER else len(AUTONOMY_ORDER)
    )

    # Default lifecycle_state if missing
    df.loc[df["lifecycle_state"] == "", "lifecycle_state"] = "DEPLOYED"

    return df


def add_synthetic_review_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds synthetic last_reviewed and next_review_due dates.

    This is for demo / portfolio only – safe even if your JSON has no date columns.
    """
    df = df.copy()
    today = pd.Timestamp(datetime.utcnow().date())

    # Spread last_reviewed dates over the last ~6 months
    idx = df.reset_index().index
    last_reviewed = today - pd.to_timedelta((idx % 12 + 1) * 15, unit="D")
    df["last_reviewed"] = last_reviewed

    # Compute next_review_due based on cadence
    offsets = []
    for cadence in df["review_cadence"]:
        off = REVIEW_OFFSETS_DAYS.get(str(cadence), 90)  # default quarterly
        offsets.append(off)
    offsets = pd.to_timedelta(offsets, unit="D")
    df["next_review_due"] = df["last_reviewed"] + offsets

    df["days_to_next_review"] = (df["next_review_due"] - today).dt.days

    # Derived governance features
    df["is_overdue"] = df["days_to_next_review"] < 0
    df["governance_risk_score"] = df.apply(compute_risk_score, axis=1)

    return df


def build_lifecycle_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a synthetic lifecycle timeline per agent.

    Since we don't have real lifecycle dates, we generate
    nice-looking but fake dates based on index + state.
    """
    base = pd.Timestamp(datetime.utcnow().date())
    rows = []

    duration_by_state = {
        "DEPLOYED": 120,
        "TESTING": 45,
        "PILOT": 60,
        "RETIRED": 30,
        "DEPRECATED": 20,
        "ARCHIVED": 10,
    }

    for i, row in df.reset_index().iterrows():
        state = str(row["lifecycle_state"]).upper()
        agent = row["agent_name"]

        # Stagger start dates back in time
        start_offset_days = (i % 10 + 1) * 12
        start_date = base - timedelta(days=start_offset_days)

        duration = duration_by_state.get(state, 60)
        end_date = start_date + timedelta(days=duration)

        rows.append(
            {
                "agent_name": agent,
                "lifecycle_state": state,
                "start": start_date,
                "end": end_date,
            }
        )

    tl_df = pd.DataFrame(rows)
    return tl_df


def render_kpis(df: pd.DataFrame):
    total_agents = len(df)
    high_risk = (df["risk_level"] == "HIGH RISK").sum()
    medium_risk = (df["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (df["risk_level"] == "LOW RISK").sum()

    no_auto = (df["autonomy_level"] == "NO_AUTONOMY").sum()
    human_loop = (df["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited_auto = (df["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (df["autonomy_level"] == "AUTO_ALLOWED").sum()

    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)

    with col1:
        st.metric("Total Agents", total_agents)
    with col2:
        st.metric("High Risk", high_risk)
    with col3:
        st.metric("Medium Risk", medium_risk)
    with col4:
        st.metric("Low Risk", low_risk)

    with col5:
        st.metric("No Autonomy", no_auto)
    with col6:
        st.metric("Human-in-Loop", human_loop)
    with col7:
        st.metric("Limited Autonomy", limited_auto)
    with col8:
        st.metric("Auto Allowed", auto_allowed)


# -------------------------------------------------------
# Sidebar – controls
# -------------------------------------------------------

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json", type=["json"]
)

df = load_data(uploaded_file)
if df.empty:
    st.stop()

df = add_synthetic_review_dates(df)

st.sidebar.success(f"Loaded {len(df)} agents")

# Filters
risk_filter = st.sidebar.selectbox(
    "Filter by Risk Level",
    options=["All"] + sorted(df["risk_level"].dropna().unique().tolist()),
    index=0,
)

auto_filter = st.sidebar.selectbox(
    "Filter by Autonomy Level",
    options=["All"] + AUTONOMY_ORDER,
    index=0,
)

life_filter = st.sidebar.selectbox(
    "Filter by Lifecycle State",
    options=["All"] + sorted(df["lifecycle_state"].dropna().unique().tolist()),
    index=0,
)

# Apply filters
filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]

if auto_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == auto_filter]

if life_filter != "All":
    filtered = filtered[filtered["lifecycle_state"] == life_filter]

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📈 Lifecycle Timeline",
        "📋 Agents Table",
        "🔍 Agent Detail",
        "💡 Insights",
    ],
)

# -------------------------------------------------------
# PAGE: Overview
# -------------------------------------------------------
if page == "🏠 Overview":
    st.title("🛡️ AI Agent Governance Portal")
    st.caption(
        "Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture."
    )

    # KPIs always reflect overall posture (not filtered)
    render_kpis(df)
    st.markdown("---")

    # ---------------------------------------------------
    # Upcoming Reviews Panel (uses filtered dataset)
    # ---------------------------------------------------
    st.subheader("🔔 Upcoming Reviews (next 30 days)")

    reviews_df = filtered[
        [
            "agent_name",
            "owner",
            "risk_level",
            "review_cadence",
            "last_reviewed",
            "next_review_due",
            "days_to_next_review",
        ]
    ].copy()

    upcoming = reviews_df[reviews_df["days_to_next_review"].between(0, 30)]

    if upcoming.empty:
        st.info(
            "No agents have reviews due in the next 30 days "
            "(based on synthetic review dates)."
        )
    else:
        upcoming = upcoming.sort_values("days_to_next_review")
        upcoming["next_review_due"] = upcoming["next_review_due"].dt.date
        upcoming["last_reviewed"] = upcoming["last_reviewed"].dt.date
        st.dataframe(
            upcoming.rename(
                columns={
                    "agent_name": "Agent",
                    "owner": "Owner",
                    "risk_level": "Risk Level",
                    "review_cadence": "Cadence",
                    "last_reviewed": "Last Reviewed",
                    "next_review_due": "Next Review Due",
                    "days_to_next_review": "Days to Next",
                }
            ),
            use_container_width=True,
        )

        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                "📄 Export upcoming reviews (CSV)",
                upcoming.to_csv(index=False),
                file_name="upcoming_reviews.csv",
            )
        with colB:
            st.download_button(
                "📄 Export upcoming reviews (JSON)",
                upcoming.to_json(orient="records", indent=2),
                file_name="upcoming_reviews.json",
            )

    # ---------------------------------------------------
    # Overdue & horizon summary (Director lens)
    # ---------------------------------------------------
    overdue_df = df[df["is_overdue"]]
    num_overdue = len(overdue_df)

    upcoming_0_7 = df[df["days_to_next_review"].between(0, 7)]
    upcoming_8_30 = df[df["days_to_next_review"].between(8, 30)]

    st.markdown("### Governance posture at a glance")

    if num_overdue == 0:
        st.success(
            "✅ **No agents are currently overdue for review.** "
            "This reduces immediate audit and compliance exposure."
        )
    else:
        owners = ", ".join(sorted(overdue_df["owner"].unique()))
        st.error(
            f"⚠️ **{num_overdue} agents are overdue for review** "
            f"across **{owners}** — this represents cross-functional governance risk."
        )

    st.markdown(
        f"""
- **0–7 days horizon**: {len(upcoming_0_7)} agents will need attention **this week**.  
- **8–30 days horizon**: {len(upcoming_8_30)} agents are in the **planning runway** — ideal for batching reviews by owner or business unit.  
- Use this to drive **review SLAs**, escalation rules, and dashboards for Security, HR, and IT.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1.4])

    # ---------------------------------------------------
    # Risk vs Autonomy Heatmap
    # ---------------------------------------------------
    with col1:
        st.subheader("Risk vs Autonomy Heatmap")

        heat = (
            filtered.groupby(["risk_level", "autonomy_level"])
            .size()
            .reset_index(name="count")
        )

        if not heat.empty:
            heat_pivot = (
                heat.pivot(
                    index="risk_level",
                    columns="autonomy_level",
                    values="count",
                )
                .fillna(0)
            )

            fig = px.imshow(
                heat_pivot,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Autonomy Level", y="Risk Level", color="Agents"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available to generate heatmap with current filters.")

    # ---------------------------------------------------
    # Risk Breakdown – donut chart
    # ---------------------------------------------------
    with col2:
        st.subheader("Risk Breakdown")

        risk_counts = (
            filtered["risk_level"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "risk_level", "risk_level": "count"})
        )

        if not risk_counts.empty:
            fig2 = px.pie(
                risk_counts,
                names="risk_level",
                values="count",
                hole=0.45,
            )
            fig2.update_traces(textinfo="label+percent")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No risk data available for current filters.")

    st.markdown("---")

    # ---------------------------------------------------
    # Review Cadence Overview
    # ---------------------------------------------------
    st.subheader("Review Cadence Overview")
    if "review_cadence" in filtered.columns:
        cad = (
            filtered["review_cadence"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "review_cadence_label", "review_cadence": "count"})
        )

        if not cad.empty:
            fig3 = px.bar(
                cad,
                x="review_cadence_label",
                y="count",
                text="count",
            )
            fig3.update_layout(xaxis_title="", yaxis_title="Agents")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No review cadence data available for current filters.")
    else:
        st.info("No review_cadence column in JSON.")

    st.markdown("---")

    # ---------------------------------------------------
    # Lifecycle State Overview – COLOR CODED
    # ---------------------------------------------------
    st.subheader("Lifecycle State Overview")

    life_counts = (
        filtered["lifecycle_state"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "state", "lifecycle_state": "count"})
    )

    if not life_counts.empty:
        fig_life = px.bar(
            life_counts,
            x="state",
            y="count",
            text="count",
            color="state",
            color_discrete_map=LIFECYCLE_COLORS,
        )
        fig_life.update_layout(xaxis_title="State", yaxis_title="Agents", showlegend=False)
        st.plotly_chart(fig_life, use_container_width=True)
    else:
        st.info("No lifecycle_state data available.")

    st.markdown("---")

    # ---------------------------------------------------
    # High-Risk Agent Spotlight
    # ---------------------------------------------------
    with st.expander("🔥 High-Risk Agent Spotlight"):
        # Sort: overdue first, then by highest composite risk score
        spotlight = (
            df.sort_values(
                by=["is_overdue", "governance_risk_score"],
                ascending=[False, False],
            )
            .head(4)
        )

        if spotlight.empty:
            st.info("No high-risk agents to highlight at the moment.")
        else:
            for _, row in spotlight.iterrows():
                st.markdown(f"### {row['agent_name']}")

                col_left, col_right = st.columns([1.2, 2])

                with col_left:
                    st.markdown(f"**Owner:** {row.get('owner', '')}")
                    st.markdown(f"**Risk Level:** {row.get('risk_level', '')}")
                    st.markdown(f"**Autonomy:** {row.get('autonomy_level', '')}")
                    st.markdown(f"**Lifecycle:** {row.get('lifecycle_state', '')}")
                    st.markdown(
                        f"**Governance Risk Score:** `{row.get('governance_risk_score', '')}` / 10"
                    )

                    last_rev = row.get("last_reviewed", "")
                    next_rev = row.get("next_review_due", "")
                    days_next = row.get("days_to_next_review", None)

                    st.markdown(
                        f"- **Last review (synthetic):** {str(last_rev)[:10]}  \n"
                        f"- **Next review due (synthetic):** {str(next_rev)[:10]}  \n"
                        f"- **Days to next review:** {days_next}"
                    )

                with col_right:
                    reasons = []

                    if row.get("is_overdue"):
                        reasons.append(
                            f"Review is **overdue by {abs(int(row['days_to_next_review']))} days**."
                        )

                    if str(row.get("risk_level", "")).upper() == "HIGH RISK":
                        reasons.append("Classified as **HIGH RISK** in the portfolio.")

                    auto = str(row.get("autonomy_level", ""))
                    if auto == "AUTO_ALLOWED":
                        reasons.append(
                            "Runs in **fully autonomous mode** with minimal human checks."
                        )
                    elif auto == "LIMITED_AUTONOMY":
                        reasons.append(
                            "Operates with **limited autonomy**, still capable of independent actions."
                        )

                    lifecycle = str(row.get("lifecycle_state", ""))
                    if lifecycle == "DEPLOYED":
                        reasons.append(
                            "Already **deployed in production** and likely touching live systems/data."
                        )
                    elif lifecycle == "PILOT":
                        reasons.append(
                            "In **pilot**, but patterns here often predict how production usage will behave."
                        )

                    st.markdown("**Why this agent is flagged:**")
                    if reasons:
                        st.markdown("\n".join(f"- {r}" for r in reasons))
                    else:
                        st.markdown("- High composite risk score based on current posture.")

                    st.markdown(
                        """
**Executive lens:**  
Treat this agent as part of your **priority backlog for risk reduction** — confirm owner accountability,
validate runbooks, and ensure the review cadence is enforced.
                        """
                    )

                st.markdown("---")


# -------------------------------------------------------
# PAGE: Lifecycle Timeline
# -------------------------------------------------------
elif page == "📈 Lifecycle Timeline":
    st.title("📈 Lifecycle Timeline")

    st.caption(
        "Synthetic lifecycle timeline per agent. "
        "In a real deployment, this would be driven by actual lifecycle dates "
        "from your systems of record."
    )

    if filtered.empty:
        st.info("No agents match the selected filters.")
    else:
        tl_df = build_lifecycle_timeline(filtered)

        fig_tl = px.timeline(
            tl_df,
            x_start="start",
            x_end="end",
            y="agent_name",
            color="lifecycle_state",
            color_discrete_map=LIFECYCLE_COLORS,
        )
        fig_tl.update_yaxes(autorange="reversed")
        fig_tl.update_layout(
            xaxis_title="Date",
            yaxis_title="Agent",
            legend_title="Lifecycle State",
        )

        st.plotly_chart(fig_tl, use_container_width=True)

        st.markdown(
            "_Note: dates are generated for demo purposes only to show how a lifecycle "
            "view would look at enterprise scale._"
        )


# -------------------------------------------------------
# PAGE: Agents Table
# -------------------------------------------------------
elif page == "📋 Agents Table":
    st.title("📋 Agents Table")
    st.caption("Filtered view based on sidebar selections.")

    render_kpis(filtered)
    st.markdown("---")

    if filtered.empty:
        st.info("No agents match the selected filters.")
    else:
        hidden_cols = [
            "reasoning",
            "autonomy_reasoning",
            "action",
        ]

        display_cols = [c for c in filtered.columns if c not in hidden_cols]

        st.dataframe(
            filtered[display_cols].sort_values(
                by="governance_risk_score", ascending=False
            ),
            use_container_width=True,
            height=600,
        )


# -------------------------------------------------------
# PAGE: Agent Detail
# -------------------------------------------------------
elif page == "🔍 Agent Detail":
    st.title("🔍 Agent Detail View")
    st.caption("Deep dive for a single agent (filtered by sidebar controls).")

    if filtered.empty:
        st.info("No agents available for selected filters.")
    else:
        agent_list = filtered["agent_name"].unique().tolist()
        selected = st.selectbox("Choose an agent", agent_list)

        row = filtered[filtered["agent_name"] == selected].iloc[0]

        st.markdown(f"### {row['agent_name']}")

        c1, c2 = st.columns([1.4, 2])

        with c1:
            st.markdown(
                f"""
**Owner:** {row.get("owner", "")}  
**Created By:** {row.get("created_by", "")}  

**Risk Level:** {row.get("risk_level", "")}  
**Autonomy Level:** {row.get("autonomy_level", "")}  
**Lifecycle State:** {row.get("lifecycle_state", "")}  

**Review Cadence:** {row.get("review_cadence", "")}  
**Last Reviewed (synthetic):** {str(row.get("last_reviewed", ""))[:10]}  
**Next Review Due (synthetic):** {str(row.get("next_review_due", ""))[:10]}  

**Governance Risk Score:** {row.get("governance_risk_score", "")} / 10  
                """
            )

        with c2:
            st.markdown("#### Governance Notes")
            reasoning = row.get("reasoning", "").strip()
            if reasoning:
                st.write(reasoning)
            else:
                st.write("_No detailed reasoning captured in JSON._")

            auto_reason = row.get("autonomy_reasoning", "").strip()
            if auto_reason:
                st.markdown("#### Autonomy Reasoning")
                st.write(auto_reason)

            st.markdown("#### Recommended Action")
            action = row.get("action", "").strip()
            if action:
                st.info(action)
            else:
                st.info("_No explicit recommended action captured in JSON._")

        st.markdown("---")

        st.download_button(
            "📄 Export agent record (JSON)",
            row.to_json(indent=2),
            file_name=f"{selected}_record.json",
        )


# -------------------------------------------------------
# PAGE: Insights
# -------------------------------------------------------
elif page == "💡 Insights":
    st.title("💡 Governance Insights & Architecture")

    total_agents = len(df)
    high_risk_df = df[df["risk_level"] == "HIGH RISK"]
    medium_risk_df = df[df["risk_level"] == "MEDIUM RISK"]
    low_risk_df = df[df["risk_level"] == "LOW RISK"]

    overdue_df = df[df["is_overdue"]]
    num_overdue = len(overdue_df)

    st.subheader("Executive summary")

    owners_overdue = (
        ", ".join(sorted(overdue_df["owner"].unique())) if num_overdue > 0 else "—"
    )
    avg_score = (
        round(df["governance_risk_score"].mean(), 1)
        if "governance_risk_score" in df.columns
        else "n/a"
    )

    st.markdown(
        f"""
- You are currently governing **{total_agents} agents** across the portfolio.  
- **Risk mix**: {len(high_risk_df)} high risk, {len(medium_risk_df)} medium, {len(low_risk_df)} low.  
- **Composite governance risk score (avg)**: `{avg_score}` / 10.  
- **Overdue posture**: {num_overdue} agents have missed their review window (owners: {owners_overdue}).  
        """
    )

    st.markdown(
        """
These insights are designed to answer the question:

> *“Where should a Director of AI / Security focus **this quarter** to reduce risk and increase confidence?”*
        """
    )

    # ======================================================
    # Insight 1 – Portfolio Risk Mix
    # ======================================================
    st.subheader("① Portfolio risk mix")

    st.markdown(
        """
- High risk agents typically map to **regulated data, production access, or financial/HR workflows**.  
- Medium risk agents often sit in **operational or analytics** paths.  
- Low risk agents are ideal candidates for **autonomy experiments** and faster iteration.
        """
    )

    risk_counts_insight = (
        df["risk_level"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "risk_level", "risk_level": "count"})
    )

    if not risk_counts_insight.empty:
        fig_insight_risk = px.bar(
            risk_counts_insight,
            x="risk_level",
            y="count",
            color="risk_level",
            title="Risk distribution across agents",
            text="count",
        )
        fig_insight_risk.update_layout(xaxis_title="", yaxis_title="Agents")
        st.plotly_chart(fig_insight_risk, use_container_width=True)
    else:
        st.info("No risk data available to show portfolio mix.")

    st.markdown(
        """
**Executive takeaway:**  
Treat this view as your **risk budget**. Ask: *Do we have more HIGH RISK autonomy than our governance processes can support this quarter?*
        """
    )

    st.markdown("---")

    # ======================================================
    # Insight 2 – Autonomy vs Risk Lens
    # ======================================================
    st.subheader("② Autonomy vs risk lens")

    st.markdown(
        """
This view surfaces where **high autonomy collides with high risk** — the quadrant most likely to create
headline incidents or audit findings.
        """
    )

    risk_auto = (
        df.groupby(["risk_level", "autonomy_level"])
        .size()
        .reset_index(name="count")
    )

    if not risk_auto.empty:
        fig_insight_heat = px.density_heatmap(
            risk_auto,
            x="autonomy_level",
            y="risk_level",
            z="count",
            color_continuous_scale="Blues",
            title="Where autonomy and risk intersect",
        )
        st.plotly_chart(fig_insight_heat, use_container_width=True)
    else:
        st.info("No data available to render the autonomy vs risk heatmap.")

    st.markdown(
        """
**Executive takeaway:**  
Focus first on blocks in the **top-right** — HIGH RISK + AUTO_ALLOWED / LIMITED_AUTONOMY.
Those agents should have: clear owners, runbooks, emergency kill-switches, and stricter review cadences.
        """
    )

    st.markdown("---")

    # ======================================================
    # Insight 3 – Lifecycle posture
    # ======================================================
    st.subheader("③ Lifecycle posture")

    st.markdown(
        """
Lifecycle status shows **where your AI estate actually lives**:

- **TESTING / PILOT** → innovation pipeline  
- **DEPLOYED** → live production surface area  
- **DEPRECATED / ARCHIVED / RETIRED** → tail that still needs controlled decommissioning
        """
    )

    life_counts_insight = (
        df["lifecycle_state"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "state", "lifecycle_state": "count"})
    )

    if not life_counts_insight.empty:
        fig_life_insight = px.bar(
            life_counts_insight,
            x="state",
            y="count",
            text="count",
            color="state",
            title="Lifecycle states across agents",
        )
        fig_life_insight.update_layout(
            xaxis_title="", yaxis_title="Agents", showlegend=False
        )
        st.plotly_chart(fig_life_insight, use_container_width=True)
    else:
        st.info("No lifecycle data available for this portfolio.")

    st.markdown(
        """
**Executive takeaway:**  
Balance your **innovation funnel** (TESTING/PILOT) with your **operational load** (DEPLOYED).  
Too many deployed agents with thin governance = **operational fragility**. Too few pilots = **stalled innovation**.
        """
    )

    st.markdown("---")

    # ======================================================
    # Insight 4 – Policy enforcement & controls
    # ======================================================
    st.subheader("④ Policy enforcement snapshot")

    high_risk_overdue = df[(df["risk_level"] == "HIGH RISK") & (df["is_overdue"])]
    num_high_overdue = len(high_risk_overdue)

    st.markdown(
        f"""
- **High-risk agents overdue**: {num_high_overdue}  
- **Agents currently in AUTO_ALLOWED**: {len(df[df['autonomy_level'] == 'AUTO_ALLOWED'])}  
- **Agents with governance score ≥ 8**: {len(df[df['governance_risk_score'] >= 8])}
        """
    )

    st.markdown(
        """
Use these as **default policy prompts** when you show the demo:

- High-risk + overdue → auto-escalate to **Security** and lock changes until review.  
- High autonomy + high score → require **Human-in-Loop** until risk score drops.  
- Agents drifting in DEPLOYED without recent review → trigger **IT Admin / HR portal workflows**.
        """
    )

    st.markdown("---")

    # ======================================================
    # Insight 5 – High-level Architecture (Mermaid)
    # ======================================================
    st.subheader("⑤ High-level architecture (Mermaid)")

    st.markdown(
        """
This shows how the **governance layer** becomes a control tower feeding downstream portals
(Security, HR, IT Admin).
        """
    )

    mermaid_snippet = """
flowchart LR
    subgraph DataLayer[Data Sources]
        GJSON[governance_decisions.json<br/>(GitHub / S3 / DB)]
    end

    subgraph GovernancePortal[AI Agent Governance Portal<br/>(Streamlit)]
        UI[Executive Dashboards & Filters]
        Rules[Risk & Policy Engine]
        Reviews[Review Scheduler<br/>(cadence & next-review calc)]
        Audit[Audit Log & Export]
    end

    subgraph DownstreamPortals[Operational Portals]
        SecPortal[Security & Compliance Portal]
        HRPortal[HR / Business Portal]
        ITPortal[IT Admin / DevOps Portal]
    end

    subgraph Integrations[Enterprise Integrations]
        IAM[IAM / PAM Systems]
        ITSM[ITSM / Ticketing]
        GRC[GRC / Compliance Repository]
        Store[Data Lake / Audit Store]
    end

    GJSON --> GovernancePortal

    GovernancePortal --> SecPortal
    GovernancePortal --> HRPortal
    GovernancePortal --> ITPortal

    Rules --> IAM
    Rules --> GRC
    Reviews --> ITSM
    Audit --> Store
    """

    st.code(mermaid_snippet, language="mermaid")
