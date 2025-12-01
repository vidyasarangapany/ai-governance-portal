import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px

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

AUTONOMY_ORDER = [
    "AUTO_ALLOWED",
    "HUMAN_IN_LOOP",
    "LIMITED_AUTONOMY",
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
    "Immediate": 0,
    "IMMEDIATE": 0,
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


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()


@st.cache_data(show_spinner=False)
def load_data_from_file(uploaded_file) -> pd.DataFrame:
    """
    Load governance_decisions.json from either:
    - User upload (Streamlit sidebar), or
    - Local file in the repo.
    """

    if uploaded_file is not None:
        data = json.load(uploaded_file)
    else:
        path = Path("governance_decisions.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    df = pd.DataFrame(data)

    # Ensure expected columns exist
    for col in [
        "agent_name",
        "owner",
        "created_by",
        "risk_level",
        "autonomy_level",
        "review_cadence",
        "lifecycle_state",
    ]:
        if col not in df.columns:
            df[col] = ""

    # Normalise text fields
    df["risk_level"] = df["risk_level"].astype(str).str.upper()
    df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()
    df["review_cadence"] = df["review_cadence"].astype(str)
    df["lifecycle_state"] = df["lifecycle_state"].astype(str).str.upper()
    df["agent_name"] = df["agent_name"].astype(str)

    # Risk score
    df["risk_score"] = df["risk_level"].map(RISK_SCORE_MAP).fillna(0)

    # Autonomy sort order
    df["autonomy_sort"] = df["autonomy_level"].apply(
        lambda x: AUTONOMY_ORDER.index(x) if x in AUTONOMY_ORDER else len(AUTONOMY_ORDER)
    )

    # Lifecycle state default
    df.loc[df["lifecycle_state"] == "", "lifecycle_state"] = "DEPLOYED"

    return df


def add_synthetic_review_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds synthetic last_reviewed and next_review_due dates.

    This is for demo only – safe even if your JSON has no date columns.
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
        state = row["lifecycle_state"]
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

df = load_data_from_file(uploaded_file)
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
                    "risk_level": "Risk",
                    "review_cadence": "Cadence",
                    "last_reviewed": "Last Reviewed",
                    "next_review_due": "Next Review Due",
                    "days_to_next_review": "Days to Next",
                }
            ),
            use_container_width=True,
        )

    st.markdown("---")

    col1, col2 = st.columns([2, 1.5])

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
                risk_counts.copy(),
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
                cad.copy(),
                x="review_cadence_label",
                y="count",
                text="count",
            )
            fig3.update_layout(xaxis_title="", yaxis_title="count")
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
            life_counts.copy(),
            x="state",
            y="count",
            text="count",
            color="state",
            color_discrete_map=LIFECYCLE_COLORS,
        )
        fig_life.update_layout(xaxis_title="state", yaxis_title="count", showlegend=False)
        st.plotly_chart(fig_life, use_container_width=True)
    else:
        st.info("No lifecycle_state data available.")


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
            tl_df.copy(),
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
            filtered[display_cols].sort_values(by="risk_score", ascending=False),
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
        st.write(
            {
                "Owner": row.get("owner", ""),
                "Created By": row.get("created_by", ""),
                "Risk Level": row.get("risk_level", ""),
                "Autonomy Level": row.get("autonomy_level", ""),
                "Review Cadence": row.get("review_cadence", ""),
                "Lifecycle State": row.get("lifecycle_state", ""),
                "Last Reviewed (synthetic)": row.get("last_reviewed", ""),
                "Next Review Due (synthetic)": row.get("next_review_due", ""),
            }
        )

        st.markdown("#### Governance Notes")
        st.write(row.get("reasoning", ""))

        st.markdown("#### Recommended Action")
        st.write(row.get("action", ""))


# -------------------------------------------------------
# PAGE: Insights (Architecture diagram, etc.)
# -------------------------------------------------------
elif page == "💡 Insights":
    st.title("💡 Governance Insights & Architecture")

    st.subheader("Executive-ready talking points")
    st.markdown(
        """
- **Single governance layer**: This portal acts as the *control tower* for all AI agents across business units.
- **Risk + Autonomy + Lifecycle**: Every agent is tracked across *risk level*, *autonomy mode*, and *lifecycle state*.
- **Synthetic review cadence engine**: Today it's demo data, but real review schedules can plug in automatically.
- **Ready for enterprise integration**: Security, HR, and IT portals can integrate with this governance engine.

        """
    )

    st.markdown("---")
    st.subheader("High-level Architecture (Mermaid diagram)")

    st.markdown(
        """
Paste this into any Mermaid-compatible tool (e.g. mermaid.live, Notion, many slide tools) to generate a diagram:

```mermaid
flowchart LR
    subgraph DataLayer[Data Sources]
        GJSON[governance_decisions.json<br/>(GitHub / S3 / DB)]
    end

    subgraph GovernancePortal[AI Agent Governance Portal<br/>(Streamlit)]
        UI[Executive Dashboards & Filters]
        Rules[Risk & Policy Engine]
        Reviews[Review Scheduler<br/>(cadence, next-review calc)]
        Audit[Audit Log & Export]
    end

    subgraph DownstreamPortals[Operational Portals]
        SecPortal[Security & Compliance Portal]
        HRPortal[HR / Business Portal]
        ITPortal[IT Admin / DevOps Portal]
    end

    subgraph Integrations[Enterprise Integrations]
        IAM[ IAM / PAM Systems ]
        ITSM[ ITSM / Ticketing ]
        GRC[ GRC / Compliance Repository ]
        Store[ Data Lake / Audit Store ]
    end

    GJSON --> GovernancePortal

    GovernancePortal --> SecPortal
    GovernancePortal --> HRPortal
    GovernancePortal --> ITPortal

    Rules --> IAM
    Rules --> GRC
    Reviews --> ITSM
    Audit --> Store
