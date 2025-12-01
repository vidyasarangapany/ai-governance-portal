import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Governance Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Data Loading & Normalization
# -------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    """
    Load governance_decisions.json either from uploader or from repo.
    Normalize key columns and add risk_score.
    """
    try:
        if uploaded_file:
            data = json.load(uploaded_file)
        else:
            with open("governance_decisions.json", "r") as f:
                data = json.load(f)

        # Ensure list
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        if df.empty:
            return df

        # Normalize text columns
        if "risk_level" in df.columns:
            df["risk_level"] = df["risk_level"].astype(str).str.upper()

        if "autonomy_level" in df.columns:
            df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()

        # Map risk to numeric score (for sorting / insights)
        risk_map = {
            "LOW RISK": 1,
            "MEDIUM RISK": 2,
            "HIGH RISK": 3,
        }
        df["risk_score"] = df["risk_level"].map(risk_map).fillna(0)

        # Ensure we have an agent_name column for identification
        if "agent_name" not in df.columns:
            df["agent_name"] = df.get("name", df.index.astype(str))

        return df

    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return pd.DataFrame()


def add_lifecycle_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic, made-up lifecycle dates and state if missing.

    Lifecycle states in order:
    REQUESTED → APPROVED → DEPLOYED → UNDER_REVIEW → DEPROVISIONED

    Dates are deterministic per agent (no randomness) using a hash-like
    offset from a base date so they stay stable across reloads.
    """
    if df.empty:
        return df

    # Ensure lifecycle_state exists
    if "lifecycle_state" not in df.columns:
        df["lifecycle_state"] = "DEPLOYED"

    # Normalize lifecycle_state text
    df["lifecycle_state"] = df["lifecycle_state"].astype(str).str.upper()

    # Deterministic offsets based on agent_name
    base_date = pd.Timestamp("2024-01-01")

    def deterministic_offset(name: str) -> int:
        # Simple stable hash: sum of character codes (mod 60 to fit 2 months)
        return sum(ord(c) for c in str(name)) % 60

    offsets = df["agent_name"].astype(str).apply(deterministic_offset)

    df["requested_date"] = base_date + pd.to_timedelta(offsets, unit="D")
    df["approved_date"] = df["requested_date"] + pd.Timedelta(days=7)
    df["deployed_date"] = df["approved_date"] + pd.Timedelta(days=7)
    df["under_review_date"] = df["deployed_date"] + pd.Timedelta(days=14)
    df["deprovisioned_date"] = df["under_review_date"] + pd.Timedelta(days=14)

    # Mask out "future" dates based on current lifecycle_state
    lifecycle_order = [
        "REQUESTED",
        "APPROVED",
        "DEPLOYED",
        "UNDER_REVIEW",
        "DEPROVISIONED",
    ]
    state_to_index = {s: i for i, s in enumerate(lifecycle_order)}

    state_idx = df["lifecycle_state"].map(state_to_index).fillna(0).astype(int)

    lifecycle_cols = [
        "requested_date",
        "approved_date",
        "deployed_date",
        "under_review_date",
        "deprovisioned_date",
    ]

    for i, col in enumerate(lifecycle_cols):
        # If an agent is earlier than this lifecycle step, blank out the date
        df.loc[state_idx < i, col] = pd.NaT

    # Deterministic "today" for day-count view (keeps demo stable)
    today = pd.Timestamp("2025-01-01")
    df["lifecycle_day_count"] = (today - df["requested_date"]).dt.days

    return df


# -------------------------------------------------------
# Load data & enrich with lifecycle metadata
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

uploaded_json = st.sidebar.file_uploader(
    "Upload governance_decisions.json",
    type=["json"],
    help="Use this to refresh dashboard data without redeploying.",
)

df = load_data(uploaded_json)
df = add_lifecycle_metadata(df)

st.sidebar.success(f"Loaded {len(df)} agents")

if df.empty:
    st.warning("No data loaded. Please upload a governance_decisions.json file.")
    st.stop()

# -------------------------------------------------------
# Sidebar Filters
# -------------------------------------------------------
# Risk filter
risk_filter = st.sidebar.selectbox(
    "Filter by Risk Level",
    ["All"] + sorted(df["risk_level"].dropna().unique().tolist()),
)

# Autonomy filter
auto_filter = st.sidebar.selectbox(
    "Filter by Autonomy Level",
    ["All"] + sorted(df["autonomy_level"].dropna().unique().tolist()),
)

# Lifecycle state filter
lifecycle_filter = st.sidebar.selectbox(
    "Filter by Lifecycle State",
    ["All"] + sorted(df["lifecycle_state"].dropna().unique().tolist()),
)

# Department filter (if present)
dept_col = None
for c in df.columns:
    if c.lower() in ["department", "dept"]:
        dept_col = c
        break

if dept_col:
    dept_filter = st.sidebar.selectbox(
        "Filter by Department",
        ["All"] + sorted(df[dept_col].dropna().unique().tolist()),
    )
else:
    dept_filter = "All"

# Apply filters to create "filtered" view (used by table/timeline/detail)
filtered = df.copy()

if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]

if auto_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == auto_filter]

if lifecycle_filter != "All":
    filtered = filtered[filtered["lifecycle_state"] == lifecycle_filter]

if dept_filter != "All" and dept_col:
    filtered = filtered[filtered[dept_col] == dept_filter]

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
pages = [
    "Overview",
    "Agents Table",
    "Agent Detail",
    "Lifecycle Timeline",
    "Insights",
]


def nav_label(key: str) -> str:
    mapping = {
        "Overview": "🏠 Overview",
        "Agents Table": "📋 Agents Table",
        "Agent Detail": "🔍 Agent Detail",
        "Lifecycle Timeline": "📅 Lifecycle Timeline",
        "Insights": "💡 Insights",
    }
    return mapping.get(key, key)


page = st.sidebar.radio("Navigate", pages, format_func=nav_label)

# -------------------------------------------------------
# KPI SECTION
# -------------------------------------------------------
def render_kpis(source_df: pd.DataFrame):
    total_agents = len(source_df)
    high_risk = (source_df["risk_level"] == "HIGH RISK").sum()
    med_risk = (source_df["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (source_df["risk_level"] == "LOW RISK").sum()

    human_loop = (source_df["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited_auto = (source_df["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (source_df["autonomy_level"] == "AUTO_ALLOWED").sum()
    no_auto = (source_df["autonomy_level"] == "NO_AUTONOMY").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Agents", total_agents)
    c2.metric("High Risk", high_risk)
    c3.metric("Medium Risk", med_risk)
    c4.metric("Low Risk", low_risk)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("No Autonomy", no_auto)
    c6.metric("Human-in-Loop", human_loop)
    c7.metric("Limited Autonomy", limited_auto)
    c8.metric("Auto Allowed", auto_allowed)


# -------------------------------------------------------
# PAGE: Overview
# -------------------------------------------------------
if page == "Overview":
    st.title("🛡️ AI Agent Governance Portal")
    st.caption(
        "Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture."
    )

    # Use full df for KPIs so they always reflect global posture
    render_kpis(df)
    st.markdown("---")

    col1, col2 = st.columns([2, 1.5])

    # Risk vs Autonomy Heatmap
    with col1:
        st.subheader("Risk vs Autonomy Heatmap")
        heat = (
            df.groupby(["risk_level", "autonomy_level"])
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
            st.info("No data available to generate heatmap.")

    # Pie chart: Risk Breakdown
    with col2:
        st.subheader("Risk Breakdown")

        risk_counts = (
            df["risk_level"]
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
            st.info("No risk data available for breakdown.")

    st.markdown("---")

    # Review cadence overview
    st.subheader("Review Cadence Overview")
    if "review_cadence" in df.columns:
        cad = (
            df["review_cadence"]
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
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No review cadence data available.")
    else:
        st.info("No review_cadence column in JSON.")

    st.markdown("---")

    # Lifecycle State Overview
    st.subheader("Lifecycle State Overview")
    life_counts = (
        df["lifecycle_state"]
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
        )
        st.plotly_chart(fig_life, use_container_width=True)
    else:
        st.info("No lifecycle_state data available.")


# -------------------------------------------------------
# PAGE: Agents Table
# -------------------------------------------------------
elif page == "Agents Table":
    st.title("📋 Agents Table")
    st.caption("Filtered view based on sidebar selections.")

    render_kpis(filtered)
    st.markdown("---")

    if filtered.empty:
        st.info("No agents match the selected filters.")
    else:
        # Hide verbose reasoning columns by default in the grid
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
elif page == "Agent Detail":
    st.title("🔍 Agent Detail View")
    st.caption("Deep dive for a single agent (filtered by sidebar controls).")

    if filtered.empty:
        st.info("No agents available for selected filters.")
    else:
        agent_list = filtered["agent_name"].unique().tolist()
        selected = st.selectbox("Choose an agent", agent_list)

        row = filtered[filtered["agent_name"] == selected].iloc[0]

        st.markdown(f"### 🧩 {row['agent_name']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Level", row["risk_level"])
        c2.metric("Autonomy", row["autonomy_level"])
        c3.metric("Lifecycle State", row["lifecycle_state"])

        c4, c5 = st.columns(2)
        c4.metric("Review Cadence", row.get("review_cadence", "N/A"))
        c5.metric("Days Since Request", int(row.get("lifecycle_day_count", 0)))

        st.markdown("#### Lifecycle Dates")
        life_cols = [
            "requested_date",
            "approved_date",
            "deployed_date",
            "under_review_date",
            "deprovisioned_date",
        ]
        life_data = {
            col: row[col] if pd.notna(row[col]) else None for col in life_cols
        }
        life_df = pd.DataFrame(
            [
                {
                    "Requested": life_data["requested_date"],
                    "Approved": life_data["approved_date"],
                    "Deployed": life_data["deployed_date"],
                    "Under Review": life_data["under_review_date"],
                    "Deprovisioned": life_data["deprovisioned_date"],
                }
            ]
        )
        st.dataframe(life_df)

        st.markdown("#### Governance Reasoning")
        st.write(row.get("reasoning", "_No reasoning provided._"))

        if "autonomy_reasoning" in row and isinstance(
            row["autonomy_reasoning"], str
        ):
            st.markdown("#### Autonomy Reasoning")
            st.write(row["autonomy_reasoning"])

        st.markdown("#### Recommended Action")
        st.info(row.get("action", "_No action provided._"))


# -------------------------------------------------------
# PAGE: Lifecycle Timeline
# -------------------------------------------------------
elif page == "Lifecycle Timeline":
    st.title("📅 Lifecycle Timeline")
    st.caption(
        "Lifecycle journey from request to current state for all filtered agents."
    )

    if filtered.empty:
        st.info("No agents available for selected filters.")
    else:
        # Build a compact timeline dataframe
        timeline_cols = [
            "agent_name",
            "lifecycle_state",
            "requested_date",
            "approved_date",
            "deployed_date",
            "under_review_date",
            "deprovisioned_date",
        ]

        available_cols = [c for c in timeline_cols if c in filtered.columns]
        timeline_df = filtered[available_cols].copy()

        # Start at requested_date
        timeline_df["start"] = timeline_df["requested_date"]

        # End at the latest non-null lifecycle date
        date_cols_for_end = [
            c
            for c in [
                "deprovisioned_date",
                "under_review_date",
                "deployed_date",
                "approved_date",
                "requested_date",
            ]
            if c in timeline_df.columns
        ]
        timeline_df["end"] = timeline_df[date_cols_for_end].max(axis=1)

        # Drop rows with no valid dates
        timeline_df = timeline_df.dropna(subset=["start", "end"])

        if timeline_df.empty:
            st.info("No lifecycle date data available to build a timeline.")
        else:
            fig_timeline = px.timeline(
                timeline_df,
                x_start="start",
                x_end="end",
                y="agent_name",
                color="lifecycle_state",
                hover_data=[
                    "requested_date",
                    "approved_date",
                    "deployed_date",
                    "under_review_date",
                    "deprovisioned_date",
                ],
            )
            fig_timeline.update_yaxes(autorange="reversed")
            fig_timeline.update_layout(
                xaxis_title="Date",
                yaxis_title="Agent",
                legend_title="Lifecycle State",
            )
            st.plotly_chart(fig_timeline, use_container_width=True)


# -------------------------------------------------------
# PAGE: Insights
# -------------------------------------------------------
elif page == "Insights":
    st.title("💡 Key Governance Insights")
    st.caption("Summaries for executives, audit teams, and risk leaders.")

    render_kpis(df)
    st.markdown("---")

    # High-risk agents
    st.subheader("Top 5 Highest-Risk Agents")
    top5 = df.sort_values(by="risk_score", ascending=False).head(5)

    if top5.empty:
        st.info("No agents available for insights.")
    else:
        for _, r in top5.iterrows():
            st.markdown(
                f"**{r['agent_name']}** — {r['risk_level']}, {r['autonomy_level']}, {r['lifecycle_state']}"
            )
            if isinstance(r.get("reasoning", ""), str) and r["reasoning"]:
                st.write(r["reasoning"])
            if isinstance(r.get("action", ""), str) and r["action"]:
                st.markdown(f"**Action:** {r['action']}")
            st.markdown("---")

    # Missing owners
    st.subheader("Agents Missing Owners")
    owner_col = None
    for c in df.columns:
        if c.lower() == "owner":
            owner_col = c
            break

    if owner_col:
        missing = df[df[owner_col].isna() | (df[owner_col].astype(str) == "Unknown")]
        if missing.empty:
            st.success("All agents have an assigned owner.")
        else:
            for _, r in missing.iterrows():
                st.markdown(f"- {r['agent_name']} — {r['risk_level']}")
    else:
        st.info("No 'owner' column found in JSON.")
