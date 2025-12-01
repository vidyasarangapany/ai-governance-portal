import json
from io import StringIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Agent Governance Portal",
    page_icon="🛡️",
    layout="wide",
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
SAMPLE_JSON = """
[
  {
    "agent_name": "InfraBot",
    "owner": "DevOps",
    "created_by": "DevOps",
    "risk_level": "HIGH RISK",
    "autonomy_level": "HUMAN_IN_LOOP",
    "review_cadence": "Quarterly",
    "lifecycle_state": "DEPLOYED"
  },
  {
    "agent_name": "SupportGenie",
    "owner": "Support",
    "created_by": "Support",
    "risk_level": "MEDIUM RISK",
    "autonomy_level": "LIMITED_AUTONOMY",
    "review_cadence": "Monthly",
    "lifecycle_state": "PILOT"
  }
]
"""


CADENCE_DAYS = {
    "Immediate": 7,        # treat as weekly checks
    "Monthly": 30,
    "Quarterly": 90,
    "Semi-Annual": 180,
    "Annual": 365,
}


def load_json(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_json(StringIO(SAMPLE_JSON))

    # Standardize column names just in case
    df = df.rename(
        columns={
            "Agent Name": "agent_name",
            "Owner": "owner",
            "Created By": "created_by",
            "Risk Level": "risk_level",
            "Autonomy Level": "autonomy_level",
            "Review Cadence": "review_cadence",
            "Lifecycle State": "lifecycle_state",
        }
    )
    return df


def augment_with_review_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic last_reviewed, next_review_due, days_to_next."""
    if df.empty:
        return df

    today = datetime.utcnow().date()
    df = df.copy()

    cadence_days = df["review_cadence"].map(CADENCE_DAYS).fillna(90).astype(int)

    # Synthetic last review ~ half a cadence ago
    df["last_reviewed"] = [
        today - timedelta(days=int(d // 2)) for d in cadence_days
    ]

    df["next_review_due"] = [
        lr + timedelta(days=int(cd))
        for lr, cd in zip(df["last_reviewed"], cadence_days)
    ]

    df["days_to_next"] = (df["next_review_due"] - today).dt.days

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    risk_levels = ["All"] + sorted(df["risk_level"].unique().tolist())
    autonomy_levels = ["All"] + sorted(df["autonomy_level"].unique().tolist())
    lifecycle_states = ["All"] + sorted(df["lifecycle_state"].unique().tolist())

    st.sidebar.markdown("### Filter by Risk Level")
    risk_filter = st.sidebar.selectbox("", risk_levels, index=0)

    st.sidebar.markdown("### Filter by Autonomy Level")
    aut_filter = st.sidebar.selectbox("", autonomy_levels, index=0)

    st.sidebar.markdown("### Filter by Lifecycle State")
    life_filter = st.sidebar.selectbox("", lifecycle_states, index=0)

    df_filtered = df.copy()
    if risk_filter != "All":
        df_filtered = df_filtered[df_filtered["risk_level"] == risk_filter]
    if aut_filter != "All":
        df_filtered = df_filtered[df_filtered["autonomy_level"] == aut_filter]
    if life_filter != "All":
        df_filtered = df_filtered[df_filtered["lifecycle_state"] == life_filter]

    return df_filtered


def metric_block(label: str, value, helper: str = ""):
    st.metric(label, value)
    if helper:
        st.caption(helper)


# ------------------------------------------------------------
# Layout – Sidebar
# ------------------------------------------------------------
st.sidebar.markdown("### Upload governance_decisions.json")
uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json", type=["json"], label_visibility="collapsed"
)

df_raw = load_json(uploaded_file)
df = augment_with_review_dates(df_raw)

st.sidebar.success(f"Loaded {len(df)} agents")

# Global filters
df_filtered = apply_filters(df)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigate")
page = st.sidebar.radio(
    "",
    [
        "Overview",
        "Insights",
        "Agents Table",
        "Agent Detail",
        "Lifecycle Timeline",
        "Policy Simulator",
    ],
    index=0,
)


# ------------------------------------------------------------
# Overview Page
# ------------------------------------------------------------
if page == "Overview":
    st.title("AI Agent Governance Portal")

    st.caption(
        "Executive dashboard for AI agent risk, autonomy, lifecycle, and overall governance posture."
    )

    # ---------- Top metrics ----------
    total_agents = len(df_filtered)
    high_risk = (df_filtered["risk_level"] == "HIGH RISK").sum()
    medium_risk = (df_filtered["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (df_filtered["risk_level"] == "LOW RISK").sum()

    no_autonomy = (df_filtered["autonomy_level"] == "NO_AUTONOMY").sum()
    human_in_loop = (df_filtered["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited_autonomy = (df_filtered["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (df_filtered["autonomy_level"] == "AUTO_ALLOWED").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_block("Total Agents", total_agents)
    with c2:
        metric_block("High Risk", high_risk)
    with c3:
        metric_block("Medium Risk", medium_risk)
    with c4:
        metric_block("Low Risk", low_risk)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_block("No Autonomy", no_autonomy)
    with c6:
        metric_block("Human-in-Loop", human_in_loop)
    with c7:
        metric_block("Limited Autonomy", limited_autonomy)
    with c8:
        metric_block("Auto Allowed", auto_allowed)

    st.markdown("---")

    # ---------- Upcoming reviews (next 30 days) ----------
    st.subheader("Upcoming Reviews (next 30 days)")

    upcoming = df_filtered[df_filtered["days_to_next"] <= 30].copy()
    upcoming = upcoming.sort_values("days_to_next")
    cols_order = [
        "agent_name",
        "owner",
        "risk_level",
        "review_cadence",
        "last_reviewed",
        "next_review_due",
        "days_to_next",
    ]
    existing = [c for c in cols_order if c in upcoming.columns]
    st.dataframe(upcoming[existing], use_container_width=True)

    # Export buttons
    csv = upcoming.to_csv(index=False).encode("utf-8")
    json_bytes = upcoming.to_json(orient="records", indent=2).encode("utf-8")

    cex1, cex2 = st.columns(2)
    with cex1:
        st.download_button(
            "⬇ Export upcoming reviews (CSV)",
            data=csv,
            file_name="upcoming_reviews.csv",
            mime="text/csv",
        )
    with cex2:
        st.download_button(
            "⬇ Export upcoming reviews (JSON)",
            data=json_bytes,
            file_name="upcoming_reviews.json",
            mime="application/json",
        )

    # ---------- Governance posture banner ----------
    st.markdown("## Governance posture at a glance")

    overdue = df_filtered[df_filtered["days_to_next"] < 0]
    if not overdue.empty:
        owners = sorted(overdue["owner"].unique().tolist())
        owners_text = ", ".join(owners)
        st.warning(
            f"⚠ {len(overdue)} agents are overdue for review across {owners_text} "
            "— this represents cross-functional governance risk."
        )
    else:
        st.success("✅ No agents are currently overdue for review.")

    # Mini executive lens
    horizon_0_7 = df_filtered[(df_filtered["days_to_next"] >= 0) & (df_filtered["days_to_next"] <= 7)]
    horizon_8_30 = df_filtered[(df_filtered["days_to_next"] > 7) & (df_filtered["days_to_next"] <= 30)]

    st.markdown(
        f"""
- **0–7 day horizon:** {len(horizon_0_7)} agents will need attention **this week**.
- **8–30 day horizon:** {len(horizon_8_30)} agents are in the **planning runway** — ideal for batching reviews by owner or business unit.
- Use this to drive **review SLAs**, automated escalations, and dashboards for Security, HR, and IT.
"""
    )

    # ---------- Risk vs Autonomy heatmap & Risk breakdown ----------
    st.markdown("---")
    st.subheader("Risk vs Autonomy view")

    c_heat, c_pie = st.columns([2, 1])

    with c_heat:
        if not df_filtered.empty:
            risk_auto = (
                df_filtered.groupby(["risk_level", "autonomy_level"])
                .size()
                .reset_index(name="count")
            )
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
        else:
            st.info("No data available for heatmap.")

    with c_pie:
        if not df_filtered.empty:
            risk_counts = df_filtered["risk_level"].value_counts()
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=risk_counts.index.tolist(),
                        values=risk_counts.values.tolist(),
                        hole=0.45,
                    )
                ]
            )
            fig_pie.update_layout(title="Risk distribution", showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available for risk breakdown.")

    # ---------- High-risk spotlight ----------
    st.markdown("---")
    st.subheader("High-risk agent spotlight")

    high = df_filtered[df_filtered["risk_level"] == "HIGH RISK"].copy()
    high = high.sort_values("days_to_next").head(4)

    if high.empty:
        st.info("No high-risk agents under the current filters.")
    else:
        for _, row in high.iterrows():
            st.markdown(f"### {row['agent_name']}")
            st.markdown(
                f"- **Owner:** {row['owner']}\n"
                f"- **Risk Level:** {row['risk_level']}\n"
                f"- **Autonomy:** {row['autonomy_level']}\n"
                f"- **Lifecycle:** {row['lifecycle_state']}\n"
                f"- **Review cadence:** {row['review_cadence']}\n"
                f"- **Next review due:** {row['next_review_due']} "
                f"({row['days_to_next']} days from now)"
            )
            st.caption(
                "Executive lens: high-risk + higher autonomy agents should have clear owners, "
                "runbooks, and sharply-defined review cadences."
            )
            st.markdown("---")


# ------------------------------------------------------------
# Insights Page
# ------------------------------------------------------------
elif page == "Insights":
    st.title("Insights & Governance Lens")

    if df_filtered.empty:
        st.info("No data under current filters. Reset filters to see insights.")
    else:
        # ---------- Insight 1: Portfolio risk mix ----------
        st.subheader("Insight 1 – Portfolio risk mix")

        risk_counts = (
            df_filtered["risk_level"].value_counts().reset_index()
        )
        risk_counts.columns = ["risk_level", "count"]

        fig_bar = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            title="Risk distribution across agents",
            text_auto=True,
        )
        fig_bar.update_layout(margin=dict(l=40, r=10, t=40, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(
            """
**Executive takeaway:**  
- High-risk agents require focused control and predictable review cycles.  
- Medium risk agents often drive operational efficiency but carry moderate dependencies.  
- Low risk agents typically represent automation of low-impact workflows.
"""
        )

        st.markdown("---")

        # ---------- Insight 2: Autonomy vs risk lens ----------
        st.subheader("Insight 2 – Autonomy vs risk lens")

        risk_auto = (
            df_filtered.groupby(["risk_level", "autonomy_level"])
            .size()
            .reset_index(name="count")
        )

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

        st.markdown(
            """
**Executive takeaway:**  
Focus first on the **top-right** (HIGH RISK + high autonomy).  
Those agents should have: named owners, clear runbooks, strong guardrails, and tighter review SLAs.
"""
        )

        st.markdown("---")

        # ---------- Insight 3: Three-portal architecture ----------
        st.subheader("Insight 3 – Governance layer across Security, HR, and IT")

        st.markdown(
            """
This portal is designed to sit **above** three operational portals:

- **Security / Compliance portal** – policies, access control, data classification.  
- **HR / Business portal** – human-in-loop approvals, accountable owners, exception handling.  
- **IT / Operations portal** – provisioning, observability, rollout / rollback.

You can represent this as a simple architecture diagram (for Mermaid or any diagramming tool):

```mermaid
flowchart LR
    Governance["AI Governance Portal"]
    Sec["Security / Compliance Portal"]
    HRP["HR / Business Portal"]
    ITP["IT / Ops Portal"]

    Governance --> Sec
    Governance --> HRP
    Governance --> ITP
