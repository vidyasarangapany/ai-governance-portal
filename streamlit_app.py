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
    "DEPLOYED": "#2ca02c",
    "TESTING": "#1f77b4",
    "PILOT": "#17becf",
    "RETIRED": "#7f7f7f",
    "DEPRECATED": "#d62728",
    "ARCHIVED": "#9467bd",
}

REVIEW_OFFSETS_DAYS = {
    "IMMEDIATE": 0,
    "MONTHLY": 30,
    "QUARTERLY": 90,
    "SEMI-ANNUAL": 180,
    "ANNUAL": 365,
}

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def normalize(s):
    if not isinstance(s, str):
        return ""
    return s.strip()


@st.cache_data(show_spinner=False)
def load_data_from_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        data = json.load(uploaded_file)
    else:
        path = Path("governance_decisions.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    df = pd.DataFrame(data)

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

    df["risk_level"] = df["risk_level"].astype(str).str.upper()
    df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()
    df["review_cadence"] = df["review_cadence"].astype(str).str.upper()
    df["lifecycle_state"] = df["lifecycle_state"].astype(str).str.upper()
    df["agent_name"] = df["agent_name"].astype(str)

    df["risk_score"] = df["risk_level"].map(RISK_SCORE_MAP).fillna(0)

    df["autonomy_sort"] = df["autonomy_level"].apply(
        lambda x: AUTONOMY_ORDER.index(x) if x in AUTONOMY_ORDER else len(AUTONOMY_ORDER)
    )

    df.loc[df["lifecycle_state"] == "", "lifecycle_state"] = "DEPLOYED"

    return df


def add_synthetic_review_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = pd.Timestamp(datetime.utcnow().date())

    idx = df.reset_index().index
    df["last_reviewed"] = today - pd.to_timedelta((idx % 12 + 1) * 15, unit="D")

    offsets = df["review_cadence"].apply(
        lambda c: REVIEW_OFFSETS_DAYS.get(c.upper(), 90)
    )
    df["next_review_due"] = df["last_reviewed"] + pd.to_timedelta(offsets, unit="D")

    df["days_to_next_review"] = (df["next_review_due"] - today).dt.days
    return df


def build_lifecycle_timeline(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.Timestamp(datetime.utcnow().date())
    duration_map = {
        "DEPLOYED": 120,
        "TESTING": 45,
        "PILOT": 60,
        "RETIRED": 30,
        "DEPRECATED": 20,
        "ARCHIVED": 10,
    }

    rows = []
    for i, row in df.reset_index().iterrows():
        start = base - timedelta(days=(i % 10 + 1) * 12)
        duration = duration_map.get(row["lifecycle_state"], 60)

        rows.append(
            {
                "agent_name": row["agent_name"],
                "lifecycle_state": row["lifecycle_state"],
                "start": start,
                "end": start + timedelta(days=duration),
            }
        )
    return pd.DataFrame(rows)


def render_kpis(df: pd.DataFrame):
    total_agents = len(df)
    high_risk = (df["risk_level"] == "HIGH RISK").sum()
    medium_risk = (df["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (df["risk_level"] == "LOW RISK").sum()

    no_auto = (df["autonomy_level"] == "NO_AUTONOMY").sum()
    human_loop = (df["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited_auto = (df["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (df["autonomy_level"] == "AUTO_ALLOWED").sum()

    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    c1.metric("Total Agents", total_agents)
    c2.metric("High Risk", high_risk)
    c3.metric("Medium Risk", medium_risk)
    c4.metric("Low Risk", low_risk)

    c5.metric("No Autonomy", no_auto)
    c6.metric("Human-in-Loop", human_loop)
    c7.metric("Limited Autonomy", limited_auto)
    c8.metric("Auto Allowed", auto_allowed)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json", type=["json"]
)

df = load_data_from_file(uploaded_file)
df = add_synthetic_review_dates(df)

st.sidebar.success(f"Loaded {len(df)} agents")

risk_filter = st.sidebar.selectbox(
    "Filter by Risk Level",
    options=["All"] + sorted(df["risk_level"].unique().tolist()),
)

auto_filter = st.sidebar.selectbox(
    "Filter by Autonomy Level",
    options=["All"] + AUTONOMY_ORDER,
)

life_filter = st.sidebar.selectbox(
    "Filter by Lifecycle State",
    options=["All"] + sorted(df["lifecycle_state"].unique().tolist()),
)

filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if auto_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == auto_filter]
if life_filter != "All":
    filtered = filtered[filtered["lifecycle_state"] == life_filter]

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Lifecycle Timeline", "📋 Agents Table", "🔍 Agent Detail", "💡 Insights"],
)
# -------------------------------------------------------
# PAGE: Overview
# -------------------------------------------------------
if page == "🏠 Overview":
    st.title("🛡️ AI Agent Governance Portal")
    st.caption(
        "Executive dashboard for AI agent risk, autonomy, lifecycle, and overall governance posture."
    )

    # KPI strip is always global (unfiltered) for a consistent top-line story
    render_kpis(df)
    st.markdown("---")

    # ---------------------------------------------------
    # Upcoming Reviews (filtered)
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
            "No agents have reviews due in the next 30 days based on the synthetic schedule. "
            "That typically indicates you’re ahead of your review plan."
        )
    else:
        upcoming = upcoming.sort_values("days_to_next_review")
        upcoming["last_reviewed"] = upcoming["last_reviewed"].dt.date
        upcoming["next_review_due"] = upcoming["next_review_due"].dt.date

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

        # Export buttons for upcoming reviews
        col_exp1, col_exp2 = st.columns(2)
        csv_bytes = upcoming.to_csv(index=False).encode("utf-8")
        json_bytes = upcoming.to_json(orient="records", indent=2).encode("utf-8")

        with col_exp1:
            st.download_button(
                "⬇️ Export upcoming reviews (CSV)",
                data=csv_bytes,
                file_name="upcoming_reviews.csv",
                mime="text/csv",
            )
        with col_exp2:
            st.download_button(
                "⬇️ Export upcoming reviews (JSON)",
                data=json_bytes,
                file_name="upcoming_reviews.json",
                mime="application/json",
            )

    # Mini executive narrative
    st.markdown(
        """
**How to read this section**

- Focus on agents with reviews due in the next 0–7 days for immediate attention.  
- 8–30 day horizon is your planning runway – where you can batch reviews by owner or line of business.  
- No upcoming reviews typically means your cadence configuration is generous or you’ve just completed a review cycle.
        """
    )

    st.markdown("---")

    # ---------------------------------------------------
    # High-Risk Spotlight (collapsible)
    # ---------------------------------------------------
    with st.expander("🔥 High-Risk Agent Spotlight", expanded=True):
        high_risk_agents = (
            filtered[filtered["risk_level"] == "HIGH RISK"]
            .sort_values("risk_score", ascending=False)
            .head(4)
        )

        if high_risk_agents.empty:
            st.info(
                "No agents are currently tagged as **HIGH RISK** under the active filters. "
                "That’s the desired end-state, but double-check that risk labels are configured properly."
            )
        else:
            cols = st.columns(len(high_risk_agents))
            for col, (_, row) in zip(cols, high_risk_agents.iterrows()):
                with col:
                    st.markdown(f"### {row['agent_name']}")
                    st.markdown(
                        f"""
- **Owner:** `{row.get('owner', '')}`
- **Autonomy:** `{row.get('autonomy_level', '')}`
- **Lifecycle:** `{row.get('lifecycle_state', '')}`
- **Review Cadence:** `{row.get('review_cadence', '')}`
- **Next Review (synthetic):** `{row.get('next_review_due')}`
                        """
                    )

            st.markdown(
                """
**Executive lens**

Treat these agents as your priority backlog for risk reduction.  
They usually:

- Carry material financial, regulatory, or privacy impact.  
- Operate with elevated autonomy (limited or auto-allowed).  
- Deserve clear ownership and sharply defined review cadences.
                """
            )

    st.markdown("---")

    # ---------------------------------------------------
    # Risk vs Autonomy Heatmap + Risk Breakdown donut
    # ---------------------------------------------------
    col_heat, col_pie = st.columns([2, 1.4])

    with col_heat:
        st.subheader("Risk vs Autonomy Heatmap")

        heat = (
            filtered.groupby(["risk_level", "autonomy_level"])
            .size()
            .reset_index(name="count")
        )

        if heat.empty:
            st.info("No data available for the current filters.")
        else:
            heat_pivot = (
                heat.pivot(
                    index="risk_level",
                    columns="autonomy_level",
                    values="count",
                )
                .fillna(0)
                .sort_index()
            )
            heat_df = pd.DataFrame(heat_pivot)

            fig_heat = px.imshow(
                heat_df,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Autonomy Level", y="Risk Level", color="Agents"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

   # ---------------------------------------------------
# Risk Breakdown – donut chart (Narwhals-safe)
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

        # 🔥 Narwhals → Pandas conversion to avoid DuplicateError
        risk_pd = risk_counts.to_pandas()

        fig_pie = px.pie(
            risk_pd,
            names="risk_level",
            values="count",
            hole=0.45,
            color="risk_level",
            color_discrete_map={
                "HIGH RISK": "#d62728",
                "MEDIUM RISK": "#ff7f0e",
                "LOW RISK": "#2ca02c",
            },
        )

        fig_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.info("No risk data available for current filters.")


         # --- Risk Breakdown Pie (Narwhals-safe Pandas conversion) ---
if not risk_df.empty:
    risk_pd = risk_df.to_pandas()   # 🔥 CRITICAL FIX

    fig_pie = px.pie(
        risk_pd,
        names="risk_level",
        values="count",
        hole=0.45,
        color="risk_level",
        color_discrete_map={
            "HIGH RISK": "#d62728",
            "MEDIUM RISK": "#ff7f0e",
            "LOW RISK": "#2ca02c",
        },
    )
    fig_pie.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("No risk data available for current filters.")

            fig_pie.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        """
**Mini-report**

- The **heatmap** shows where risk and autonomy intersect – the darker the cell, the more agents with that combination.  
- The **donut chart** summarises overall risk posture under the active filters.  
Use these together to answer questions like: *“How many high-risk agents are still auto-allowed?”* or *“Where are we safe but overly manual?”*
        """
    )

    st.markdown("---")

    # ---------------------------------------------------
    # Review Cadence Overview
    # ---------------------------------------------------
    st.subheader("Review Cadence Overview")
    if "review_cadence" in filtered.columns:
        cad_counts = (
            filtered["review_cadence"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "review_cadence_label", "review_cadence": "count"})
        )

        if cad_counts.empty:
            st.info("No review cadence data available for the current filters.")
        else:
            cad_df = pd.DataFrame(cad_counts)

            fig_cad = px.bar(
                cad_df,
                x="review_cadence_label",
                y="count",
                text="count",
            )
            fig_cad.update_layout(xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig_cad, use_container_width=True)
    else:
        st.info("No review_cadence column found in the dataset.")

    st.markdown(
        """
**Cadence perspective**

- A **heavy cluster in Quarterly** is normal for most enterprises.  
- A large number of **Immediate or Monthly** agents usually signals either higher inherent risk or immature controls that still need tuning.  
- **Annual** should typically be reserved for low-risk, low-autonomy agents with strong upstream controls.
        """
    )

    st.markdown("---")

    # ---------------------------------------------------
    # Lifecycle State Overview
    # ---------------------------------------------------
    st.subheader("Lifecycle State Overview")

    life_counts = (
        filtered["lifecycle_state"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "state", "lifecycle_state": "count"})
    )

    if life_counts.empty:
        st.info("No lifecycle_state data available for the current filters.")
    else:
        life_df = pd.DataFrame(life_counts)

        fig_life = px.bar(
            life_df,
            x="state",
            y="count",
            text="count",
            color="state",
            color_discrete_map=LIFECYCLE_COLORS,
        )
        fig_life.update_layout(
            xaxis_title="State", yaxis_title="Agents", showlegend=False
        )
        st.plotly_chart(fig_life, use_container_width=True)

    st.markdown(
        """
**Lifecycle health-check**

- A balanced portfolio usually has a mix of **Testing, Pilot, and Deployed** agents.  
- A growing **Deprecated or Retired** count is healthy – it means you are actually turning things off.  
- If everything is **Deployed** and nothing ever retires, that is a long-term operational and compliance risk.
        """
    )

# -------------------------------------------------------
# PAGE: Lifecycle Timeline
# -------------------------------------------------------
elif page == "📈 Lifecycle Timeline":
    st.title("📈 Lifecycle Timeline")
    st.caption(
        "Synthetic lifecycle view for each agent. In production, these dates would come from your SDLC, MLOps, or ITSM systems."
    )

    if filtered.empty:
        st.info("No agents match the selected filters.")
    else:
        tl_df = build_lifecycle_timeline(filtered)
        tl_df = pd.DataFrame(tl_df)

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
            """
This is a **demo timeline** using generated dates to illustrate how a real lifecycle view would look.

In a production deployment you would typically wire this into:

- SDLC or MLOps pipelines (for build / deploy dates).  
- ITSM systems (for change tickets and approvals).  
- Decommission processes (for retirement and archival signals).
            """
        )

# -------------------------------------------------------
# PAGE: Agents Table
# -------------------------------------------------------
elif page == "📋 Agents Table":
    st.title("📋 Agents Table")
    st.caption("Tabular view of agents under the active filters. Ideal for export and bulk review planning.")

    # KPIs reflect filtered view here
    render_kpis(filtered)
    st.markdown("---")

    if filtered.empty:
        st.info("No agents match the selected filters.")
    else:
        hidden_cols = ["reasoning", "autonomy_reasoning", "action"]
        display_cols = [c for c in filtered.columns if c not in hidden_cols]

        table_df = filtered[display_cols].sort_values(
            by=["risk_score", "agent_name"], ascending=[False, True]
        )
        st.dataframe(table_df, use_container_width=True, height=600)

        # Export full filtered view
        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        json_bytes = table_df.to_json(orient="records", indent=2).encode("utf-8")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                "⬇️ Export filtered agents (CSV)",
                data=csv_bytes,
                file_name="agents_filtered.csv",
                mime="text/csv",
            )
        with col_exp2:
            st.download_button(
                "⬇️ Export filtered agents (JSON)",
                data=json_bytes,
                file_name="agents_filtered.json",
                mime="application/json",
            )

        st.markdown(
            """
**How to use this table**

- Filter in the sidebar by **risk, autonomy, or lifecycle** to create an actionable slice.  
- Export that slice to CSV/JSON for bulk reviews, Jira upload, or service-now style workflows.  
- Sort by owner or function to prepare review packets for specific stakeholders.
            """
        )

# -------------------------------------------------------
# PAGE: Agent Detail
# -------------------------------------------------------
elif page == "🔍 Agent Detail":
    st.title("🔍 Agent Detail View")
    st.caption("Deep dive on a single agent under the current filters.")

    if filtered.empty:
        st.info("No agents available for the selected filters.")
    else:
        # Optional additional filters
        col_af1, col_af2 = st.columns(2)
        with col_af1:
            owner_filter = st.selectbox(
                "Filter by Owner (optional)",
                options=["All"] + sorted(filtered["owner"].unique().tolist()),
            )
        with col_af2:
            risk_filter_detail = st.selectbox(
                "Filter by Risk (optional)",
                options=["All"] + sorted(filtered["risk_level"].unique().tolist()),
            )

        detail_df = filtered.copy()
        if owner_filter != "All":
            detail_df = detail_df[detail_df["owner"] == owner_filter]
        if risk_filter_detail != "All":
            detail_df = detail_df[detail_df["risk_level"] == risk_filter_detail]

        if detail_df.empty:
            st.info("No agents left after applying the detail filters.")
        else:
            agent_list = detail_df["agent_name"].unique().tolist()
            selected = st.selectbox("Choose an agent", agent_list)

            row = detail_df[detail_df["agent_name"] == selected].iloc[0]

            st.markdown(f"## {row['agent_name']}")
            st.markdown(
                f"""
**Profile**

- **Owner:** `{row.get('owner', '')}`  
- **Created By:** `{row.get('created_by', '')}`  
- **Risk Level:** `{row.get('risk_level', '')}`  
- **Autonomy Level:** `{row.get('autonomy_level', '')}`  
- **Lifecycle State:** `{row.get('lifecycle_state', '')}`  
- **Review Cadence:** `{row.get('review_cadence', '')}`  
- **Last Reviewed (synthetic):** `{row.get('last_reviewed', '')}`  
- **Next Review Due (synthetic):** `{row.get('next_review_due', '')}`  
                """
            )

            st.markdown("### Governance Notes")
            st.write(row.get("reasoning", ""))

            st.markdown("### Recommended Action")
            st.write(row.get("action", ""))

            st.markdown(
                """
**Executive summary for this agent**

- Check whether the **risk label** matches the real-world impact (financial, regulatory, customer-facing).  
- Confirm that the **autonomy level** is appropriate for that risk.  
- Ensure that the **owner and review cadence** are explicit and aligned with your internal policies.
                """
            )

# (Insights page continues in Part 3)
# -------------------------------------------------------
# PAGE: Insights (Architecture, Governance Model, Enterprise Integration)
# -------------------------------------------------------
elif page == "💡 Insights":
    st.title("💡 Governance Insights & Architecture")
    st.caption("Executive-level narrative, architecture diagrams, and integration blueprint.")

    # ---------------------------------------------------
    # Executive Talking Points
    # ---------------------------------------------------
    st.subheader("📌 Executive Talking Points")
    st.markdown(
        """
These are the **core principles** your governance portal communicates to leaders:

- **Single governance layer** — A unified control tower for all AI agents across the enterprise.  
- **Risk × Autonomy × Lifecycle** — Every agent is tracked across the three dimensions that matter for audits.  
- **Synthetic cadence engine** — Today it is sample data; in real-world deployments it connects to HR, ITSM, GRC, and DevOps.  
- **Enterprise-ready** — Designed to integrate with IAM, Data Lake, Ticketing, and Compliance systems.
        """
    )

    # ---------------------------------------------------
    # Collapsible Section: Governance Maturity Model
    # ---------------------------------------------------
    with st.expander("🧭 Governance Maturity Model (Simple 3-Level Framework)", expanded=False):
        st.markdown(
            """
**Level 1 — Manual & Reactive**  
- Risk ratings tracked inconsistently  
- No unified catalog of agents  
- Reviews triggered by incidents or audits  

**Level 2 — Standardized & Measurable**  
- Standard risk labels  
- Review cadences set  
- Owners and autonomy levels defined  
- Dashboards available  

**Level 3 — Automated & Continuous (Target State)**  
- Auto-populated lifecycle events  
- Continuous monitoring  
- GRC & IAM integration  
- Data-driven review frequency  
- Decommission workflows automated  
            """
        )

    st.markdown("---")

    # ---------------------------------------------------
    # High-Level Architecture
    # ---------------------------------------------------
    st.subheader("🏗️ High-Level Architecture")

    st.markdown(
        """
Below is your enterprise-grade architecture diagram rendered in **Mermaid** format.  
Copy/paste into:  
- https://mermaid.live  
- Notion code block  
- Obsidian  
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
        Reviews[Review Scheduler<br/>(cadence, next-review calc)]
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

    # SAFE DISPLAY (avoids triple-quote termination)
    st.code(mermaid_snippet, language="mermaid")

    st.markdown(
        """
**Interpretation for Executives**

- Data flows from **source systems** → into the portal → out toward **Security, HR, IT Ops, and GRC**.  
- Your portal becomes a **single pane of glass** for agent risk posture.  
- GRC and IAM integration are the long-term levers for continuous compliance and monitoring.
        """
    )

    st.markdown("---")

    # ---------------------------------------------------
    # Suggested Enterprise Integrations
    # ---------------------------------------------------
    st.subheader("🔌 Integration Roadmap")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown(
            """
### Phase 1 — Foundations  
- Automate ingestion of agent metadata  
- Assign owners, risk labels, autonomy modes  
- Establish review cadences  
- Enable export to CSV/JSON for audit packets  
            """
        )

    with col_i2:
        st.markdown(
            """
### Phase 2 — Deep System Hooks  
- **IAM Integration:** enforce allowed actions per agent  
- **ITSM Integration:** open review/approval tickets automatically  
- **Data Lake Integration:** centralize audit logs  
- **GRC Integration:** push risk posture downstream  
            """
        )

    st.markdown("---")

    # ---------------------------------------------------
    # Compliance & Audit Notes
    # ---------------------------------------------------
    with st.expander("🛡️ Compliance & Audit Guidance"):
        st.markdown(
            """
Your governance model aligns with common enterprise frameworks:

- **NIST AI RMF:** Maps to governance, oversight, and continuous monitoring  
- **ISO 42001:** Aligns with AI management system requirements  
- **EU AI Act (2025+):** Supports risk classification, logs, human oversight  

This portal gives auditors:  
- Clear ownership  
- Review history  
- Lifecycle trail  
- Risk & autonomy justification  
- Exportable evidence packets  
            """
        )

    st.markdown("---")

    # ---------------------------------------------------
    # Closing Insight
    # ---------------------------------------------------
    st.subheader("🎯 Final Takeaway")
    st.markdown(
        """
AI governance doesn’t need to be heavy to be effective.  
A lightweight portal with **risk, autonomy, lifecycle, ownership, cadence, and audits**  
already covers 80% of what enterprises need for responsible AI operations.

This dashboard is structured to be:

- Executive-friendly  
- Audit-ready  
- Integratable  
- Extensible  
- Future-proof  

**You now have a complete AI Governance Portal you can demo anywhere.**
        """
    )
