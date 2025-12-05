import json
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def load_agent_data(uploaded_file):
    """Load JSON into a pandas DataFrame."""
    if uploaded_file is None:
        st.warning("Upload a governance_decisions.json file to get started.")
        return pd.DataFrame()

    # Try JSON first, fall back to pandas read_json
    try:
        data = json.load(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        data = pd.read_json(uploaded_file)

    df = pd.DataFrame(data)

    # Normalise expected column names
    rename_map = {
        "agent": "agent_name",
        "name": "agent_name",
        "risk": "risk_level",
        "autonomy": "autonomy_level",
        "cadence": "review_cadence",
        "lifecycle": "lifecycle_state",
    }
    df = df.rename(columns=rename_map)
        # ------------------------------------------------------------
    # Normalize date columns so missing fields do not crash
    # ------------------------------------------------------------
    date_columns = [
        "requested_date",
        "approved_date",
        "testing_start",
        "deployment_date",
        "pilot_start",
        "last_reviewed",
        "next_review_due"
    ]

    for col in date_columns:
        if col not in df.columns:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], errors="coerce")


    required = [
        "agent_name",
        "owner",
        "created_by",
        "risk_level",
        "autonomy_level",
        "review_cadence",
        "lifecycle_state",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            f"The following required columns are missing from the JSON: {', '.join(missing)}"
        )
        return pd.DataFrame()

    # Ensure text columns are strings
    for col in required:
        df[col] = df[col].astype(str)

    # Add synthetic scheduling fields for the demo
    today = date.today()
    cadence_days = {
        "Immediate": 0,
        "Monthly": 30,
        "Quarterly": 90,
        "Semi-Annual": 180,
        "Annual": 365,
    }

    last_reviewed_list = []
    next_review_list = []
    days_to_next_list = []

    for idx, row in df.iterrows():
        # Spread last_reviewed dates so they aren't all identical
        offset_days = (idx + 1) * 7
        last_reviewed = today - timedelta(days=offset_days)

        # Normalise cadence spelling a bit and map to days
        gap = cadence_days.get(row["review_cadence"].title(), 90)

        next_review = last_reviewed + timedelta(days=gap)
        days_to_next = (next_review - today).days

        last_reviewed_list.append(last_reviewed)
        next_review_list.append(next_review)
        days_to_next_list.append(days_to_next)

    df["last_reviewed"] = last_reviewed_list
    df["next_review_due"] = next_review_list
    df["days_to_next"] = days_to_next_list
    df["agent_id"] = range(1, len(df) + 1)

    return df


def apply_sidebar_filters(df):
    """Apply risk/autonomy/lifecycle filters and return filtered DataFrame."""
    if df.empty:
        return df

    st.sidebar.markdown("### Filter by Risk Level")
    risk_options = ["All"] + sorted(df["risk_level"].unique())
    risk_choice = st.sidebar.selectbox("Risk Level", risk_options, index=0)

    st.sidebar.markdown("### Filter by Autonomy Level")
    auto_options = ["All"] + sorted(df["autonomy_level"].unique())
    auto_choice = st.sidebar.selectbox("Autonomy Level", auto_options, index=0)

    st.sidebar.markdown("### Filter by Lifecycle State")
    life_options = ["All"] + sorted(df["lifecycle_state"].unique())
    life_choice = st.sidebar.selectbox("Lifecycle State", life_options, index=0)

    filtered = df.copy()
    if risk_choice != "All":
        filtered = filtered[filtered["risk_level"] == risk_choice]
    if auto_choice != "All":
        filtered = filtered[filtered["autonomy_level"] == auto_choice]
    if life_choice != "All":
        filtered = filtered[filtered["lifecycle_state"] == life_choice]

    st.sidebar.markdown(f"**Loaded {len(filtered)} agents after filters**")
    return filtered


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------

def render_overview(df, df_filtered):
    st.title("AI Agent Governance Portal")
    st.caption(
        "Executive dashboard for AI agent risk, autonomy, lifecycle, and overall governance posture."
    )

    if df.empty:
        st.info("Upload data to see the portfolio overview.")
        return

    # ------------------ Top KPIs ------------------
    total_agents = len(df_filtered)
    high_risk = (df_filtered["risk_level"] == "HIGH RISK").sum()
    med_risk = (df_filtered["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (df_filtered["risk_level"] == "LOW RISK").sum()

    no_auto = (df_filtered["autonomy_level"] == "NO_AUTONOMY").sum()
    human_loop = (df_filtered["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited_auto = (df_filtered["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (df_filtered["autonomy_level"] == "AUTO_ALLOWED").sum()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Agents", total_agents)
    kpi2.metric("High Risk", high_risk)
    kpi3.metric("Medium Risk", med_risk)
    kpi4.metric("Low Risk", low_risk)
    kpi5.metric("Auto Allowed", auto_allowed)

    kpi6, kpi7, kpi8 = st.columns(3)
    kpi6.metric("No Autonomy", no_auto)
    kpi7.metric("Human-in-Loop", human_loop)
    kpi8.metric("Limited Autonomy", limited_auto)

    # ------------------ Overdue banner ------------------
    overdue = df_filtered[df_filtered["days_to_next"] < 0]
    if not overdue.empty:
        units = sorted(overdue["owner"].unique())
        st.error(
            f"{len(overdue)} agents are overdue for review across "
            f"{', '.join(units)} — this represents cross-functional governance risk."
        )

    # ------------------ Upcoming reviews ------------------
    st.subheader("Upcoming Reviews (next 30 days)")

    upcoming = df_filtered[df_filtered["days_to_next"] <= 30].copy()
    upcoming = upcoming.sort_values("days_to_next")

    if upcoming.empty:
        st.info("No reviews due in the next 30 days under the current filters.")
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
        st.dataframe(upcoming[display_cols], use_container_width=True)

        csv_bytes = upcoming.to_csv(index=False).encode("utf-8")
        json_bytes = upcoming.to_json(orient="records", indent=2).encode("utf-8")

        export_cols = st.columns(2)
        with export_cols[0]:
            st.download_button(
                label="Export upcoming reviews (CSV)",
                data=csv_bytes,
                file_name="upcoming_reviews.csv",
                mime="text/csv",
            )
        with export_cols[1]:
            st.download_button(
                label="Export upcoming reviews (JSON)",
                data=json_bytes,
                file_name="upcoming_reviews.json",
                mime="application/json",
            )

    st.markdown("---")

    # ------------------ Governance posture text ------------------
    st.subheader("How to read this section")
    st.markdown(
        "- Focus on agents with reviews due in the next 0–7 days for immediate attention.\n"
        "- 8–30 day horizon is your planning runway to batch reviews by owner or business unit.\n"
        "- Use these metrics to drive review SLAs, escalation rules, and dashboards for Security, HR, and IT.\n"
    )

    # ------------------ Heatmap + Pie ------------------
    st.subheader("Governance posture at a glance")
    col_heat, col_pie = st.columns(2)

    # Risk vs Autonomy heatmap
    with col_heat:
        risk_auto = (
            df_filtered.groupby(["risk_level", "autonomy_level"])
            .size()
            .reset_index(name="count")
        )
        if risk_auto.empty:
            st.info("No data available for heatmap under current filters.")
        else:
            risk_auto = pd.DataFrame(risk_auto)
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

    # Risk breakdown pie chart
    with col_pie:
        risk_counts = (
            df_filtered["risk_level"]
            .value_counts()
            .rename_axis("risk_level")
            .reset_index(name="count")
        )
        if risk_counts.empty:
            st.info("No data available for risk breakdown under current filters.")
        else:
            risk_counts = pd.DataFrame(risk_counts)
            fig_pie = px.pie(
                risk_counts,
                values="count",
                names="risk_level",
                hole=0.45,
                title="Portfolio risk mix",
            )
            fig_pie.update_traces(textinfo="label+percent")
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=40))
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        "**Executive takeaway:** focus on high-risk agents with high autonomy or auto-allowed status. "
        "Those combinations deserve clear ownership, playbooks, and tighter review cadences."
    )

    st.markdown("---")

    # ------------------ High-risk spotlight ------------------
    high_risk_agents = df_filtered[df_filtered["risk_level"] == "HIGH RISK"].copy()
    high_risk_agents = high_risk_agents.sort_values("days_to_next").head(4)

    if not high_risk_agents.empty:
        with st.expander("High-Risk Agent Spotlight", expanded=False):
            cols = st.columns(len(high_risk_agents))
            for col, (_, row) in zip(cols, high_risk_agents.iterrows()):
                with col:
                    st.markdown(f"### {row['agent_name']}")
                    st.markdown(f"**Owner:** {row['owner']}")
                    st.markdown(f"**Risk Level:** {row['risk_level']}")
                    st.markdown(f"**Autonomy:** {row['autonomy_level']}")
                    st.markdown(f"**Lifecycle:** {row['lifecycle_state']}")
                    st.markdown(f"**Next Review Due:** {row['next_review_due']}")
                    st.markdown(f"**Days to Next:** {row['days_to_next']}")
            st.markdown(
                "_These agents form your priority backlog for risk reduction across Security, HR, and IT._"
            )


def render_insights(df_filtered):
    st.title("Insights and Governance Lens")

    if df_filtered.empty:
        st.info("No agents available under the current filters.")
        return

    # -------- Insight 1 – Portfolio Risk Mix --------
    st.subheader("Insight 1 – Portfolio Risk Mix")
    risk_counts = (
        df_filtered["risk_level"]
        .value_counts()
        .rename_axis("risk_level")
        .reset_index(name="count")
    )
    if not risk_counts.empty:
        risk_counts = pd.DataFrame(risk_counts)
        fig = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            color="risk_level",
            title="Agents by risk level",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- High counts in the high-risk category indicate where governance time should be concentrated.\n"
        "- Low and medium risk agents are candidates for automation or relaxed review cadence.\n"
    )

    st.markdown("---")

    # -------- Insight 2 – Autonomy vs Risk --------
    st.subheader("Insight 2 – Autonomy vs Risk")
    risk_auto = (
        df_filtered.groupby(["risk_level", "autonomy_level"])
        .size()
        .reset_index(name="count")
    )
    if risk_auto.empty:
        st.info("No data available for autonomy vs risk under current filters.")
    else:
        risk_auto = pd.DataFrame(risk_auto)
        fig_heat = px.density_heatmap(
            risk_auto,
            x="autonomy_level",
            y="risk_level",
            z="count",
            color_continuous_scale="Blues",
            title="Where autonomy and risk intersect",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(
        "- Focus on cells where high risk intersects with auto-allowed or limited autonomy.\n"
        "- These patterns point to candidates for stricter guardrails or human-in-loop enforcement.\n"
    )

    st.markdown("---")

    # -------- Insight 3 – Lifecycle Health --------
    st.subheader("Insight 3 – Lifecycle Health")
    lifecycle_counts = (
        df_filtered["lifecycle_state"]
        .value_counts()
        .rename_axis("lifecycle_state")
        .reset_index(name="count")
    )
    if lifecycle_counts.empty:
        st.info("No lifecycle data available.")
    else:
        lifecycle_counts = pd.DataFrame(lifecycle_counts)
        fig_life = px.bar(
            lifecycle_counts,
            x="lifecycle_state",
            y="count",
            title="Agents by lifecycle state",
        )
        st.plotly_chart(fig_life, use_container_width=True)

    st.markdown(
        "- A heavy concentration in testing or pilot suggests experimentation without graduation criteria.\n"
        "- Retired or archived agents should have access fully removed and logs retained for audit.\n"
    )


def render_agents_table(df_filtered):
    st.title("Agents Table")

    if df_filtered.empty:
        st.info("No agents match the current filters.")
        return

    display_cols = [
        "agent_name",
        "owner",
        "created_by",
        "risk_level",
        "autonomy_level",
        "review_cadence",
        "lifecycle_state",
        "last_reviewed",
        "next_review_due",
        "days_to_next",
    ]
    st.dataframe(df_filtered[display_cols], use_container_width=True)

    csv_bytes = df_filtered[display_cols].to_csv(index=False).encode("utf-8")
    json_bytes = (
        df_filtered[display_cols].to_json(orient="records", indent=2).encode("utf-8")
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export filtered agents (CSV)",
            data=csv_bytes,
            file_name="agents_filtered.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Export filtered agents (JSON)",
            data=json_bytes,
            file_name="agents_filtered.json",
            mime="application/json",
        )


def render_agent_detail(df_filtered):
    st.title("Agent Detail")

    # Select an agent
    names = df_filtered["agent_name"].tolist()
    selected_name = st.selectbox("Select an agent", names)

    # Retrieve agent row
    agent = df_filtered[df_filtered["agent_name"] == selected_name].iloc[0]

    # Display basic info
    st.markdown(f"## {agent['agent_name']}")
    st.markdown(f"**Owner:** {agent['owner']}")
    st.markdown(f"**Created by:** {agent['created_by']}")
    st.markdown(f"**Risk Level:** {agent['risk_level']}")
    st.markdown(f"**Autonomy Level:** {agent['autonomy_level']}")
    st.markdown(f"**Review Cadence:** {agent['review_cadence']}")
    st.markdown(f"**Lifecycle State:** {agent['lifecycle_state']}")
    st.markdown(f"**Last Reviewed:** {agent['last_reviewed']}")
    st.markdown(f"**Next Review Due:** {agent['next_review_due']}")
    st.markdown(f"**Days to Next Review:** {agent['days_to_next']}")

    # Governance notes
    st.subheader("Governance Notes")
    notes = []

    if agent.get("risk_level") == "HIGH RISK":
        notes.append("High-risk agent — ensure data classification, logging, and rollback procedures.")

    if notes:
        for n in notes:
            st.markdown(f"- {n}")
    else:
        st.markdown("This agent appears within normal governance thresholds.")

  
        

     # ------------------------------------------------------------
    # Safe date handling for the selected agent (prevents NameError)
    # ------------------------------------------------------------
    date_fields = [
        "requested_date",
        "approved_date",
        "testing_start",
        "deployment_date",
        "pilot_start",
        "decommissioned_date",
        "last_reviewed",
        "next_review_due"
    ]

    for col in date_fields:
        if col not in agent.index:
            agent[col] = pd.NaT
        agent[col] = pd.to_datetime(agent[col], errors="coerce")

    # Pick safest base event date for future calculations
    base_for_90 = agent["requested_date"]
    if pd.isnull(base_for_90):
        base_for_90 = agent.get("deployment_date", pd.NaT)
    if pd.isnull(base_for_90):
        base_for_90 = agent.get("approved_date", pd.NaT)
    if pd.isnull(base_for_90):
        base_for_90 = agent.get("testing_start", pd.NaT)

    st.subheader("⭐ Executive Summary")

    # ----------------------------------------------------
# 90-day KPI metrics for the entire dataset
# ----------------------------------------------------
cutoff = pd.Timestamp.today() - pd.Timedelta(days=90)

safe = df_filtered.copy()
safe["created_date"] = pd.to_datetime(safe.get("created_date"), errors="coerce")
safe["deployment_date"] = pd.to_datetime(safe.get("deployment_date"), errors="coerce")
safe["decommissioned_date"] = pd.to_datetime(safe.get("decommissioned_date"), errors="coerce")

total_90 = safe[safe["created_date"] >= cutoff].shape[0]
deployed_90 = safe[safe["deployment_date"] >= cutoff].shape[0]
decomm_90 = safe[safe["decommissioned_date"] >= cutoff].shape[0]

# Display 3 KPI columns
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Created (last 90 days)", total_90)

with c2:
    st.metric("Deployed (last 90 days)", deployed_90)

with c3:
    st.metric("Decommissioned (last 90 days)", decomm_90)

st.divider()

    with c5:
        st.metric(
            "Average time to deploy",
            f"{avg_time_to_deploy} days" if avg_time_to_deploy is not None else "–",
        )
    with c6:
        st.metric("Agents in approval queue", in_queue)
    with c7:
        st.metric("Lifecycle completion rate", f"{lifecycle_completion_rate}%")

    st.divider()

    # ----------------------------------------------------------
    # How to read this section
    # ----------------------------------------------------------
    st.subheader("📘 How to read this section")
    st.markdown(
        """
- The timeline shows each agent’s journey from request to deployment using dated milestones.
- Focus on agents stuck in **Testing** for 30–45+ days — these indicate approval bottlenecks.
- Lifecycle events double as evidence for audit readiness and governance maturity.
- Use velocity trends to size SLAs and approval staffing.
- Decommissioned/retired agents demonstrate clean lifecycle closure.
        """
    )


    # ------------------------------------------------------------------
    # Bottleneck analysis – agents stuck in testing
    # ------------------------------------------------------------------
    stuck_threshold_days = 45
    stuck_mask = testing_mask & df["testing_start"].notna() & (
        (today - df["testing_start"]).dt.days > stuck_threshold_days
    )
    stuck_df = df.loc[stuck_mask].copy()
    stuck_count = len(stuck_df)
    stuck_names = list(stuck_df["agent_name"].dropna().unique())[:3]
    stuck_list_str = ", ".join(stuck_names) if stuck_names else "—"

    if stuck_count > 0:
        # Strategic banner styled like Overview risk banner
        st.markdown(
            f"""
<div style="background-color:#ffecec;border-left:4px solid #e00000;padding:0.9rem 1.1rem;margin-top:1rem;margin-bottom:1rem;">
<strong>⚠️ Security / Approval Bottleneck:</strong> {stuck_count} agent(s) have been in <code>Testing</code> for more than {stuck_threshold_days} days — e.g. <strong>{stuck_list_str}</strong>.  
This indicates slow approvals and longer deployment cycles for high-risk agents.
</div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Agent Lifecycle Progression Timeline (scatter-style Gantt)
    # ------------------------------------------------------------------
    st.subheader("📈 Agent Lifecycle Progression Timeline")
    st.caption("Lifecycle Timeline (Requested → Approved → Testing → Deployed)")

    timeline_rows = []
    for _, row in df.iterrows():
        agent = row.get("agent_name", "")

        steps = [
            ("Requested", row.get("requested_date")),
            ("Approved", row.get("approved_date")),
            ("Testing", row.get("testing_start")),
            ("Deployed", row.get("deployment_date")),
        ]

        for state, dt in steps:
            if pd.notnull(dt):
                timeline_rows.append(
                    {
                        "Agent": agent,
                        "State": state,
                        "Date": dt,
                    }
                )

    timeline_df = pd.DataFrame(timeline_rows)

    if not timeline_df.empty:
        state_order = ["Requested", "Approved", "Testing", "Deployed"]

        fig = px.scatter(
            timeline_df,
            x="Date",
            y="Agent",
            color="State",
            symbol="State",
            category_orders={"State": state_order},
            hover_data=["State", "Date"],
        )
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Agent",
            legend_title="Lifecycle state",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No lifecycle events with dates are available to plot.")

    # ------------------------------------------------------------------
    # Lifecycle health indicators – cards
    # ------------------------------------------------------------------
    st.subheader("🩺 Lifecycle health indicators")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown("**🔴 Approval bottleneck**")
        st.write(f"{stuck_count} agent(s) in Testing for more than {stuck_threshold_days} days.")
        if stuck_names:
            st.write("Examples: " + ", ".join(stuck_names))

    with h2:
        st.markdown("**⚠️ Deployment velocity**")
        deployed_df = df.loc[deployed_mask & df["deployment_date"].notna()].copy()
        if not deployed_df.empty:
            deployed_df["deploy_quarter"] = deployed_df["deployment_date"].dt.to_period("Q")
            q_counts = (
                deployed_df.groupby("deploy_quarter")["agent_name"]
                .count()
                .sort_index()
            )
            last_q = q_counts.tail(3)
            if not last_q.empty:
                lines = [f"- {str(idx)}: {val} deployed" for idx, val in last_q.items()]
                st.markdown("\n".join(lines))
            else:
                st.write("Insufficient deployment history.")
        else:
            st.write("No deployed agents with dates available.")

    with h3:
        st.markdown("**🟢 Decommissioning discipline**")
        st.write(f"{decomm_90} agent(s) decommissioned in the last 90 days.")
        if decomm_90 == 0:
            st.write("No recent decommissions – review whether legacy agents need formal retirement.")

    # ------------------------------------------------------------------
    # Export lifecycle events
    # ------------------------------------------------------------------
    st.subheader("📤 Export lifecycle events")

    if not timeline_df.empty:
        export_cols = ["Agent", "State", "Date"]
        export_df = timeline_df[export_cols].sort_values(["Agent", "Date"])

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        json_bytes = export_df.to_json(orient="records", indent=2).encode("utf-8")

        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "Download lifecycle events (CSV)",
                csv_bytes,
                file_name="lifecycle_events.csv",
                mime="text/csv",
            )
        with e2:
            st.download_button(
                "Download lifecycle events (JSON)",
                json_bytes,
                file_name="lifecycle_events.json",
                mime="application/json",
            )
    else:
        st.info("No dated lifecycle events to export yet.")

    # ------------------------------------------------------------------
    # Executive takeaway – director-level summary
    # ------------------------------------------------------------------
    st.subheader("🧭 Executive takeaway")

    strengths_lines = []
    if deployed_90:
        strengths_lines.append(
            f"- Deployment velocity is healthy with **{deployed_90} agent(s)** deployed in the last 90 days."
        )
    if decomm_90:
        strengths_lines.append(
            f"- Governance discipline is visible with **{decomm_90} formal decommission(s)** in the last 90 days."
        )
    if lifecycle_completion_rate:
        strengths_lines.append(
            f"- **{lifecycle_completion_rate}%** lifecycle completion rate shows that most agents make it through to deployment or clean retirement."
        )

    risks_lines = []
    if stuck_count:
        risks_lines.append(
            f"- **{stuck_count} agent(s)** stuck in Testing beyond {stuck_threshold_days} days indicate an approval bottleneck and extended risk window."
        )
    if in_queue:
        risks_lines.append(
            f"- **{in_queue} agent(s)** sitting in the approval queue suggest that review capacity may become a constraint as you scale."
        )

    if not strengths_lines:
        strengths_lines.append("- Baseline lifecycle data is captured, enabling future trend analysis.")
    if not risks_lines:
        risks_lines.append("- No major lifecycle bottlenecks detected with current filters.")

    st.markdown(
        f"""
**Strengths**

{chr(10).join(strengths_lines)}

**Risks / Opportunities**

{chr(10).join(risks_lines)}

**Recommended next steps**

1. Set explicit SLAs for Testing and approval stages (for example, 7–14 days depending on risk level).
2. Monitor agents breaching those SLAs using this timeline view and escalate to owning teams.
3. As volume increases, automate approvals for low-risk / low-autonomy agents while keeping stronger controls on high-risk ones.
4. Continue to enforce clean decommissioning – retired, deprecated, or archived agents should always have a clear end date and audit trail.
        """
    )

def render_policy_simulator(df_filtered):
    st.title("Policy Simulator")

    if df_filtered.empty:
        st.info("Load data to experiment with governance policy scenarios.")
        return

    st.markdown(
        "This simple simulator lets you test how tightening review cadences could change the backlog of reviews."
    )

    risk_choice = st.selectbox(
        "Select a risk level to adjust",
        sorted(df_filtered["risk_level"].unique()),
    )
    new_cadence = st.selectbox(
        "New review cadence",
        ["Immediate", "Monthly", "Quarterly", "Semi-Annual", "Annual"],
        index=1,
    )

    subset = df_filtered[df_filtered["risk_level"] == risk_choice].copy()
    if subset.empty:
        st.info("No agents match this risk level.")
        return

    today = date.today()
    cadence_days = {
        "Immediate": 0,
        "Monthly": 30,
        "Quarterly": 90,
        "Semi-Annual": 180,
        "Annual": 365,
    }
    gap = cadence_days[new_cadence]

    simulated_next = subset["last_reviewed"].apply(lambda d: d + timedelta(days=gap))
    simulated_days_to_next = simulated_next.apply(lambda d: (d - today).days)

    col1, col2 = st.columns(2)
    col1.metric(
        "Current average days to next review",
        f"{subset['days_to_next'].mean():.1f}",
    )
    col2.metric(
        "Simulated average days to next review",
        f"{simulated_days_to_next.mean():.1f}",
    )

    st.markdown(
        "Use this as talking points with risk owners: how would moving all "
        f"{risk_choice.lower()} agents to {new_cadence.lower()} reviews shift the workload?"
    )


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Agent Governance Portal",
        layout="wide",
        page_icon="🛡️",
    )

    st.sidebar.title("Controls")
    uploaded = st.sidebar.file_uploader(
        "Upload governance_decisions.json", type=["json"]
    )

    df = load_agent_data(uploaded)
    if df.empty:
        return

    st.sidebar.success(f"Loaded {len(df)} agents")

    df_filtered = apply_sidebar_filters(df)

    st.sidebar.markdown("### Navigate")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Insights",
            "Agents Table",
            "Agent Detail",
            "Lifecycle Timeline",
            "Policy Simulator",
        ],
    )

    if page == "Overview":
        render_overview(df, df_filtered)
    elif page == "Insights":
        render_insights(df_filtered)
    elif page == "Agents Table":
        render_agents_table(df_filtered)
  

    elif page == "Agent Detail":
        render_agent_detail(df_filtered)
    elif page == "Lifecycle Timeline":
        render_lifecycle_timeline(df_filtered)
    elif page == "Policy Simulator":
        render_policy_simulator(df_filtered)


    

# ⬇️ PASTE FUNCTION BELOW THIS LINE
def render_lifecycle_timeline(df_filtered):
    import pandas as pd
    import plotly.express as px

    st.title("📊 Lifecycle Timeline – Deployment & Governance Insights")

    if df_filtered.empty:
        st.info("No agents available under the current filters.")
        return

    df = df_filtered.copy()

    date_cols = ["requested_date", "approved_date", "testing_start", "deployment_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT

    timeline_rows = []
    for _, row in df.iterrows():
        steps = [
            ("Requested", row["requested_date"]),
            ("Approved", row["approved_date"]),
            ("Testing", row["testing_start"]),
            ("Deployed", row["deployment_date"]),
        ]
        for state, dt in steps:
            if pd.notnull(dt):
                timeline_rows.append({
                    "Agent": row["agent_name"],
                    "State": state,
                    "Date": dt,
                })

    timeline_df = pd.DataFrame(timeline_rows)

    if timeline_df.empty:
        st.info("No lifecycle events available.")
        return

    state_order = ["Requested", "Approved", "Testing", "Deployed"]

    fig = px.scatter(
        timeline_df,
        x="Date",
        y="Agent",
        color="State",
        symbol="State",
        category_orders={"State": state_order},
        title="Lifecycle Events Timeline"
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
