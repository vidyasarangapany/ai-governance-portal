import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI Agent Governance Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# HELPER: calculate next review date
# -----------------------------------------------------------
def compute_next_review(last_reviewed, cadence):
    if pd.isna(last_reviewed):
        return None

    if cadence == "Immediate":
        return last_reviewed + timedelta(days=1)
    if cadence == "Monthly":
        return last_reviewed + timedelta(days=30)
    if cadence == "Quarterly":
        return last_reviewed + timedelta(days=90)
    if cadence == "Semi-Annual":
        return last_reviewed + timedelta(days=180)
    if cadence == "Annual":
        return last_reviewed + timedelta(days=365)
    return None


# -----------------------------------------------------------
# SIDEBAR — FILE UPLOAD
# -----------------------------------------------------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json",
    type=["json"]
)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Insights", "📋 Agents Table",
     "🔍 Agent Detail", "⏳ Lifecycle Timeline", "🛡 Policy Simulator"]
)


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
if uploaded_file:
    df = pd.read_json(uploaded_file)

    st.sidebar.success(f"Loaded {len(df)} agents")

    # Normalize last/next review fields for governance engine
    df["last_reviewed"] = pd.to_datetime(
        df.get("last_reviewed", datetime.now() - timedelta(days=45))
    )

    df["next_review_due"] = df.apply(
        lambda r: compute_next_review(r["last_reviewed"], r["review_cadence"]),
        axis=1
    )

    df["days_to_next"] = (df["next_review_due"] - datetime.now()).dt.days

else:
    st.title("AI Agent Governance Portal")
    st.info("Upload governance_decisions.json to begin.")
    st.stop()


# -----------------------------------------------------------
# FILTERS
# -----------------------------------------------------------
risk_filter = st.sidebar.selectbox(
    "Filter by Risk Level", ["All"] + sorted(df["risk_level"].unique())
)

aut_filter = st.sidebar.selectbox(
    "Filter by Autonomy Level", ["All"] + sorted(df["autonomy_level"].unique())
)

life_filter = st.sidebar.selectbox(
    "Filter by Lifecycle State", ["All"] + sorted(df["lifecycle_state"].unique())
)

# Apply filters
df_filtered = df.copy()
if risk_filter != "All":
    df_filtered = df_filtered[df_filtered["risk_level"] == risk_filter]
if aut_filter != "All":
    df_filtered = df_filtered[df_filtered["autonomy_level"] == aut_filter]
if life_filter != "All":
    df_filtered = df_filtered[df_filtered["lifecycle_state"] == life_filter]


# -----------------------------------------------------------
# PAGE: OVERVIEW
# -----------------------------------------------------------
if page == "🏠 Overview":

    st.title("🛡 AI Agent Governance Portal")
    st.caption("Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture.")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Agents", len(df))
    col2.metric("High Risk", sum(df["risk_level"] == "HIGH RISK"))
    col3.metric("Medium Risk", sum(df["risk_level"] == "MEDIUM RISK"))
    col4.metric("Low Risk", sum(df["risk_level"] == "LOW RISK"))
    col5.metric("No Autonomy", sum(df["autonomy_level"] == "NO_AUTONOMY"))
    col6.metric("Auto Allowed", sum(df["autonomy_level"] == "AUTO_ALLOWED"))

    st.markdown("---")

    # -----------------------------------------------------------
    # OVERDUE REVIEW ALERT (DIRECTOR-LEVEL SIGNAL)
    # -----------------------------------------------------------
    overdue = df[df["days_to_next"] < 0]

    if len(overdue) > 0:
        owners = ", ".join(sorted(overdue["owner"].unique()))
        st.error(
            f"⚠️ **{len(overdue)} agents are overdue for review** across **{owners}** — "
            "this represents cross-functional governance risk."
        )

    st.markdown("### 🔔 Upcoming Reviews (next 30 days)")

    upcoming = df[df["days_to_next"].between(0, 30)]
    st.dataframe(
        upcoming[["agent_name", "owner", "risk_level", "review_cadence",
                   "last_reviewed", "next_review_due", "days_to_next"]],
        use_container_width=True
    )

    st.download_button("📥 Export upcoming reviews (CSV)",
                       upcoming.to_csv(index=False),
                       "upcoming_reviews.csv")

    st.download_button("📥 Export upcoming reviews (JSON)",
                       upcoming.to_json(orient="records"),
                       "upcoming_reviews.json")

    st.markdown("## Governance posture at a glance")

    st.markdown("""
    - **0–7 days horizon:** Agents needing review *this week*
    - **8–30 days horizon:** Planning runway to batch by owner or business unit
    - Use these to drive **review SLAs**, automated escalations, and dashboards for **Security, HR, and IT**.
    """)

    # -----------------------------------------------------------
    # HEATMAP & PIE CHART SECTION
    # -----------------------------------------------------------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Risk vs Autonomy Heatmap")

        heat = df.groupby(["risk_level", "autonomy_level"]).size().reset_index(name="count")

        fig_heat = px.density_heatmap(
            heat,
            x="autonomy_level",
            y="risk_level",
            z="count",
            color_continuous_scale="Blues",
            title="Where autonomy and risk intersect"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with colB:
        st.subheader("Risk Breakdown")

   # -----------------------------------------------
# SAFE PIE CHART (Prevents Plotly DuplicateError)
# -----------------------------------------------

risk_counts = (
    df["risk_level"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "risk_level", "risk_level": "count"})
)

fig_pie = px.pie(
    data_frame=risk_counts,
    names="risk_level",
    values="count",
    hole=0.45,
)

fig_pie.update_traces(
    textinfo="label+percent",
    pull=[0.05] * len(risk_counts)
)

fig_pie.update_layout(
    title="Risk Level Breakdown",
    legend_title="Risk Level"
)

st.plotly_chart(fig_pie, use_container_width=True)


# -----------------------------------------------------------
# PAGE: INSIGHTS
# -----------------------------------------------------------
elif page == "📊 Insights":

    st.title("💡 Insights & Governance Lens")

    # Recompute overdue & upcoming windows for insights
    overdue = df[df["days_to_next"] < 0]
    window_0_7 = df[df["days_to_next"].between(0, 7)]
    window_8_30 = df[df["days_to_next"].between(8, 30)]

    # ===============================
    # Insight 1 – Portfolio Risk Mix
    # ===============================
    st.subheader("① Portfolio Risk Mix")

    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "risk_level", "risk_level": "count"})
    )

    fig_risk_bar = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        color="risk_level",
        title="Risk distribution across all agents",
        text="count"
    )
    fig_risk_bar.update_traces(textposition="outside")
    fig_risk_bar.update_layout(yaxis_title="Number of agents")

    st.plotly_chart(fig_risk_bar, use_container_width=True)

    # Executive lens for risk mix
    total_agents = len(df)
    high_risk = risk_counts.loc[risk_counts["risk_level"] == "HIGH RISK", "count"].sum()
    med_risk = risk_counts.loc[risk_counts["risk_level"] == "MEDIUM RISK", "count"].sum()
    low_risk = risk_counts.loc[risk_counts["risk_level"] == "LOW RISK", "count"].sum()

    st.markdown(
        f"""
**Executive takeaway:**

- **High risk:** {high_risk} agents  
- **Medium risk:** {med_risk} agents  
- **Low risk:** {low_risk} agents  

High-risk agents should map to **clear owners, runbooks, and review SLAs**.  
Use this mix to justify **investment in controls, testing environments, and human-in-loop guardrails**.
"""
    )

    st.markdown("---")

    # =======================================
    # Insight 2 – Autonomy vs Risk Lens
    # =======================================
    st.subheader("② Autonomy vs Risk Lens")

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
            title="Where autonomy and risk intersect"
        )
        st.plotly_chart(fig_insight_heat, use_container_width=True)

    st.markdown(
        """
**Executive takeaway:**

- Focus first on blocks in the **top-right**: high-risk agents with **AUTO_ALLOWED** or **LIMITED_AUTONOMY**.  
- Those agents should have: **strict access scopes, human approval workflows, and aggressive review cadences**.  
- Low-risk & low-autonomy agents are ideal candidates for **self-service onboarding** or relaxed SLAs.
"""
    )

    st.markdown("---")

    # =======================================
    # Insight 3 – Cross-Functional Governance Gaps
    # =======================================
    st.subheader("③ Cross-Functional Governance Gaps")

    if len(overdue) == 0:
        st.success("✅ No agents are overdue for review. Governance SLAs are currently being met.")
    else:
        owners_overdue = overdue.groupby("owner").size().reset_index(name="overdue_count")
        owners_overdue = owners_overdue.sort_values("overdue_count", ascending=False)

        st.markdown(
            f"""
**Strategic signal:**  
There are **{len(overdue)} agents overdue for review**, spanning
**{owners_overdue['owner'].nunique()} functions**.

This is a **cross-functional governance gap** – not just an isolated team issue.
"""
        )

        st.dataframe(owners_overdue, use_container_width=True)

        st.markdown(
            """
**Director-level implication:**

- Define **review SLAs by risk level** (e.g., high-risk = 30 days, medium = 90, low = 180).  
- Implement **automated escalation** when high-risk reviews are missed (e.g., notify Security + HR + IT).  
- Use this table as the backbone for a **governance OKR**:  
  “100% of high-risk agents reviewed on time for Q1.”
"""
        )

    st.markdown("---")

    # =======================================
    # Insight 4 – Three-Portal Architecture View
    # =======================================
    st.subheader("④ How this Governance Layer Orchestrates the 3 Portals")

    col_arch, col_text = st.columns([1, 1.4])

    with col_text:
        st.markdown(
            """
Think of this portal as the **control tower**:

- **Security / Compliance Portal** → applies access policies, RBAC, logging  
- **HR / Business Portal** → manages human approvals, role changes, and exceptions  
- **IT / Admin Portal** → provisions infrastructure, credentials, and runtime configs  

This governance app decides **who can do what**, at **what risk level**, and **when a review is mandatory**.
"""
        )

    with col_arch:
        mermaid_diagram = """
graph LR
    A[AI Agent Governance Portal] --> B[Security / Compliance Portal]
    A --> C[HR & Business Portal]
    A --> D[IT Admin / Infra Portal]

    B --> E[Access Policies / RBAC]
    C --> F[Approvals & Exceptions]
    D --> G[Provisioning & Runtime Config]

    A --> H[Audit Log & Evidence Store]
"""
        # Shown as code so you can paste into mermaid.live, Notion, etc.
        st.code(mermaid_diagram, language="mermaid")

    st.markdown(
        """
Use this architecture in **presentations and portfolios** to show how your governance layer  
ties together **policy**, **people**, and **platforms**.
"""
    )


# -----------------------------------------------------------
# PAGE: AGENTS TABLE
# -----------------------------------------------------------
elif page == "📋 Agents Table":

    st.title("📋 Agents Table")

    st.caption(
        "Filters from the left sidebar apply here so you can slice by risk, autonomy, or lifecycle."
    )

    st.dataframe(
        df_filtered[
            [
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
        ],
        use_container_width=True,
    )

    st.download_button(
        "📥 Export filtered agents (CSV)",
        df_filtered.to_csv(index=False),
        "agents_filtered.csv",
    )

    st.download_button(
        "📥 Export all agents (CSV)",
        df.to_csv(index=False),
        "agents_all.csv",
    )
# -----------------------------------------------------------
# PAGE: AGENT DETAIL
# -----------------------------------------------------------
elif page == "🔎 Agent Detail":

    st.title("🔎 Agent Detail View")

    agent_names = df["agent_name"].unique().tolist()
    selected_agent = st.selectbox("Select an agent", agent_names)

    agent_df = df[df["agent_name"] == selected_agent].iloc[0]

    st.header(f"🧩 {selected_agent}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Owner:** `{agent_df['owner']}`")
        st.markdown(f"**Created By:** `{agent_df['created_by']}`")
        st.markdown(f"**Risk Level:** `{agent_df['risk_level']}`")
        st.markdown(f"**Autonomy Level:** `{agent_df['autonomy_level']}`")

    with col2:
        st.markdown(f"**Lifecycle State:** `{agent_df['lifecycle_state']}`")
        st.markdown(f"**Review Cadence:** `{agent_df['review_cadence']}`")
        st.markdown(f"**Last Reviewed:** `{agent_df['last_reviewed']}`")
        st.markdown(f"**Next Review Due:** `{agent_df['next_review_due']}`")

    # ------------------
    # AGENT STATUS BADGE
    # ------------------
    overdue_days = agent_df["days_to_next"]

    st.markdown("---")
    if overdue_days < 0:
        st.error(f"⚠️ **This agent is overdue by {-overdue_days} days**")
    elif overdue_days <= 7:
        st.warning(f"⏳ **Review due in {overdue_days} days** — high attention window")
    else:
        st.success(f"✅ Next review in {overdue_days} days")

    # -----------------------------------
    # DIRECTOR-LEVEL ANALYTICS
    # -----------------------------------
    st.markdown("### 🧠 Governance Intelligence")

    score = 0
    if agent_df["risk_level"] == "HIGH RISK":
        score += 4
    if agent_df["autonomy_level"] in ["AUTO_ALLOWED", "LIMITED_AUTONOMY"]:
        score += 3
    if overdue_days < 0:
        score += 5
    if agent_df["lifecycle_state"] in ["DEPLOYED"]:
        score += 2

    st.metric("Governance Risk Score", f"{score} / 10")

    st.markdown(
        """
**Interpretation:**

- 0–3 → Low governance priority  
- 4–6 → Medium priority, check autonomy + cadence  
- 7–10 → 🚨 **High priority** — requires Security + HR review  
"""
    )

    # -----------------------------------
    # MINI ACTION PLAN
    # -----------------------------------
    st.markdown("### 📌 Recommended Action Plan")

    if score >= 7:
        st.markdown(
            """
🔥 **High Priority Actions:**
- Require immediate review by Security + HR  
- Validate runbooks + monitoring  
- Reduce autonomy level until review is completed  
- Enable weekly health checks  
"""
        )
    elif score >= 4:
        st.markdown(
            """
⚠️ **Medium Priority Actions:**
- Check access scopes  
- Ensure owner acknowledgment  
- Validate logging & evidence  
"""
        )
    else:
        st.markdown(
            """
✅ **Low Priority Actions:**
- Keep current cadence  
- No escalations needed  
"""
        )


    st.markdown("---")

    # ----------------------------
    # EXPORT SINGLE AGENT AS JSON
    # ----------------------------
    st.download_button(
        "📥 Export this agent as JSON",
        agent_df.to_json(indent=2),
        file_name=f"{selected_agent}.json",
    )


# -----------------------------------------------------------
# PAGE: LIFECYCLE TIMELINE
# -----------------------------------------------------------
elif page == "📅 Lifecycle Timeline":

    st.title("📅 Agent Lifecycle Timeline")

    st.caption("Shows an enterprise-style lifecycle model for AI agents.")

    # Create synthetic timeline events
    timeline = []

    for _, row in df.iterrows():
        timeline.append(
            dict(
                Task=row["agent_name"],
                Start=row["last_reviewed"],
                Finish=row["next_review_due"],
                Resource=row["risk_level"],
            )
        )

    if len(timeline) > 0:
        fig_timeline = px.timeline(
            timeline,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Resource",
            title="Agent Review Timeline",
            color_discrete_map={
                "HIGH RISK": "#ff4d4d",
                "MEDIUM RISK": "#ffae42",
                "LOW RISK": "#5cb85c",
            },
        )

        fig_timeline.update_yaxes(autorange="reversed")
        fig_timeline.update_layout(height=700)

        st.plotly_chart(fig_timeline, use_container_width=True)

    # Executive text
    st.markdown(
        """
### 🧠 What this timeline tells you

- Agents near the **left side** are overdue or about to breach SLA  
- Agents far on the **right** are healthy  
- Compare multiple teams to detect SLA compliance patterns  
"""
    )

    st.download_button(
        "📥 Export timeline data",
        pd.DataFrame(timeline).to_csv(index=False),
        "timeline_export.csv",
    )


# -----------------------------------------------------------
# Footer Branding
# -----------------------------------------------------------
st.markdown("---")
st.markdown(
    """
**AI Agent Governance Portal**  
Built with ❤️ for enterprise-grade Responsible AI oversight.  
"""
)
