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

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def load_json(uploaded_file):
    if uploaded_file is None:
        return None
    return json.load(uploaded_file)

def compute_next_review(last_date_str, cadence):
    if not last_date_str:
        return None
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
    except:
        return None

    if cadence.lower() == "monthly":
        next_date = last_date + timedelta(days=30)
    elif cadence.lower() == "quarterly":
        next_date = last_date + timedelta(days=90)
    elif cadence.lower() == "immediate":
        next_date = last_date + timedelta(days=7)
    else:
        next_date = last_date + timedelta(days=60)

    return next_date.strftime("%Y-%m-%d")

# --------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------
st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload governance_decisions.json",
    type=["json"],
)

data = load_json(uploaded_file)
if data:
    st.sidebar.success(f"Loaded {len(data)} agents")
else:
    st.sidebar.info("Waiting for file...")
    st.stop()

# Convert JSON → DataFrame
df = pd.DataFrame(data)

# Filters
risk_filter = st.sidebar.selectbox("Filter by Risk Level", ["All"] + sorted(df["risk"].unique()))
aut_filter = st.sidebar.selectbox("Filter by Autonomy Level", ["All"] + sorted(df["autonomy"].unique()))
life_filter = st.sidebar.selectbox("Filter by Lifecycle State", ["All"] + sorted(df["lifecycle"].unique()))

# Apply filters
df_view = df.copy()
if risk_filter != "All":
    df_view = df_view[df_view["risk"] == risk_filter]
if aut_filter != "All":
    df_view = df_view[df_view["autonomy"] == aut_filter]
if life_filter != "All":
    df_view = df_view[df_view["lifecycle"] == life_filter]

# Navigation
st.sidebar.markdown("### Navigate")
page = st.sidebar.radio(
    "",
    [
        "Overview",
        "Lifecycle Timeline",
        "Agents Table",
        "Agent Detail",
        "Insights"
    ]
)

# --------------------------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------------------------
if page == "Overview":

    st.title("🛡️ AI Agent Governance Portal")
    st.caption("Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture.")

    # KPI Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Agents", len(df))
    col2.metric("High Risk", sum(df["risk"] == "HIGH RISK"))
    col3.metric("Medium Risk", sum(df["risk"] == "MEDIUM RISK"))
    col4.metric("Low Risk", sum(df["risk"] == "LOW RISK"))
    col5.metric("Human-in-Loop", sum(df["autonomy"] == "HUMAN_IN_LOOP"))
    col6.metric("Auto Allowed", sum(df["autonomy"] == "AUTO_ALLOWED"))

    st.markdown("---")

    # ---------------------------------------------------------------
    # Upcoming Reviews Table
    # ---------------------------------------------------------------
    st.subheader("🔔 Upcoming Reviews (next 30 days)")

    preview_df = df.copy()
    preview_df["next_review"] = preview_df.apply(
        lambda row: compute_next_review(row["last_reviewed"], row["review_cadence"]),
        axis=1
    )
    preview_df["days_to_next"] = preview_df["next_review"].apply(
        lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime.today()).days
        if x else None
    )

    upcoming_df = preview_df[
        (preview_df["days_to_next"].notnull()) &
        (preview_df["days_to_next"] <= 30)
    ].sort_values("days_to_next")

    st.dataframe(upcoming_df[
        ["agent", "owner", "risk", "review_cadence", "last_reviewed", "next_review", "days_to_next"]
    ], use_container_width=True)

    # Export buttons
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "📤 Export upcoming reviews (CSV)",
            upcoming_df.to_csv(index=False),
            "upcoming_reviews.csv"
        )
    with colB:
        st.download_button(
            "📤 Export upcoming reviews (JSON)",
            upcoming_df.to_json(orient="records", indent=2),
            "upcoming_reviews.json"
        )

    st.markdown("### How to read this section")
    st.markdown("""
    • Focus on reviews due in the next **0–7 days** for immediate attention.  
    • 8–30 days gives you a planning window for batching reviews by owner or LoB.  
    • No upcoming reviews usually means cycles were completed recently.
    """)

    st.markdown("---")

    # ---------------------------------------------------------------
    # Spotlight: High-Risk Agent Summaries (Collapsible)
    # ---------------------------------------------------------------
    high_risk = df[df["risk"] == "HIGH RISK"]

    with st.expander("🔥 High-Risk Agent Spotlight", expanded=False):

        for agent in high_risk.itertuples():
            st.markdown(f"## {agent.agent}")

            st.markdown(f"""
            - **Owner:** `{agent.owner}`
            - **Autonomy:** `{agent.autonomy}`
            - **Lifecycle:** `{agent.lifecycle}`
            - **Review Cadence:** `{agent.review_cadence}`
            - **Last Reviewed:** `{agent.last_reviewed}`
            """)

            next_review = compute_next_review(agent.last_reviewed, agent.review_cadence)
            st.markdown(f"- **Next Review (synthetic):** `{next_review}`")

            st.markdown("**Executive Lens**")
            st.markdown("""
            These agents typically:
            • Carry high financial / regulatory / privacy impact  
            • Operate with elevated autonomy  
            • Require tight ownership and predictable review cadence  
            ---  
            """)
    # ----------------------------------------------------------------
    # RISK vs AUTONOMY HEATMAP
    # ----------------------------------------------------------------
    st.subheader("Risk vs Autonomy Heatmap")

    pivot = df.pivot_table(
        values="agent",
        index="risk",
        columns="autonomy",
        aggfunc="count",
        fill_value=0
    )

    fig_heatmap = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        labels=dict(x="Autonomy Level", y="Risk Level", color="Agents")
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ----------------------------------------------------------------
    # RISK BREAKDOWN DONUT
    # ----------------------------------------------------------------
    st.subheader("Risk Breakdown")

    risk_counts = df["risk"].value_counts().reset_index()
    risk_counts.columns = ["risk", "count"]

    fig_pie = px.pie(
        risk_counts,
        names="risk",
        values="count",
        hole=0.45
    )
    fig_pie.update_traces(textinfo="label+percent")

    st.plotly_chart(fig_pie, use_container_width=True)

# ====================================================================
# LIFECYCLE TIMELINE PAGE
# ====================================================================
elif page == "Lifecycle Timeline":

    st.title("📅 Agent Lifecycle Timeline")

    # Sort by lifecycle buckets for clarity
    order = ["IDEATION", "PILOT", "TESTING", "DEPLOYED", "DECOMMISSIONED"]

    df_life = df.copy()
    df_life["lifecycle_order"] = df_life["lifecycle"].apply(
        lambda x: order.index(x) if x in order else len(order)
    )
    df_life = df_life.sort_values("lifecycle_order")

    st.dataframe(
        df_life[["agent", "owner", "risk", "autonomy", "lifecycle"]],
        use_container_width=True
    )

# ====================================================================
# AGENTS TABLE PAGE
# ====================================================================
elif page == "Agents Table":

    st.title("📊 All Agents Table")
    st.caption("Filters on the left sidebar apply here.")

    st.dataframe(
        df_view.sort_values("agent"),
        use_container_width=True
    )

    # Export
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Export filtered table (CSV)",
            df_view.to_csv(index=False),
            "agents_filtered.csv"
        )
    with col2:
        st.download_button(
            "📥 Export filtered table (JSON)",
            df_view.to_json(orient="records", indent=2),
            "agents_filtered.json"
        )

# ====================================================================
# AGENT DETAIL PAGE
# ====================================================================
elif page == "Agent Detail":

    st.title("🔍 Agent Detail")

    selected_agent = st.selectbox(
        "Choose an agent",
        sorted(df["agent"].unique())
    )

    row = df[df["agent"] == selected_agent].iloc[0]

    st.markdown(f"## {row['agent']}")

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"**Owner:** `{row['owner']}`")
        st.markdown(f"**Risk:** `{row['risk']}`")
        st.markdown(f"**Autonomy:** `{row['autonomy']}`")
    with colB:
        st.markdown(f"**Lifecycle:** `{row['lifecycle']}`")
        st.markdown(f"**Review Cadence:** `{row['review_cadence']}`")
        st.markdown(f"**Last Reviewed:** `{row['last_reviewed']}`")

    next_review = compute_next_review(row["last_reviewed"], row["review_cadence"])
    st.info(f"📅 **Next Review (synthetic):** {next_review}")

    st.markdown("---")

# ====================================================================
# INSIGHTS PAGE
# ====================================================================
elif page == "Insights":

    st.title("💡 Insights & Governance Lens")

    # =======================
    # Insight 1 — Risk Mix
    # =======================
    st.subheader("1️⃣ Portfolio Risk Mix")
    st.markdown("""
    • High risk agents require focused control and predictable review cycles.  
    • Medium risk agents often drive operational efficiency but carry moderate dependencies.  
    • Low risk agents typically represent automation of low-impact workflows.  
    """)

    risk_counts = df["risk"].value_counts().reset_index()
    risk_counts.columns = ["risk", "count"]
    fig_insight_risk = px.bar(
        risk_counts,
        x="risk",
        y="count",
        color="risk",
        title="Risk Distribution"
    )
    st.plotly_chart(fig_insight_risk, use_container_width=True)

    # =======================
    # Insight 2 — Autonomy Mix
    # =======================
    st.subheader("2️⃣ Autonomy Mode Distribution")
    aut_counts = df["autonomy"].value_counts().reset_index()
    aut_counts.columns = ["autonomy", "count"]
    fig_aut = px.bar(
        aut_counts,
        x="autonomy",
        y="count",
        color="autonomy",
        title="Autonomy Levels"
    )
    st.plotly_chart(fig_aut, use_container_width=True)

    # =======================
    # Insight 3 — Lifecycle Health
    # =======================
    st.subheader("3️⃣ Lifecycle Health")
    life_counts = df["lifecycle"].value_counts().reset_index()
    life_counts.columns = ["lifecycle", "count"]

    fig_life = px.bar(
        life_counts,
        x="lifecycle",
        y="count",
        color="lifecycle",
        title="Agents by Lifecycle Stage"
    )
    st.plotly_chart(fig_life, use_container_width=True)

    st.markdown("---")
    st.markdown("### Executive Summary")
    st.markdown("""
    • **Deployed** agents should receive highest governance coverage.  
    • **Testing/Pilot** agents need stronger change control.  
    • **Ideation** should be monitored for proliferation risk.  
    • **Decommissioned** should retain audit artifacts for traceability.  
    """)
# ====================================================================
# MERMAID ARCHITECTURE SECTION (SAFE VERSION)
# ====================================================================
elif page == "Insights":  # continue inside Insights page

    st.markdown("---")
    st.subheader("4️⃣ High-Level Architecture (Mermaid Diagram)")

    st.markdown("Copy/paste the snippet below into Mermaid Live, Notion, Obsidian, or GitHub preview.")

    # SAFE triple-quoted Mermaid snippet (no indentation issues)
    mermaid_snippet = """
flowchart LR
    subgraph DataLayer[Data Sources]
        GJSON[governance_decisions.json<br/>(GitHub / S3 / DB)]
    end

    subgraph Processing[Processing Engine]
        Clean[Data Cleaning & Normalization]
        Metrics[Portfolio Metrics]
        Reviews[Review Calendar Engine]
        Insights[Risk Insights]
    end

    subgraph Portal[AI Governance Portal]
        Dashboard[Executive Dashboard]
        Heatmap[Risk & Autonomy Heatmap]
        ReviewTable[Upcoming Reviews]
        Spotlight[High-Risk Spotlight]
        Timeline[Lifecycle Timeline]
        Table[Agents Table]
        Detail[Agent Detail Pages]
    end

    GJSON --> Clean --> Metrics --> Dashboard
    Metrics --> Heatmap
    Reviews --> ReviewTable
    Insights --> Spotlight
    Metrics --> Timeline
    Metrics --> Table
    Metrics --> Detail
    """

    st.code(mermaid_snippet, language="markdown")

    st.markdown("### Architecture Interpretation (Executive View)")
    st.markdown("""
    • **Data Layer** — Single source of truth for agent metadata, risk, autonomy, lifecycle, ownership, and cadence.  
    • **Processing Engine** — Normalizes input, calculates health metrics, detects review windows, and derives insights.  
    • **Portal Experience** — Clear navigation for executives, auditors, HR, Security, and product teams.  
    • **Governance Value** — Provides accountability, reduces audit friction, and offers continuous risk visibility.  
    """)

# ====================================================================
# FOOTER / END
# ====================================================================

st.markdown("---")
st.caption("AI Agent Governance Portal • Built with Streamlit • Executive & Audit Ready")
