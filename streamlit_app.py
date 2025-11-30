import json
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Governance Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Load Data Function
# -------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    """Load governance_decisions.json either from uploader or repo."""
    try:
        if uploaded_file:
            data = json.load(uploaded_file)
        else:
            with open("governance_decisions.json", "r") as f:
                data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        # Normalize columns
        if "risk_level" in df.columns:
            df["risk_level"] = df["risk_level"].astype(str).str.upper()

        if "autonomy_level" in df.columns:
            df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()

        # Risk numeric score
        risk_map = {
            "LOW RISK": 1,
            "MEDIUM RISK": 2,
            "HIGH RISK": 3
        }
        df["risk_score"] = df["risk_level"].map(risk_map).fillna(0)

        return df
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return pd.DataFrame()


# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

# File uploader
uploaded_json = st.sidebar.file_uploader(
    "Upload a new governance_decisions.json",
    type=["json"],
    help="Use this to refresh dashboard data without redeploying."
)

df = load_data(uploaded_json)

st.sidebar.success(f"Loaded {len(df)} agents")

# Filters
risk_filter = st.sidebar.selectbox(
    "Filter by Risk Level",
    ["All"] + sorted(df["risk_level"].unique())
)

auto_filter = st.sidebar.selectbox(
    "Filter by Autonomy Level",
    ["All"] + sorted(df["autonomy_level"].unique())
)

dept_col = None
for c in df.columns:
    if c.lower() in ["department", "dept"]:
        dept_col = c
        break

if dept_col:
    dept_filter = st.sidebar.selectbox(
        "Filter by Department",
        ["All"] + sorted(df[dept_col].dropna().unique())
    )
else:
    dept_filter = "All"

# Apply filters
filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if auto_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == auto_filter]
if dept_filter != "All" and dept_col:
    filtered = filtered[filtered[dept_col] == dept_filter]


# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📋 Agents Table", "🔍 Agent Detail", "🚨 Insights"]
)


# -------------------------------------------------------
# KPI SECTION
# -------------------------------------------------------
def render_kpis():
    total_agents = len(df)
    high_risk = (df["risk_level"] == "HIGH RISK").sum()
    med_risk = (df["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (df["risk_level"] == "LOW RISK").sum()

    human_loop = (df["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited_auto = (df["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (df["autonomy_level"] == "AUTO_ALLOWED").sum()
    no_auto = (df["autonomy_level"] == "NO_AUTONOMY").sum()

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
if page == "📊 Overview":
    st.title("🛡️ AI Agent Governance Portal")
    st.caption("Executive dashboard for AI agent risk, autonomy, and governance posture.")

    render_kpis()
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
        heat_pivot = heat.pivot(
            index="risk_level",
            columns="autonomy_level",
            values="count"
        ).fillna(0)

        fig = px.imshow(
            heat_pivot,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Autonomy Level", y="Risk Level", color="Agents"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Pie chart: Risk distribution
    with col2:
       # Pie chart: Risk breakdown
with col2:
    st.subheader("Risk Breakdown")

    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "risk_level", "risk_level": "count"})
    )

    fig2 = px.pie(
        risk_counts,
        names="risk_level",
        values="count",
        hole=0.45,
    )
    fig2.update_traces(textinfo="label+percent")
    st.plotly_chart(fig2, use_container_width=True)


    st.markdown("---")

    # Review cadence overview
    st.subheader("Review Cadence Overview")
    if "review_cadence" in df.columns:
        cad = (
            df["review_cadence"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "review_cadence", "review_cadence": "count"})
        )
        fig3 = px.bar(
            cad,
            x="review_cadence",
            y="count",
            text="count"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No review_cadence column in JSON.")


# -------------------------------------------------------
# PAGE: Agents Table
# -------------------------------------------------------
elif page == "📋 Agents Table":
    st.title("📋 Agents Table")
    st.caption("Filtered view based on sidebar selections.")

    render_kpis()
    st.markdown("---")

    display_cols = [
        c for c in filtered.columns
        if c not in ["reasoning", "autonomy_reasoning", "action", "risk_score"]
    ]

    st.dataframe(
        filtered[display_cols].sort_values(by="risk_score", ascending=False),
        use_container_width=True,
        height=600
    )


# -------------------------------------------------------
# PAGE: Agent Detail
# -------------------------------------------------------
elif page == "🔍 Agent Detail":
    st.title("🔍 Agent Detail View")
    st.caption("Deep dive for a single agent.")

    render_kpis()
    st.markdown("---")

    agent_list = filtered["agent_name"].unique().tolist()
    if not agent_list:
        st.info("No agents available for selected filters.")
    else:
        selected = st.selectbox("Choose an agent", agent_list)
        row = filtered[filtered["agent_name"] == selected].iloc[0]

        st.markdown(f"### 🧩 {row['agent_name']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Level", row["risk_level"])
        c2.metric("Autonomy", row["autonomy_level"])
        c3.metric("Review Cadence", row.get("review_cadence", "N/A"))

        st.markdown("#### Governance Reasoning")
        st.write(row.get("reasoning", "_No reasoning provided._"))

        if "autonomy_reasoning" in row:
            st.markdown("#### Autonomy Reasoning")
            st.write(row["autonomy_reasoning"])

        st.markdown("#### Recommended Action")
        st.info(row.get("action", "_No action provided._"))


# -------------------------------------------------------
# PAGE: Insights
# -------------------------------------------------------
elif page == "🚨 Insights":
    st.title("🚨 Key Governance Insights")
    st.caption("Summaries for executives, audit teams, and risk leaders.")

    render_kpis()
    st.markdown("---")

    # High-risk agents
    st.subheader("Top 5 Highest-Risk Agents")
    top5 = df.sort_values(by="risk_score", ascending=False).head(5)
    for _, r in top5.iterrows():
        st.markdown(
            f"**{r['agent_name']}** — {r['risk_level']}, {r['autonomy_level']}"
        )
        st.write(r.get("reasoning", ""))
        st.markdown(f"**Action:** {r.get('action', '')}")
        st.markdown("---")

    # Missing owners
    st.subheader("Agents Missing Owners")
    owner_col = None
    for c in df.columns:
        if c.lower() == "owner":
            owner_col = c
            break

    if owner_col:
        missing = df[df[owner_col].isna() | (df[owner_col] == "Unknown")]
        if missing.empty:
            st.success("All agents have an assigned owner.")
        else:
            for _, r in missing.iterrows():
                st.markdown(f"- {r['agent_name']} — {r['risk_level']}")
    else:
        st.info("No 'owner' column found in JSON.")
