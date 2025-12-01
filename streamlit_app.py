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
# Load Data
# -------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            data = json.load(uploaded_file)
        else:
            with open("governance_decisions.json", "r") as f:
                data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        # Normalize risk & autonomy
        if "risk_level" in df.columns:
            df["risk_level"] = df["risk_level"].astype(str).str.upper()

        if "autonomy_level" in df.columns:
            df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()

        # Normalize lifecycle
        if "lifecycle_state" in df.columns:
            df["lifecycle_state"] = df["lifecycle_state"].astype(str)

        # Risk scoring
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
# Sidebar
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

uploaded_json = st.sidebar.file_uploader(
    "Upload governance_decisions.json",
    type=["json"]
)

df = load_data(uploaded_json)

st.sidebar.success(f"Loaded {len(df)} agents")

# Filters
risk_filter = st.sidebar.selectbox("Filter by Risk Level", ["All"] + sorted(df["risk_level"].unique()))
auto_filter = st.sidebar.selectbox("Filter by Autonomy Level", ["All"] + sorted(df["autonomy_level"].unique()))

dept_col = None
for c in df.columns:
    if c.lower() in ["department", "dept"]:
        dept_col = c
        break

dept_filter = st.sidebar.selectbox(
    "Filter by Department",
    ["All"] + sorted(df[dept_col].dropna().unique())
) if dept_col else "All"

filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if auto_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == auto_filter]
if dept_filter != "All" and dept_col:
    filtered = filtered[filtered[dept_col] == dept_filter]

# Navigation (FIXED)
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📋 Agents Table", "🔍 Agent Detail", "💡 Insights"]
)

# -------------------------------------------------------
# KPI Renderer
# -------------------------------------------------------
def render_kpis(data):
    total_agents = len(data)
    high_risk = (data["risk_level"] == "HIGH RISK").sum()
    med_risk = (data["risk_level"] == "MEDIUM RISK").sum()
    low_risk = (data["risk_level"] == "LOW RISK").sum()

    human_loop = (data["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited_auto = (data["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (data["autonomy_level"] == "AUTO_ALLOWED").sum()
    no_auto = (data["autonomy_level"] == "NO_AUTONOMY").sum()

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
# PAGE: OVERVIEW
# -------------------------------------------------------
if page == "🏠 Overview":
    st.title("🛡️ AI Agent Governance Portal")
    st.caption("Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture.")

    render_kpis(filtered)
    st.markdown("---")

    col1, col2 = st.columns([2, 1.5])

    # Heatmap
    with col1:
        st.subheader("Risk vs Autonomy Heatmap")
        heat = (
            filtered.groupby(["risk_level", "autonomy_level"])
            .size()
            .reset_index(name="count")
        )
        pivot = heat.pivot(index="risk_level", columns="autonomy_level", values="count").fillna(0)
        fig = px.imshow(pivot, text_auto=True, aspect="auto",
                        labels=dict(x="Autonomy Level", y="Risk Level", color="Agents"))
        st.plotly_chart(fig, use_container_width=True)

    # Pie Chart (FIXED)
    with col2:
        st.subheader("Risk Breakdown")
        rc = (
            filtered["risk_level"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "risk_level", "risk_level": "count"})
        )
        fig2 = px.pie(rc, names="risk_level", values="count", hole=0.45)
        fig2.update_traces(textinfo="label+percent")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Lifecycle View (NEW - Step B)
    st.subheader("Agent Lifecycle Distribution")
    if "lifecycle_state" in filtered.columns:
        lc = (
            filtered["lifecycle_state"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "lifecycle_state", "lifecycle_state": "count"})
        )
        fig3 = px.bar(lc, x="lifecycle_state", y="count", text="count")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No lifecycle_state in JSON.")

# -------------------------------------------------------
# PAGE: AGENTS TABLE
# -------------------------------------------------------
elif page == "📋 Agents Table":
    st.title("📋 Agents Table")
    st.caption("Filtered view based on sidebar selections.")
    render_kpis(filtered)
    st.markdown("---")

    drop_cols = ["reasoning", "autonomy_reasoning", "action"]
    display_cols = [c for c in filtered.columns if c not in drop_cols]

    st.dataframe(filtered[display_cols], use_container_width=True, height=600)

# -------------------------------------------------------
# PAGE: AGENT DETAIL
# -------------------------------------------------------
elif page == "🔍 Agent Detail":
    st.title("🔍 Agent Detail View")
    st.caption("Deep dive into a single agent.")
    render_kpis(filtered)
    st.markdown("---")

    if filtered.empty:
        st.info("No agents match current filters.")
    else:
        agent_names = filtered["agent_name"].unique().tolist()
        selected = st.selectbox("Select agent", agent_names)
        row = filtered[filtered["agent_name"] == selected].iloc[0]

        st.markdown(f"### 🧩 {row['agent_name']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Level", row["risk_level"])
        c2.metric("Autonomy", row["autonomy_level"])
        c3.metric("Lifecycle", row.get("lifecycle_state", "N/A"))

        st.subheader("Governance Reasoning")
        st.write(row.get("reasoning", "No reasoning provided."))

        if "autonomy_reasoning" in row:
            st.subheader("Autonomy Reasoning")
            st.write(row["autonomy_reasoning"])

        st.subheader("Recommended Action")
        st.info(row.get("action", "No action provided."))

# -------------------------------------------------------
# PAGE: INSIGHTS
# -------------------------------------------------------
elif page == "💡 Insights":
    st.title("💡 Key Governance Insights")
    render_kpis(filtered)
    st.markdown("---")

    st.subheader("Top 5 Highest-Risk Agents")
    top5 = filtered.sort_values(by="risk_score", ascending=False).head(5)
    for _, r in top5.iterrows():
        st.markdown(f"### {r['agent_name']} — {r['risk_level']}")
        st.write(r.get("reasoning", ""))
        st.info(f"Action: {r.get('action', '')}")
        st.markdown("---")
