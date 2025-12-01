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
    try:
        if uploaded_file:
            data = json.load(uploaded_file)
        else:
            with open("governance_decisions.json", "r") as f:
                data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        # Normalize values
        df["risk_level"] = df.get("risk_level", "").astype(str).str.upper()
        df["autonomy_level"] = df.get("autonomy_level", "").astype(str).str.upper()

        # Numeric risk score
        risk_map = {"LOW RISK": 1, "MEDIUM RISK": 2, "HIGH RISK": 3}
        df["risk_score"] = df["risk_level"].map(risk_map).fillna(0)

        # Lifecycle fallback
        if "lifecycle_state" not in df.columns:
            df["lifecycle_state"] = "DEPLOYED"

        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()


# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

uploaded_json = st.sidebar.file_uploader(
    "Upload governance_decisions.json",
    type=["json"]
)

df = load_data(uploaded_json)
st.sidebar.success(f"Loaded {len(df)} agents")

risk_filter = st.sidebar.selectbox("Filter by Risk Level", ["All"] + sorted(df["risk_level"].unique()))
auto_filter = st.sidebar.selectbox("Filter by Autonomy Level", ["All"] + sorted(df["autonomy_level"].unique()))
lifecycle_filter = st.sidebar.selectbox("Filter by Lifecycle State", ["All"] + sorted(df["lifecycle_state"].unique()))

# Apply filters
filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if auto_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == auto_filter]
if lifecycle_filter != "All":
    filtered = filtered[filtered["lifecycle_state"] == lifecycle_filter]

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Agents Table", "Agent Detail", "Insights"]
)


# -------------------------------------------------------
# KPI Block
# -------------------------------------------------------
def render_kpis(data):

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Agents", len(data))
    c2.metric("High Risk", (data["risk_level"] == "HIGH RISK").sum())
    c3.metric("Medium Risk", (data["risk_level"] == "MEDIUM RISK").sum())
    c4.metric("Low Risk", (data["risk_level"] == "LOW RISK").sum())

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("No Autonomy", (data["autonomy_level"] == "NO_AUTONOMY").sum())
    c6.metric("Human-in-Loop", (data["autonomy_level"] == "HUMAN_IN_LOOP").sum())
    c7.metric("Limited Autonomy", (data["autonomy_level"] == "LIMITED_AUTONOMY").sum())
    c8.metric("Auto Allowed", (data["autonomy_level"] == "AUTO_ALLOWED").sum())


# -------------------------------------------------------
# PAGE: Overview
# -------------------------------------------------------
if page == "Overview":
    st.title("🛡️ AI Agent Governance Portal")
    st.caption("Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture.")

    render_kpis(df)
    st.markdown("---")

    col1, col2 = st.columns([2, 1.5])

    # --------------------------
    # Risk vs Autonomy Heatmap
    # --------------------------
    with col1:
        st.subheader("Risk vs Autonomy Heatmap")

        heat = df.groupby(["risk_level", "autonomy_level"]).size().reset_index(name="count")
        pivot = heat.pivot(index="risk_level", columns="autonomy_level", values="count").fillna(0)

        fig = px.imshow(
            pivot,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Autonomy Level", y="Risk Level", color="Agents"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # Risk Breakdown (FIXED)
    # --------------------------
    with col2:
        st.subheader("Risk Breakdown")

        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]  # FIX: no duplicate names

        fig2 = px.pie(
            risk_counts,
            names="risk",
            values="count",
            hole=0.45
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --------------------------
    # NEW: Lifecycle State Overview
    # --------------------------
    st.subheader("Lifecycle State Overview")

    lifecycle = df["lifecycle_state"].value_counts().reset_index()
    lifecycle.columns = ["state", "count"]

    fig3 = px.bar(lifecycle, x="state", y="count", text="count")
    st.plotly_chart(fig3, use_container_width=True)


# -------------------------------------------------------
# PAGE: Agents Table
# -------------------------------------------------------
elif page == "Agents Table":
    st.title("📋 Agents Table")

    st.dataframe(filtered, use_container_width=True)


# -------------------------------------------------------
# PAGE: Agent Detail
# -------------------------------------------------------
elif page == "Agent Detail":
    st.title("🔍 Agent Detail")

    agent_list = df["agent_name"].unique().tolist()
    selected = st.selectbox("Select an Agent", agent_list)

    row = df[df["agent_name"] == selected].iloc[0]

    st.subheader(f"Details for: {selected}")

    st.write(row)


# -------------------------------------------------------
# PAGE: Insights
# -------------------------------------------------------
elif page == "Insights":
    st.title("💡 Insights")

    st.subheader("Top 5 Highest-Risk Agents")
    top5 = df.sort_values(by="risk_score", ascending=False).head(5)

    for _, r in top5.iterrows():
        st.markdown(f"### {r['agent_name']} — {r['risk_level']}")
        st.write(r.get("reasoning", "No reasoning provided."))
        st.markdown("---")

