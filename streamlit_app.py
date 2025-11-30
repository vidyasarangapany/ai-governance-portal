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
    initial_sidebar_state="expanded",
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

        # Normalize
        df["risk_level"] = df["risk_level"].astype(str).str.upper()
        df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()

        risk_map = {"LOW RISK": 1, "MEDIUM RISK": 2, "HIGH RISK": 3}
        df["risk_score"] = df["risk_level"].map(risk_map).fillna(0)

        return df
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return pd.DataFrame()

# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

uploaded_json = st.sidebar.file_uploader(
    "Upload governance_decisions.json", type=["json"]
)

df = load_data(uploaded_json)
st.sidebar.success(f"Loaded {len(df)} agents")

risk_filter = st.sidebar.selectbox("Filter by Risk Level", ["All"] + sorted(df["risk_level"].unique()))
auto_filter = st.sidebar.selectbox("Filter by Autonomy Level", ["All"] + sorted(df["autonomy_level"].unique()))

dept_col = None
for col in df.columns:
    if col.lower() in ["dept", "department"]:
        dept_col = col
        break

if dept_col:
    dept_filter = st.sidebar.selectbox(
        "Filter by Department",
        ["All"] + sorted(df[dept_col].dropna().unique())
    )
else:
    dept_filter = "All"

filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if auto_filter != "All":
    filtered = filtered[filtered["autonomy_level"] == auto_filter]
if dept_filter != "All" and dept_col:
    filtered = filtered[dept_col] == dept_filter

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Agents Table", "Agent Detail", "Insights"],
    format_func=lambda x: {
        "Overview": "🏠 Overview",
        "Agents Table": "📋 Agents Table",
        "Agent Detail": "🔍 Agent Detail",
        "Insights": "💡 Insights",
    }[x],
)

# -------------------------------------------------------
# KPI SECTION
# -------------------------------------------------------
def render_kpis():
    total = len(df)
    high = (df["risk_level"] == "HIGH RISK").sum()
    medium = (df["risk_level"] == "MEDIUM RISK").sum()
    low = (df["risk_level"] == "LOW RISK").sum()

    hloop = (df["autonomy_level"] == "HUMAN_IN_LOOP").sum()
    limited = (df["autonomy_level"] == "LIMITED_AUTONOMY").sum()
    auto_allowed = (df["autonomy_level"] == "AUTO_ALLOWED").sum()
    no_auto = (df["autonomy_level"] == "NO_AUTONOMY").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Agents", total)
    c2.metric("High Risk", high)
    c3.metric("Medium Risk", medium)
    c4.metric("Low Risk", low)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("No Autonomy", no_auto)
    c6.metric("Human-in-Loop", hloop)
    c7.metric("Limited Autonomy", limited)
    c8.metric("Auto Allowed", auto_allowed)

# -------------------------------------------------------
# PAGE: OVERVIEW
# -------------------------------------------------------
if page == "Overview":
    st.title("🛡️ AI Agent Governance Portal")
    st.caption("Executive dashboard for AI agent risk, autonomy, lifecycle, and governance posture.")

    render_kpis()
    st.markdown("---")

    col1, col2 = st.columns([2, 1.3])

    # Heatmap
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

    # Single Pie Chart
    with col2:
        st.subheader("Risk Breakdown")

        pie_data = df["risk_level"].value_counts().reset_index()
        pie_data.columns = ["risk_level", "count"]

        fig2 = px.pie(
            pie_data,
            names="risk_level",
            values="count",
            hole=0.45,
        )
        fig2.update_traces(textinfo="label+percent")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Review Cadence
    st.subheader("Review Cadence Overview")
    if "review_cadence" in df.columns:
        cad = df["review_cadence"].value_counts().reset_index()
        cad.columns = ["review_cadence_label", "count"]

        fig3 = px.bar(cad, x="review_cadence_label", y="count", text="count")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No review_cadence column in JSON.")

# -------------------------------------------------------
# PAGE: AGENTS TABLE
# -------------------------------------------------------
elif page == "Agents Table":
    st.title("📋 Agents Table")
    st.caption("Filtered view based on sidebar selections.")
    render_kpis()
    st.markdown("---")

    hide_cols = ["reasoning", "autonomy_reasoning", "action", "risk_score"]
    show_cols = [c for c in filtered.columns if c not in hide_cols]

    st.dataframe(filtered[show_cols], use_container_width=True, height=600)

# -------------------------------------------------------
# PAGE: AGENT DETAIL
# -------------------------------------------------------
elif page == "Agent Detail":
    st.title("🔍 Agent Detail View")
    render_kpis()
    st.markdown("---")

    if filtered.empty:
        st.info("No agents available.")
    else:
        selected = st.selectbox("Select an Agent", filtered["agent_name"].unique())
        row = filtered[filtered["agent_name"] == selected].iloc[0]

        st.markdown(f"### 🧩 {row['agent_name']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Level", row["risk_level"])
        c2.metric("Autonomy", row["autonomy_level"])
        c3.metric("Review Cadence", row.get("review_cadence", "N/A"))

        st.subheader("Governance Reasoning")
        st.write(row.get("reasoning", "_No reasoning provided._"))

        if "autonomy_reasoning" in row:
            st.subheader("Autonomy Reasoning")
            st.write(row["autonomy_reasoning"])

        st.subheader("Recommended Action")
        st.info(row.get("action", "_No recommended action provided._"))

# -------------------------------------------------------
# PAGE: INSIGHTS
# -------------------------------------------------------
elif page == "Insights":
    st.title("💡 Key Governance Insights")
    render_kpis()
    st.markdown("---")

    st.subheader("Top 5 Highest-Risk Agents")
    top5 = df.sort_values("risk_score", ascending=False).head(5)
    for _, r in top5.iterrows():
        st.markdown(f"**{r['agent_name']}** — {r['risk_level']} / {r['autonomy_level']}")
        st.write(r.get("reasoning", ""))
        st.markdown("---")

    st.subheader("Agents Missing Owners")
    owners = [c for c in df.columns if "owner" in c.lower()]
    if owners:
        owner_col = owners[0]
        missing = df[df[owner_col].isna() | (df[owner_col] == "Unknown")]
        if missing.empty:
            st.success("All agents have owners.")
        else:
            for _, r in missing.iterrows():
                st.markdown(f"- {r['agent_name']} — {r['risk_level']}")
    else:
        st.info("No owner column found.")
