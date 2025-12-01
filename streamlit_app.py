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
# Load Data Function
# -------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    """Load governance_decisions.json either from uploader or local file."""
    try:
        if uploaded_file:
            data = json.load(uploaded_file)
        else:
            with open("governance_decisions.json", "r") as f:
                data = json.load(f)

        # Normalise to list
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        # Normalise risk / autonomy text
        if "risk_level" in df.columns:
            df["risk_level"] = df["risk_level"].astype(str).str.upper()

        if "autonomy_level" in df.columns:
            df["autonomy_level"] = df["autonomy_level"].astype(str).str.upper()

        # Numeric risk score
        risk_map = {
            "LOW RISK": 1,
            "MEDIUM RISK": 2,
            "HIGH RISK": 3,
        }
        if "risk_level" in df.columns:
            df["risk_score"] = df["risk_level"].map(risk_map).fillna(0)
        else:
            df["risk_score"] = 0

        # Ensure lifecycle_state exists (synthetic if missing)
        if "lifecycle_state" not in df.columns:
            df["lifecycle_state"] = "DEPLOYED"
        else:
            # Fill any missing with a default
            df["lifecycle_state"] = df["lifecycle_state"].fillna("DEPLOYED")

        return df

    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return pd.DataFrame()


# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

uploaded_json = st.sidebar.file_uploader(
    "Upload governance_decisions.json",
    type=["json"],
    help="Use this to refresh dashboard data without redeploying.",
)

df = load_data(uploaded_json)

st.sidebar.success(f"Loaded {len(df)} agents")

# Risk filter
if "risk_level" in df.columns and not df.empty:
    risk_choices = ["All"] + sorted(df["risk_level"].unique().tolist())
else:
    risk_choices = ["All"]

risk_filter = st.sidebar.selectbox("Filter by Risk Level", risk_choices)

# Autonomy filter
if "autonomy_level" in df.columns and not df.empty:
    auto_choices = ["All"] + sorted(df["autonomy_level"].unique().tolist())
else:
    auto_choices = ["All"]

auto_filter = st.sidebar.selectbox("Filter by Autonomy Level", auto_choices)

# Department filter (if present)
dept_col = None
for c in df.columns:
    if c.lower() in ["department", "dept"]:
        dept_col = c
        break

if dept_col and not df.empty:
    dept_choices = ["All"] + sorted(df[dept_col].dropna().unique().tolist())
    dept_filter = st.sidebar.selectbox("Filter by Department", dept_choices)
else:
    dept_filter = "All"

# Lifecycle filter
if "lifecycle_state" in df.columns and not df.empty:
    life_choices = ["All"] + sorted(df["lifecycle_state"].unique().tolist())
else:
    life_choices = ["All"]

life_filter = st.sidebar.selectbox("Filter by Lifecycle State", life_choices)

# Apply filters to build `filtered`
filtered = df.copy()
if risk_filter != "All" and "risk_level" in filtered.columns:
    filtered = filtered[filtered["risk_level"] == risk_filter]

if auto_filter != "All" and "autonomy_level" in filtered.columns:
    filtered = filtered[filtered["autonomy_level"] == auto_filter]

if dept_filter != "All" and dept_col:
    filtered = filtered[filtered[dept_col] == dept_filter]

if life_filter != "All" and "lifecycle_state" in filtered.columns:
    filtered = filtered[filtered["lifecycle_state"] == life_filter]


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
def render_kpis(source_df: pd.DataFrame):
    if source_df.empty:
        total_agents = 0
        high_risk = med_risk = low_risk = 0
        human_loop = limited_auto = auto_allowed = no_auto = 0
    else:
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

    # Use full df for KPIs so they reflect global posture
    render_kpis(df)
    st.markdown("---")

    col1, col2 = st.columns([2, 1.5])

    # ---------- Risk vs Autonomy Heatmap ----------
    with col1:
        st.subheader("Risk vs Autonomy Heatmap")

        if not df.empty and {"risk_level", "autonomy_level"}.issubset(df.columns):
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
                    labels=dict(
                        x="Autonomy Level", y="Risk Level", color="Agents"
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available to generate heatmap.")
        else:
            st.info("Risk / autonomy columns not found in data.")

    # ---------- Risk Breakdown Pie ----------
    with col2:
        st.subheader("Risk Breakdown")

        if not df.empty and "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]

            if not risk_counts.empty:
                fig2 = px.pie(
                    data_frame=risk_counts,
                    names="risk_level",
                    values="count",
                    hole=0.45,
                )
                fig2.update_traces(textinfo="label+percent")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No risk data available for breakdown.")
        else:
            st.info("No risk_level column in JSON.")

    st.markdown("---")

    # ---------- Review Cadence Overview ----------
    st.subheader("Review Cadence Overview")
    if not df.empty and "review_cadence" in df.columns:
        cad = df["review_cadence"].value_counts().reset_index()
        cad.columns = ["review_cadence_label", "count"]

        if not cad.empty:
            fig3 = px.bar(
                data_frame=cad,
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

    # ---------- Lifecycle State Overview ----------
    st.subheader("Lifecycle State Overview")
    if not df.empty and "lifecycle_state" in df.columns:
        life_counts = df["lifecycle_state"].value_counts().reset_index()
        life_counts.columns = ["state", "count"]

        if not life_counts.empty:
            fig_life = px.bar(
                data_frame=life_counts,
                x="state",
                y="count",
                text="count",
            )
            st.plotly_chart(fig_life, use_container_width=True)
        else:
            st.info("No lifecycle_state data available.")
    else:
        st.info("No lifecycle_state column in JSON.")


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
        hidden_cols = ["reasoning", "autonomy_reasoning", "action"]
        display_cols = [c for c in filtered.columns if c not in hidden_cols]

        st.dataframe(
            filtered[display_cols].sort_values(
                by="risk_score", ascending=False
            ),
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
        if "agent_name" not in filtered.columns:
            st.info("No 'agent_name' column found in JSON.")
        else:
            agent_list = filtered["agent_name"].unique().tolist()
            selected = st.selectbox("Choose an agent", agent_list)

            row = filtered[filtered["agent_name"] == selected].iloc[0]

            st.markdown(f"### 🧩 {row['agent_name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Level", row.get("risk_level", "N/A"))
            c2.metric("Autonomy", row.get("autonomy_level", "N/A"))
            c3.metric("Lifecycle", row.get("lifecycle_state", "N/A"))

            st.markdown("#### Governance Reasoning")
            st.write(row.get("reasoning", "_No reasoning provided._"))

            if "autonomy_reasoning" in row:
                st.markdown("#### Autonomy Reasoning")
                st.write(row.get("autonomy_reasoning", "_None provided._"))

            st.markdown("#### Recommended Action")
            st.info(row.get("action", "_No action provided._"))


# -------------------------------------------------------
# PAGE: Insights
# -------------------------------------------------------
elif page == "Insights":
    st.title("🚨 Key Governance Insights")
    st.caption("Summaries for executives, audit teams, and risk leaders.")

    render_kpis(df)
    st.markdown("---")

    if df.empty:
        st.info("No data available for insights.")
    else:
        st.subheader("Top 5 Highest-Risk Agents")
        top5 = df.sort_values(by="risk_score", ascending=False).head(5)
        for _, r in top5.iterrows():
            st.markdown(
                f"**{r.get('agent_name', 'Unknown Agent')}** — "
                f"{r.get('risk_level', 'N/A')}, "
                f"{r.get('autonomy_level', 'N/A')}"
            )
            st.write(r.get("reasoning", ""))
            st.markdown(f"**Action:** {r.get('action', '')}")
            st.markdown("---")

        # Missing owners section
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
                    st.markdown(
                        f"- {r.get('agent_name', 'Unknown Agent')} — "
                        f"{r.get('risk_level', 'N/A')}"
                    )
        else:
            st.info("No 'owner' column found in JSON.")
