import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import tiktoken
import json
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(page_title="HealthStay Tourism Lead Dashboard", layout="wide")

# Get the absolute path to the logo
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "static", "images", "logo.png")

# --- Custom CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #070969;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    .stSelectbox label, .stMultiSelect label {
        color: white !important;
    }
    .stSelectbox svg, .stMultiSelect svg {
        color: white !important;
    }
    section[data-testid="stSidebar"] > div > div:first-child {
        padding-top: 2rem;
        background-color: #070969;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Header with Logo ---
try:
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)
    else:
        st.error(f"Logo not found at: {logo_path}")
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")
    st.error("Please ensure the logo is a valid PNG image file")

# --- Token Management Helpers ---
MODEL_NAME = "gpt-4"
MAX_TOKENS = 5000

def trim_messages(messages, max_tokens=MAX_TOKENS):
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    total_tokens = 0
    trimmed = []
    for message in reversed(messages):
        message_tokens = sum(len(encoding.encode(v)) for v in message.values())
        if total_tokens + message_tokens > max_tokens:
            break
        trimmed.insert(0, message)
        total_tokens += message_tokens
    return trimmed or [messages[-1]]

# --- System Prompt ---
SYSTEM_PROMPT = """
You are a smart data assistant helping a hospital analyze its medical tourism leads.
You can answer natural language questions, suggest insights, and generate chart configurations.
Use the available chart generation tool when a chart is requested.
"""

# --- Function (Tool) Schema for Chart Requests ---
chart_tool = {
    "type": "function",
    "function": {
        "name": "generate_chart_config",
        "description": "Generate a chart configuration from the dataset and question",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["pie", "bar", "line"]
                },
                "x": {"type": "string"},
                "y": {"type": "string"},
                "filter": {
                    "type": "object",
                    "description": "Optional filter for the chart",
                    "additionalProperties": {"type": "string"}
                },
                "title": {"type": "string"}
            },
            "required": ["type", "x", "title"]
        }
    }
}

try:
    # --- Load Data ---
    df = pd.read_csv("synthetic_leads_data.csv")

    # --- Filters ---
    st.sidebar.header("Filter Options")
    status_filter = st.sidebar.multiselect("Filter by Status", options=df["Status"].unique())
    country_filter = st.sidebar.multiselect("Filter by Country", options=df["Country"].dropna().unique())
    agent_filter = st.sidebar.multiselect("Filter by Agent", options=df["Assigned To"].dropna().unique())

    filtered_df = df.copy()
    if status_filter:
        filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]
    if country_filter:
        filtered_df = filtered_df[filtered_df["Country"].isin(country_filter)]
    if agent_filter:
        filtered_df = filtered_df[filtered_df["Assigned To"].isin(agent_filter)]

    # --- Data Preview ---
    st.subheader("🔍 Data Preview")
    st.dataframe(filtered_df, use_container_width=True)

    if not filtered_df.empty:
        # --- Ask a Question or Request a Chart ---
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        csv_data = filtered_df.head(50).to_csv(index=False)

        st.subheader("💬 Ask a Question or Request a Chart")
        user_question = st.text_input("Ask something like: 'Show a pie chart of leads by status in Qatar'")

        if user_question:
            with st.spinner("Thinking..."):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here is the dataset:\n{csv_data}\n\nQuestion: {user_question}"}
                ]
                trimmed_messages = trim_messages(messages)

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=trimmed_messages,
                        tools=[chart_tool],
                        tool_choice="auto"
                    )

                    choice = response.choices[0]

                    # If GPT used the tool (chart request)
                    if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                        tool_call = choice.message.tool_calls[0]
                        chart_config = json.loads(tool_call.function.arguments)
                        st.markdown(f"### 📊 {chart_config.get('title', 'Generated Chart')}")

                        # Filter dataset
                        df_to_plot = filtered_df.copy()
                        for col, val in chart_config.get("filter", {}).items():
                            df_to_plot = df_to_plot[df_to_plot[col] == val]

                        if df_to_plot.empty:
                            st.warning("⚠️ No data available for this chart. Try a different filter or country.")
                        else:
                            chart_type = chart_config["type"]
                            x = chart_config["x"]
                            y = chart_config.get("y")

                            if chart_type == "pie":
                                fig = px.pie(df_to_plot, names=x, title=chart_config["title"])
                            elif chart_type == "bar":
                                fig = px.bar(df_to_plot, x=x, y=y, title=chart_config["title"])
                            elif chart_type == "line":
                                fig = px.line(df_to_plot, x=x, y=y, title=chart_config["title"])
                            else:
                                st.warning("⚠️ Unsupported chart type.")
                                st.json(chart_config)
                                st.stop()

                            st.plotly_chart(fig, use_container_width=True)

                    else:
                        # GPT answered with natural language (not a chart)
                        st.markdown("### 🤖 Answer")
                        st.write(choice.message.content)

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

        # --- Suggested Insights ---
        if st.button("🔍 Generate Suggested Insights"):
            with st.spinner("Analyzing your data..."):
                insight_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Based on this dataset, suggest 5 insightful findings relevant to hospital lead management and medical tourism:\n{csv_data}"}
                ]
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=insight_messages,
                        max_tokens=600
                    )
                    st.markdown("### 📌 Suggested Insights")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"❌ Failed to generate insights: {e}")

    else:
        st.warning("⚠️ No data available after filtering. Please adjust your filters.")

except FileNotFoundError:
    st.error("❌ Could not find the data file (synthetic_leads_data.csv). Please ensure it is in the correct location.")
