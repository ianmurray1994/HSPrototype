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
        logo = Image.open(logo_path)
        st.image(logo, width=300)
    else:
        st.error(f"Logo not found at: {logo_path}")
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")

# --- Load Data ---
try:
    df = pd.read_csv("synthetic_leads_data.csv")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# --- Filters ---
st.sidebar.header("Filter Options")
status_filter = st.sidebar.multiselect("Filter by Status", options=df["Status"].unique())
country_filter = st.sidebar.multiselect("Filter by Country", options=df["Country"].dropna().unique())
agent_filter = st.sidebar.multiselect("Filter by Agent", options=df["Assigned To"].dropna().unique())

# Apply filters
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

# --- Key Insights Charts ---
st.subheader("📊 Key Insights")

# Create two columns for the charts
left_col, right_col = st.columns(2)

with left_col:
    # Chart 1: Lead Status Distribution
    status_counts = filtered_df["Status"].value_counts()
    fig1 = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Lead Status Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Top Countries by Lead Count
    country_counts = filtered_df["Country"].value_counts().head(10)
    fig2 = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        title="Top 10 Countries by Lead Count",
        labels={"x": "Country", "y": "Number of Leads"}
    )
    st.plotly_chart(fig2, use_container_width=True)

with right_col:
    # Chart 3: Treatment Type Distribution
    treatment_counts = filtered_df["Procedure Name"].value_counts()
    fig3 = px.pie(
        values=treatment_counts.values,
        names=treatment_counts.index,
        title="Treatment Type Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Chart 4: Assigned Doctor Distribution
    doctor_counts = filtered_df["Doctor Assigned"].value_counts().head(10)
    fig4 = px.bar(
        x=doctor_counts.index,
        y=doctor_counts.values,
        title="Top 10 Assigned Doctors",
        labels={"x": "Doctor", "y": "Number of Leads"}
    )
    fig4.update_xaxes(tickangle=45)
    st.plotly_chart(fig4, use_container_width=True)

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

# --- Ask a Question Section ---
if not filtered_df.empty:
    try:
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
                    
                    # Generate and display the requested chart
                    if chart_config["type"] == "pie":
                        fig = px.pie(
                            filtered_df,
                            names=chart_config["x"],
                            title=chart_config["title"]
                        )
                    else:  # bar chart
                        fig = px.bar(
                            filtered_df,
                            x=chart_config["x"],
                            y=chart_config.get("y"),
                            title=chart_config["title"]
                        )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Display the text response
                    st.write(choice.message.content)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
