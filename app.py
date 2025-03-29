import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Medical Tourism Lead Dashboard", layout="wide")
st.title("🏥 Medical Tourism Lead Analysis Tool")

# --- Upload and Load Data ---
st.sidebar.header("1. Upload Your Lead Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df, use_container_width=True)

    # --- Filters ---
    st.sidebar.header("2. Filter Data")
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

    st.subheader("📊 Filtered Leads")
    st.dataframe(filtered_df, use_container_width=True)

    # --- Visualizations ---
    st.subheader("📈 Visual Insights")

    if not filtered_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(filtered_df, x="Country", title="Leads by Country")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(filtered_df, x="Procedure Name", title="Most Requested Procedures")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            status_counts = filtered_df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig3 = px.pie(status_counts, names="Status", values="Count", title="Lead Status Distribution")
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            agent_counts = filtered_df["Assigned To"].value_counts().reset_index()
            agent_counts.columns = ["Agent", "Count"]
            fig4 = px.bar(agent_counts, x="Agent", y="Count", title="Leads by Agent")
            st.plotly_chart(fig4, use_container_width=True)

    # --- Ask a Question ---
    st.subheader("💬 Ask a Question about Your Leads")
    user_question = st.text_input("Type your question (e.g., 'Which procedures are most requested from the UK?')")

    if user_question:
        with st.spinner("Thinking..."):
            SYSTEM_PROMPT = """
            You are a data assistant for a hospital operating in the medical tourism space. You are helping the team analyze lead data and extract useful insights.

            Dataset fields:
            - Status, Assigned To, Created Date, Patient Name, Phone Number, Email, Country, Nationality, Subject (Query), Procedure Name, Appointment Date, Doctor Assigned, Patient Reply, Ref Company, MRN.

            Your goals:
            1. Answer user questions about the dataset.
            2. Identify patterns or anomalies.
            3. Suggest ways to improve conversions and efficiency.
            4. Be specific, clear, and actionable.
            """

            # Convert DataFrame to CSV string to pass as context
            csv_data = filtered_df.to_csv(index=False)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the dataset:\n{csv_data}\n\nQuestion: {user_question}"}
            ]

            openai.api_key = st.secrets["OPENAI_API_KEY"]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=800
            )

            st.markdown("### 🤖 Answer")
            st.write(response.choices[0].message.content)

    # --- Suggest Insights ---
    if st.button("🔎 Generate Suggested Insights"):
        with st.spinner("Generating insights from your data..."):
            suggestion_prompt = f"Based on this dataset, suggest 5 insightful findings relevant to hospital lead management and medical tourism: \n{csv_data}"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": suggestion_prompt}
                ],
                max_tokens=600
            )
            st.markdown("### 📌 Suggested Insights")
            st.write(response.choices[0].message.content)

else:
    st.info("👈 Upload your lead CSV file to get started.")
