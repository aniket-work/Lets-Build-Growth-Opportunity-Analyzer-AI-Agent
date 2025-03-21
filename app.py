import streamlit as st
import pandas as pd
import re
from io import StringIO

from src.logger import setup_logger
from src.agents import create_agents
from src.utils import extract_csv_data

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Business Opportunity Analyzer", layout="wide")
st.title("🚀 Business Opportunity Analyzer AI Agent")
st.caption(
    "Analyze the latest business trends with AI-powered news analysis, summarization, and insights — complete with dynamic visualizations and source citations.")

# --- Initialize Logger ---
logger = setup_logger()
logger.info("Application started.")

# --- User Input ---
with st.container():
    st.markdown("### 🌐 Enter Your Topic of Interest")
    topic = st.text_input("Enter the business area you're interested in:",
                          placeholder="e.g., Renewable Energy, Fintech, AI Startups")

# --- Main Processing ---
if st.button("🔍 Discover Opportunities"):
    if not topic.strip():
        st.warning("⚠️ Please enter a valid topic before proceeding.")
    else:
        with st.spinner("Processing your request..."):
            logger.info(f"User input topic: {topic}")
            try:
                # Initialize agents from our configuration
                news_collector, summary_writer, trend_analyzer = create_agents(topic, logger)

                # --- Workflow Execution ---
                logger.info("Step 1: Collecting news articles...")
                news_response = news_collector.run(f"Collect recent business news on {topic}")
                articles = news_response.content
                logger.info(f"News collection complete. Articles snippet: {articles[:300]}")

                logger.info("Step 2: Summarizing collected articles...")
                summary_response = summary_writer.run(f"Summarize these articles with citations:\n{articles}")
                summaries = summary_response.content
                logger.info(f"Summary complete. Summaries snippet: {summaries[:300]}")

                logger.info("Step 3: Analyzing trends and extracting visualization data...")
                trend_response = trend_analyzer.run(f"Analyze business trends from the summaries:\n{summaries}")
                analysis = trend_response.content
                logger.info(f"Trend analysis complete. Analysis snippet: {analysis[:300]}")

                # --- Display Analysis ---
                st.subheader("📊 Business Trend Analysis Report")
                st.markdown(analysis)

                # --- Extract and Visualize CSV Data ---
                st.divider()
                st.subheader("📈 Dynamic Trend Visualization")
                csv_data = extract_csv_data(analysis)
                if csv_data:
                    try:
                        df = pd.read_csv(StringIO(csv_data))
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df = df.dropna(subset=['Date'])
                        if not df.empty:
                            st.line_chart(df.set_index('Date'))
                            logger.info("Dynamic trend visualization generated successfully.")
                        else:
                            st.info("No valid date data found for visualization.")
                    except Exception as e:
                        logger.warning(f"Failed to parse extracted trend data: {e}")
                        st.info("⚠️ Trend data was extracted but could not be visualized due to parsing issues.")
                else:
                    st.info(
                        "ℹ️ No structured trend data found for visualization. Consider rephrasing your request or checking agent outputs.")

            except Exception as e:
                logger.exception("Error during processing.")
                st.error(f"❌ An error occurred during processing: {str(e)}")
else:
    st.info("💡 Enter a topic and click **Discover Opportunities** to start.")

# --- Display Detailed Logs ---
with st.expander("📝 View Detailed Logs"):
    try:
        log_filename = logger.handlers[0].baseFilename  # assuming the file handler is first
        with open(log_filename, 'r') as log_file:
            log_content = log_file.read()
            st.text_area("Log Output", log_content, height=300)
    except Exception as log_error:
        st.error("Error loading logs.")
        logger.exception("Error loading log file.")
