import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
import re
from io import StringIO

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama  # Using local Ollama models
from agno.tools.newspaper4k import Newspaper4kTools

# --- Logging Setup ---
log_filename = f"logs/app_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Business Opportunity Analyzer", layout="wide")
st.title("🚀 Business Opportunity Analyzer AI Agent")
st.caption("Analyze the latest business trends with AI-powered news analysis, summarization, and insights — complete with dynamic visualizations and source citations.")

# --- User Input ---
with st.container():
    st.markdown("### 🌐 Enter Your Topic of Interest")
    topic = st.text_input("Enter the business area you're interested in:", placeholder="e.g., Renewable Energy, Fintech, AI Startups")

# --- Main Processing ---
if st.button("🔍 Discover Opportunities"):
    if not topic.strip():
        st.warning("⚠️ Please enter a valid topic before proceeding.")
    else:
        with st.spinner("Processing your request..."):
            logger.info(f"User input topic: {topic}")
            try:
                # Initialize Ollama model
                logger.info("Initializing Ollama model...")
                ollama_model = Ollama(id="llama3.2:3b")  # Adjust model ID if needed

                # --- Agent Setup ---
                # News Collector Agent: Collects recent business news articles
                logger.info("Setting up News Collector Agent...")
                search_tool = DuckDuckGoTools(search=True, news=True, fixed_max_results=5)
                news_collector = Agent(
                    name="News Collector",
                    role="Collects recent business news articles including source citations.",
                    tools=[search_tool],
                    model=ollama_model,
                    instructions=[
                        "Search for recent news articles about the specified business area.",
                        "Include articles on market performance, investments, and developments.",
                        "Ensure each article has a source or citation (e.g., URL or publication name)."
                    ],
                    show_tool_calls=True,
                    markdown=True,
                )

                # Summary Writer Agent: Summarizes articles with detailed source information
                logger.info("Setting up Summary Writer Agent...")
                news_tool = Newspaper4kTools(read_article=True, include_summary=True)
                summary_writer = Agent(
                    name="Summary Writer",
                    role="Summarizes business news articles with detailed source information.",
                    tools=[news_tool],
                    model=ollama_model,
                    instructions=[
                        "Summarize each article and highlight business trends or insights.",
                        "Include citations or source references for all summarized content."
                    ],
                    show_tool_calls=True,
                    markdown=True,
                )

                # Trend Analyzer Agent: Analyzes summaries and extracts structured trend data for visualization
                logger.info("Setting up Trend Analyzer Agent...")
                trend_analyzer = Agent(
                    name="Trend Analyzer",
                    role="Analyzes summaries to identify trends and generate data for visualization.",
                    model=ollama_model,
                    instructions=[
                        "Analyze the provided summaries to extract emerging business trends and opportunities.",
                        "Provide actionable strategies backed by references to the source articles.",
                        "Extract data that shows trends over time (e.g., monthly mentions, sentiment scores).",
                        "Output the data in a tabular format (CSV) with columns like 'Date' and 'Trend Score' for visualization.",
                        "After the table, provide a brief explanation of the trend observed."
                    ],
                    show_tool_calls=True,
                    markdown=True,
                )

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

                # --- Attempt to Extract CSV Data for Visualization ---
                st.divider()
                st.subheader("📈 Dynamic Trend Visualization")
                csv_pattern = re.compile(r"Date\s*,\s*Trend Score\s*\n(?:.*\n)+", re.IGNORECASE)
                csv_match = csv_pattern.search(analysis)
                if csv_match:
                    csv_data = csv_match.group()
                    try:
                        df = pd.read_csv(StringIO(csv_data))
                        # Convert 'Date' to datetime; allow flexible formats
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df = df.dropna(subset=['Date'])
                        if not df.empty:
                            st.line_chart(df.set_index('Date'))
                            logger.info("Dynamic trend visualization generated successfully.")
                        else:
                            st.info("No valid date data found for visualization.")
                            logger.warning("Extracted CSV did not produce a valid DataFrame after date conversion.")
                    except Exception as e:
                        logger.warning(f"Failed to parse extracted trend data: {e}")
                        st.info("⚠️ Trend data was extracted but could not be visualized due to parsing issues.")
                else:
                    st.info("ℹ️ No structured trend data found for visualization. Consider rephrasing your request or checking agent outputs.")

            except Exception as e:
                logger.exception("Error during processing.")
                st.error(f"❌ An error occurred during processing: {str(e)}")
else:
    st.info("💡 Enter a topic and click **Discover Opportunities** to start.")

# --- Display Detailed Logs ---
with st.expander("📝 View Detailed Logs"):
    try:
        with open(log_filename, 'r') as log_file:
            log_content = log_file.read()
            st.text_area("Log Output", log_content, height=300)
    except Exception as log_error:
        st.error("Error loading logs.")
        logger.exception("Error loading log file.")
