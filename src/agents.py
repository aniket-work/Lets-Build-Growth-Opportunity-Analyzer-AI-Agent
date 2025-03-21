import json
import yaml
from src.constants import CONFIG_JSON, SETTINGS_YAML
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama
from agno.tools.newspaper4k import Newspaper4kTools

def create_agents(topic, logger):
    with open(CONFIG_JSON, "r") as f:
        config = json.load(f)
    with open(SETTINGS_YAML, "r") as f:
        settings = yaml.safe_load(f)

    model_id = config.get("model_id", "llama3.2:3b")
    ollama_model = Ollama(id=model_id)
    logger.info("Ollama model initialized.")

    news_conf = settings.get("agents", {}).get("news_collector", {})
    search_tool = DuckDuckGoTools(search=True, news=True, fixed_max_results=config.get("news_search", {}).get("max_results", 5))
    news_collector = Agent(
        name=news_conf.get("name", "News Collector"),
        role=news_conf.get("role", ""),
        tools=[search_tool],
        model=ollama_model,
        instructions=news_conf.get("instructions", []),
        show_tool_calls=True,
        markdown=True,
    )
    logger.info("News Collector Agent set up.")

    summary_conf = settings.get("agents", {}).get("summary_writer", {})
    news_tool = Newspaper4kTools(read_article=True, include_summary=True)
    summary_writer = Agent(
        name=summary_conf.get("name", "Summary Writer"),
        role=summary_conf.get("role", ""),
        tools=[news_tool],
        model=ollama_model,
        instructions=summary_conf.get("instructions", []),
        show_tool_calls=True,
        markdown=True,
    )
    logger.info("Summary Writer Agent set up.")

    trend_conf = settings.get("agents", {}).get("trend_analyzer", {})
    trend_analyzer = Agent(
        name=trend_conf.get("name", "Trend Analyzer"),
        role=trend_conf.get("role", ""),
        model=ollama_model,
        instructions=trend_conf.get("instructions", []),
        show_tool_calls=True,
        markdown=True,
    )
    logger.info("Trend Analyzer Agent set up.")

    return news_collector, summary_writer, trend_analyzer
