import os
import logging
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import argparse
from rich import print as rprint

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate API keys
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY or TAVILY_API_KEY in .env file")

# Define the shared state
class AgentState(TypedDict):
    """State shared across agents."""
    query: str
    collected_data: List[str]
    draft: str
    revision_number: int
    max_revisions: int

def setup_llm(model_name: str) -> ChatOpenAI:
    """Initialize the LLM with the specified model."""
    try:
        return ChatOpenAI(model=model_name, temperature=0)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def setup_tavily(max_results: int = 5) -> TavilySearchResults:
    """Initialize Tavily search tool."""
    try:
        return TavilySearchResults(max_results=max_results)
    except Exception as e:
        logger.error(f"Failed to initialize Tavily tool: {e}")
        raise

def research_node(state: AgentState) -> AgentState:
    """Research Agent: Collects and filters web data using Tavily."""
    query = state["query"]
    logger.info(f"Searching for: {query}")
    
    try:
        tavily_tool = setup_tavily()
        results = tavily_tool.invoke(query)
        collected_data = [result["content"] for result in results if "content" in result]
        
        # Basic relevance filtering
        relevant_data = [data for data in collected_data if query.lower() in data.lower()]
        
        logger.info(f"Found {len(relevant_data)} relevant items")
        return {
            "collected_data": relevant_data,
            "revision_number": state.get("revision_number", 0)
        }
    except Exception as e:
        logger.error(f"Research node failed: {e}")
        return {"collected_data": [], "revision_number": state.get("revision_number", 0)}

def drafter_node(state: AgentState) -> AgentState:
    """Answer Drafter Agent: Generates a structured report from collected data."""
    logger.info("Generating draft...")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert writer tasked with creating a structured report.
            Organize the information into a markdown report with sections: Introduction, Findings, Conclusion.
            Use the provided data: {data}"""),
            ("user", "Query: {query}\nWrite a report based on the collected data.")
        ])
        
        llm = setup_llm(state.get("model_name", "gpt-4o"))
        chain = prompt | llm
        response = chain.invoke({
            "query": state["query"],
            "data": "\n".join(state["collected_data"])
        })
        
        draft = response.content
        revision_number = state.get("revision_number", 0) + 1
        
        logger.info(f"Draft revision {revision_number} completed")
        return {
            "draft": draft,
            "revision_number": revision_number
        }
    except Exception as e:
        logger.error(f"Drafter node failed: {e}")
        return {"draft": "", "revision_number": state.get("revision_number", 0) + 1}

def should_continue(state: AgentState) -> str:
    """Determine if drafting should continue or end."""
    if state["revision_number"] >= state.get("max_revisions", 2):
        return END
    return "drafter"

def setup_workflow() -> StateGraph:
    """Set up the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("research", research_node)
    workflow.add_node("drafter", drafter_node)
    workflow.add_edge("research", "drafter")
    workflow.add_conditional_edges("drafter", should_continue, {END: END, "drafter": "drafter"})
    workflow.set_entry_point("research")
    return workflow

def run_research(query: str, model_name: str = "gpt-4o", max_revisions: int = 2) -> dict:
    """Run the research workflow for a given query."""
    initial_state = {
        "query": query,
        "collected_data": [],
        "draft": "",
        "revision_number": 0,
        "max_revisions": max_revisions,
        "model_name": model_name
    }

    config = {
        "configurable": {
            "thread_id": "cpu_vs_gpu_test"  # or generate a unique ID if needed
        }
    }
    
    logger.info(f"Starting research for query: {query}")
    try:
        # memory = SqliteSaver.from_conn_string(":memory:")
        # graph = setup_workflow().compile(checkpointer=memory)
        # result = graph.invoke(initial_state)
        with SqliteSaver.from_conn_string(":memory:") as memory:
            graph = setup_workflow().compile(checkpointer=memory)
            result = graph.invoke(initial_state, config=config)
        
        rprint("\n[Final Report]")
        rprint(result["draft"])
        return result
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return {"draft": "", "error": str(e)}

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Deep Research AI Agentic System")
    parser.add_argument("--query", default="Compare CPU and GPU architectures for AI workloads", help="Research query")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--max-revisions", type=int, default=2, help="Maximum draft revisions")
    args = parser.parse_args()
    
    run_research(args.query, args.model, args.max_revisions)

if __name__ == "__main__":
    main()