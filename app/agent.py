import asyncio
import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.nodes.query_analyzer import query_analyzer_node
from app.nodes.web_navigator import web_navigator_node
from app.nodes.rag_indexer import rag_indexer_node
from app.nodes.synthesizer import synthesizer_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NeuralFetch.Agent")

def route_after_navigation(state: AgentState) -> str:
    """Conditional routing based on web navigation results."""
    scraped_docs = state.get("scraped_documents", [])
    if not scraped_docs:
        logger.warning("No scraped documents found. Routing directly to synthesizer with fallback.")
        return "synthesizer"
    return "rag_indexer"

def build_graph() -> StateGraph:
    """
    Constructs the LangGraph multi-agent state graph for Intelligent Web Information Retrieval:
    1. Query Analyzer: Analyzes question, identifies URLs or search terms.
    2. Web Navigator: Discovers URLs and scrapes HTML into text.
    3. RAG Indexer: Chunks content, embeds locally (Hugging Face / ROCm), and stores in local Qdrant.
    4. Synthesizer: Queries Qdrant vectors, extracts exact answer, formats citations.
    """
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("query_analyzer", query_analyzer_node)
    workflow.add_node("web_navigator", web_navigator_node)
    workflow.add_node("rag_indexer", rag_indexer_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define Workflow Edges
    workflow.add_edge(START, "query_analyzer")
    workflow.add_edge("query_analyzer", "web_navigator")
    
    # Conditional branching after scraping
    workflow.add_conditional_edges(
        "web_navigator",
        route_after_navigation,
        {
            "rag_indexer": "rag_indexer",
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_edge("rag_indexer", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow

# Compile the graph into an executable agent
workflow = build_graph()
agent = workflow.compile()

async def run_agent(question: str) -> Dict[str, Any]:
    """
    Primary interface to execute the Intelligent Web Information Retrieval Agent.
    
    Args:
        question (str): User question in natural language.
        
    Returns:
        Dict[str, Any]: Resulting state with 'answer', 'citations', 'retrieved_chunks', etc.
    """
    initial_state: AgentState = {
        "question": question,
        "search_queries": [],
        "target_urls": [],
        "intent_summary": "",
        "scraped_documents": [],
        "indexed_chunks_count": 0,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "status": "initialized",
        "error": None
    }
    
    logger.info(f"Starting agent execution for question: '{question}'")
    result_state = await agent.ainvoke(initial_state)
    logger.info(f"Agent finished execution with status: {result_state.get('status')}")
    return result_state

if __name__ == "__main__":
    import sys
    # Ensure Windows console handles UTF-8 characters without CharMap crashes
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    test_question = sys.argv[1] if len(sys.argv) > 1 else "What are the latest architecture features of AMD ROCm 6.0?"
    print(f"\n================ Running NeuralFetch Agent ================")
    print(f"Query: {test_question}\n")
    
    final_state = asyncio.run(run_agent(test_question))
    
    print("\n---------------- FINAL ANSWER ----------------")
    print(final_state.get("answer", ""))
    print("\n---------------- CITATIONS ----------------")
    for cit in final_state.get("citations", []):
        print(f"- URL: {cit['url']} (Score: {cit['score']})")
