import re
import logging
from typing import List, Dict, Any
from app.state import AgentState
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

URL_REGEX = re.compile(r"(?:https?://)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

def query_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Query Analyzer Node:
    Takes a natural language question and determines the best search strategy.
    Identifies target URLs or generates optimized search queries for the Web Navigator.
    """
    question = state["question"].strip()
    logger.info(f"Analyzing user question: {question}")
    
    # 1. Detect explicit URLs in user prompt and normalize
    raw_urls = URL_REGEX.findall(question)
    found_urls: List[str] = []
    for u in raw_urls:
        # Ignore false positives like file extensions or version numbers
        if "." in u and not u.endswith((".py", ".txt", ".json", ".md")):
            normalized = u if u.startswith(("http://", "https://")) else f"https://{u}"
            if normalized not in found_urls:
                found_urls.append(normalized)
    
    # 2. Generate search queries using local LLM or fallback heuristics
    search_queries: List[str] = []
    intent_summary = "General search"

    system_prompt = (
        "You are an expert Query Analyzer for an Intelligent Web Search Agent. "
        "Analyze the user's natural language question and generate 1 to 3 concise, keyword-rich search queries. "
        "Return a JSON object with keys 'search_queries' (list of strings) and 'intent' (brief string)."
    )
    user_prompt = f"User Question: {question}\n\nProvide the search strategy in JSON format."

    try:
        analysis_json = llm_service.invoke_json(prompt=user_prompt, system_prompt=system_prompt)
        search_queries = analysis_json.get("search_queries", [])
        intent_summary = analysis_json.get("intent", "Web information retrieval")
    except Exception as e:
        logger.warning(f"LLM query analysis failed ({e}). Falling back to direct query.")
        # Fallback keyword cleaning
        clean_q = re.sub(r"[^\w\s]", "", question)
        search_queries = [clean_q]

    if not search_queries:
        search_queries = [question]

    return {
        "search_queries": search_queries[:3],
        "target_urls": found_urls,
        "intent_summary": intent_summary,
        "status": "query_analyzed"
    }
