import asyncio
import logging
from typing import Dict, Any, List
from app.state import AgentState, ScrapedDocument
from app.services.scraper_service import scraper_service
from app.config import settings

logger = logging.getLogger(__name__)

async def web_navigator_node(state: AgentState) -> Dict[str, Any]:
    """
    Web Navigator Tool Node:
    Navigates to target websites, scrapes the text, and handles HTML parsing.
    """
    target_urls = list(state.get("target_urls", []))
    search_queries = state.get("search_queries", [])
    
    # If no explicit URLs were extracted from the question, execute web search
    if not target_urls:
        logger.info(f"Navigating web via search queries: {search_queries}")
        found_urls: List[str] = []
        for q in search_queries:
            results = scraper_service.search_web(q, max_results=settings.MAX_SEARCH_RESULTS)
            found_urls.extend(results)
            if len(found_urls) >= settings.MAX_SEARCH_RESULTS:
                break
        
        # De-duplicate while preserving order
        seen = set()
        for u in found_urls:
            if u not in seen:
                seen.add(u)
                target_urls.append(u)

    target_urls = target_urls[:settings.MAX_SEARCH_RESULTS]
    logger.info(f"Target URLs to scrape: {target_urls}")

    if not target_urls:
        logger.warning("No URLs discovered for navigation.")
        return {
            "scraped_documents": [],
            "status": "navigation_no_urls",
            "error": "Could not locate relevant web resources for query."
        }

    scraped_docs = await scraper_service.scrape_multiple_urls(target_urls)
    
    # If direct target URLs yielded no text, fall back to searching search queries
    if not scraped_docs and search_queries:
        logger.info(f"Target URL scrape yielded 0 documents. Executing fallback web search for: {search_queries}")
        discovered_urls: List[str] = []
        for q in search_queries:
            results = scraper_service.search_web(q, max_results=settings.MAX_SEARCH_RESULTS)
            for u in results:
                if u not in discovered_urls and u not in target_urls:
                    discovered_urls.append(u)
            if len(discovered_urls) >= settings.MAX_SEARCH_RESULTS:
                break
        
        if discovered_urls:
            target_urls.extend(discovered_urls[:settings.MAX_SEARCH_RESULTS])
            scraped_docs = await scraper_service.scrape_multiple_urls(discovered_urls[:settings.MAX_SEARCH_RESULTS])

    logger.info(f"Successfully scraped {len(scraped_docs)} documents.")

    return {
        "scraped_documents": scraped_docs,
        "target_urls": target_urls,
        "status": "navigation_completed"
    }
