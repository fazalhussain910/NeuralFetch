import logging
from typing import Dict, Any, List
from app.state import AgentState, RetrievedChunk, Citation
from app.services.qdrant_service import qdrant_service
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizer Node:
    Retrieves the most relevant chunks from Qdrant, extracts the exact answer,
    and formats the final output with explicit citations and source URLs.
    """
    question = state["question"]
    indexed_count = state.get("indexed_chunks_count", 0)

    if indexed_count == 0:
        logger.warning("No chunks available in Qdrant for synthesis.")
        return {
            "retrieved_chunks": [],
            "answer": "I was unable to retrieve relevant web content to answer your query. Please check network connectivity or provide direct URLs.",
            "citations": [],
            "status": "synthesis_failed_empty"
        }

    # 1. Retrieve top-k semantic chunks from local Qdrant
    search_results = qdrant_service.search(query=question, limit=5)
    
    retrieved_chunks: List[RetrievedChunk] = []
    for r in search_results:
        retrieved_chunks.append({
            "text": r["text"],
            "url": r["url"],
            "title": r.get("title", ""),
            "score": r["score"]
        })

    # 2. Build Context String with Source URLs
    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"--- SOURCE [{idx}] ---\nURL: {c['url']}\nCONTENT:\n{c['text']}\n"
        )
    context_str = "\n".join(context_blocks)

    # 3. Construct System and User Prompts
    system_prompt = (
        "You are an expert AI Research Assistant built for the AMD AI League. "
        "Your task is to answer the user's question directly and factually using ONLY the provided web context.\n"
        "STRICT REQUIREMENTS:\n"
        "1. Give a clear, direct answer to the question first.\n"
        "2. Ground every claim in the provided context.\n"
        "3. Explicitly cite the exact source URL(s) used for each key piece of information.\n"
        "4. If the context does not contain sufficient details, state clearly what is missing."
    )

    clean_q = question.strip()
    if clean_q.startswith(("http://", "https://")) or (len(clean_q.split()) <= 2 and "." in clean_q):
        task_description = f"USER REQUEST: Analyze, summarize, and extract key details about: {clean_q}"
    else:
        task_description = f"USER QUESTION: {question}"

    user_prompt = (
        f"{task_description}\n\n"
        f"RETRIEVED WEB CONTEXT:\n{context_str}\n\n"
        "Please provide the synthesized answer followed by an exact 'Sources and Citations' section."
    )

    # 4. Invoke Local LLM with Fallback
    try:
        raw_answer = llm_service.invoke(prompt=user_prompt, system_prompt=system_prompt)
    except Exception as e:
        logger.warning(f"Local LLM invocation failed ({e}). Generating extraction from highest scoring chunk.")
        best_chunk = retrieved_chunks[0] if retrieved_chunks else None
        if best_chunk:
            raw_answer = (
                f"**Direct Answer (Extracted)**:\n{best_chunk['text']}\n\n"
                f"**Source URL**:\n{best_chunk['url']}"
            )
        else:
            raw_answer = "No relevant context found."

    # 5. Extract Unique Citations
    seen_urls = set()
    citations: List[Citation] = []
    for c in retrieved_chunks:
        url = c["url"]
        if url and url not in seen_urls:
            seen_urls.add(url)
            citations.append({
                "url": url,
                "snippet": c["text"][:200] + "...",
                "score": round(c["score"], 4)
            })

    return {
        "retrieved_chunks": retrieved_chunks,
        "answer": raw_answer,
        "citations": citations,
        "status": "synthesis_completed"
    }
