import logging
from typing import Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.state import AgentState
from app.services.qdrant_service import qdrant_service
from app.config import settings

logger = logging.getLogger(__name__)

def rag_indexer_node(state: AgentState) -> Dict[str, Any]:
    """
    Local RAG Pipeline Node:
    Chunks the scraped data, creates embeddings using local Hugging Face model,
    and stores them in local Qdrant vector database.
    """
    scraped_docs = state.get("scraped_documents", [])
    if not scraped_docs:
        logger.warning("No scraped documents found to index.")
        return {
            "indexed_chunks_count": 0,
            "status": "indexing_skipped"
        }

    # Reset collection for a clean search context for this request
    qdrant_service.reset_collection()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks_to_index: List[Dict[str, Any]] = []

    for doc in scraped_docs:
        text = doc.get("content", "")
        url = doc.get("url", "")
        title = doc.get("title", "")
        
        split_texts = text_splitter.split_text(text)
        for chunk_text in split_texts:
            if len(chunk_text.strip()) > 40:
                chunks_to_index.append({
                    "text": chunk_text,
                    "url": url,
                    "title": title
                })

    logger.info(f"Generated {len(chunks_to_index)} text chunks across {len(scraped_docs)} documents.")
    indexed_count = qdrant_service.index_chunks(chunks_to_index)

    return {
        "indexed_chunks_count": indexed_count,
        "status": "indexing_completed"
    }
