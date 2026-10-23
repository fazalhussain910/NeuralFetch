from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class ScrapedDocument(TypedDict):
    url: str
    title: str
    content: str
    length: int

class RetrievedChunk(TypedDict):
    text: str
    url: str
    title: str
    score: float

class Citation(TypedDict):
    url: str
    snippet: str
    score: float

class AgentState(TypedDict):
    # User Input
    question: str
    
    # Query Analyzer output
    search_queries: List[str]
    target_urls: List[str]
    intent_summary: str
    
    # Web Navigator output
    scraped_documents: List[ScrapedDocument]
    
    # RAG Indexer output
    indexed_chunks_count: int
    
    # Synthesizer output
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    citations: List[Citation]
    
    # Execution status / diagnostics
    status: str
    error: Optional[str]
