import os
import time
import logging
import importlib
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.agent import run_agent
from app.services.scraper_service import scraper_service
from app.services.qdrant_service import qdrant_service
from app.services.llm_service import llm_service, LocalLLMService

logger = logging.getLogger("NeuralFetch.API")

def get_device_info():
    """Detects if PyTorch with AMD ROCm/CUDA acceleration is present, falling back gracefully to CPU."""
    try:
        torch = importlib.import_module("torch")
        rocm_available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if rocm_available else "CPU (Fallback)"
        return rocm_available, device_name
    except Exception:
        return False, "CPU (PyTorch not installed)"

app = FastAPI(
    title="NeuralFetch - AMD AI League Agent",
    description="Intelligent Web Information Retrieval Agent running on local open-source models with ROCm/AMD acceleration.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to research")
    target_urls: Optional[List[str]] = Field(default=None, description="Optional explicit URLs to target")

class CitationModel(BaseModel):
    url: str
    snippet: str
    score: float

class RetrievedChunkModel(BaseModel):
    text: str
    url: str
    title: str
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationModel]
    retrieved_chunks: List[RetrievedChunkModel]
    scraped_urls: List[str]
    indexed_chunks_count: int
    execution_time_seconds: float
    device_used: str

class DirectIndexRequest(BaseModel):
    url: str = Field(..., description="Target URL to scrape and store into local Qdrant")

class DirectIndexResponse(BaseModel):
    url: str
    title: str
    chunks_indexed: int
    message: str

class LLMTestRequest(BaseModel):
    prompt: Optional[str] = Field(default="Hello! Please state your model name and summarize your function in one sentence.", description="Prompt to test LLM generation")
    provider: Optional[str] = Field(default=None, description="Optional override provider: 'groq', 'ollama', or 'huggingface'")
    model: Optional[str] = Field(default=None, description="Optional model override")
    groq_api_key: Optional[str] = Field(default=None, description="Optional Groq API key for testing")

class LLMTestResponse(BaseModel):
    provider: str
    model: str
    prompt: str
    response: str
    success: bool
    execution_time_seconds: float

@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Returns local system health, Qdrant status, Groq availability, and AMD ROCm device status."""
    rocm_available, device_name = get_device_info()
    groq_configured = bool(settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY"))

    return {
        "status": "healthy",
        "agent": "NeuralFetch Web Information Retrieval Agent",
        "llm_provider": llm_service.provider,
        "llm_model": llm_service.model_name,
        "groq_configured": groq_configured,
        "embedding_model": settings.EMBEDDING_MODEL,
        "device": device_name,
        "rocm_cuda_accelerated": rocm_available,
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "qdrant_storage": settings.QDRANT_STORAGE_PATH
    }

@app.post("/api/v1/test-llm", response_model=LLMTestResponse, tags=["Testing & Diagnostics"])
async def test_llm_endpoint(request: LLMTestRequest):
    """
    Diagnostic endpoint to test LLM connectivity (especially Groq API for test mode)
    without executing the full web retrieval / scraping pipeline.
    """
    start_t = time.time()
    
    # If custom provider or API key passed, spin up a configured instance for this test
    if request.provider or request.model or request.groq_api_key:
        test_provider = request.provider or settings.LLM_PROVIDER
        tester = LocalLLMService()
        tester.configure(
            provider=test_provider,
            model_name=request.model,
            api_key=request.groq_api_key
        )
    else:
        tester = llm_service

    try:
        reply = tester.invoke(request.prompt)
        return LLMTestResponse(
            provider=tester.provider,
            model=tester.model_name,
            prompt=request.prompt,
            response=reply,
            success=True,
            execution_time_seconds=round(time.time() - start_t, 2)
        )
    except Exception as e:
        logger.error(f"Test LLM call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM test failed ({tester.provider} / {tester.model_name}): {str(e)}"
        )

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Retrieval Agent"])
async def handle_agent_query(request: QueryRequest):
    """
    Executes the full LangGraph agent workflow:
    Query Analyzer -> Web Navigator -> Local RAG Indexer (Qdrant) -> Synthesizer.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    start_time = time.time()
    
    # If the user supplied explicit URLs, prepend them to prompt context
    full_query = request.question
    if request.target_urls:
        urls_joined = " ".join(request.target_urls)
        full_query = f"{request.question} (Refer directly to: {urls_joined})"

    try:
        final_state = await run_agent(full_query)
    except Exception as e:
        logger.error(f"Agent execution encountered an unhandled error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow error: {str(e)}"
        )

    elapsed = round(time.time() - start_time, 2)
    _, device_name = get_device_info()

    scraped_urls = [doc["url"] for doc in final_state.get("scraped_documents", [])]

    return QueryResponse(
        question=request.question,
        answer=final_state.get("answer", "No answer generated."),
        citations=final_state.get("citations", []),
        retrieved_chunks=final_state.get("retrieved_chunks", []),
        scraped_urls=scraped_urls,
        indexed_chunks_count=final_state.get("indexed_chunks_count", 0),
        execution_time_seconds=elapsed,
        device_used=device_name
    )

@app.post("/api/v1/index-url", response_model=DirectIndexResponse, tags=["RAG Ingestion"])
async def index_url_directly(request: DirectIndexRequest):
    """Scrapes a specific URL and ingests its chunks directly into the local Qdrant collection."""
    doc = await scraper_service.scrape_url(request.url)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to scrape content from provided URL: {request.url}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    texts = splitter.split_text(doc["content"])
    
    chunks = [{"text": t, "url": doc["url"], "title": doc["title"]} for t in texts]
    count = qdrant_service.index_chunks(chunks)

    return DirectIndexResponse(
        url=request.url,
        title=doc["title"],
        chunks_indexed=count,
        message=f"Successfully indexed {count} chunks into local Qdrant."
    )

# Mount static frontend directory if present for unified local serving
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)
