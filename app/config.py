import os
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM Provider: "ollama", "huggingface", or "groq" (for testing)
    LLM_PROVIDER: Literal["ollama", "huggingface", "groq"] = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:latest"
    
    # Groq API Configuration (for testing / development)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_MAX_TOKENS: int = 512
    
    # Local Embeddings (HuggingFace sentence-transformers)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Local Qdrant Storage
    QDRANT_STORAGE_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION: str = "web_knowledge_league"
    
    # Scraping & Retrieval Parameters
    MAX_SEARCH_RESULTS: int = 4
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    PLAYWRIGHT_HEADLESS: bool = True
    USE_DYNAMIC_SCRAPING: bool = False
    
    # Hardware Acceleration (ROCm / CUDA / CPU)
    DEVICE: str = "auto"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_torch_device(self) -> str:
        """Resolves target computing device, auto-detecting AMD ROCm (exposed via PyTorch CUDA interface) or CPU."""
        if self.DEVICE != "auto":
            return self.DEVICE
        try:
            import importlib
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                # AMD ROCm presents itself as CUDA in PyTorch (e.g. 'AMD Radeon Graphics')
                return "cuda"
        except Exception:
            pass
        return "cpu"

settings = Settings()
