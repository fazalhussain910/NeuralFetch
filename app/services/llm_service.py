import os
import logging
import json
import re
from typing import Optional, Any, Dict, List
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

class LocalLLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model_name = self._resolve_model_name()
        self._llm = None
        self._init_client()

    def _resolve_model_name(self) -> str:
        """Determines active model name according to selected provider."""
        if self.provider == "groq":
            if settings.GROQ_MODEL:
                return settings.GROQ_MODEL
            if settings.LLM_MODEL and "llama3.2" not in settings.LLM_MODEL:
                return settings.LLM_MODEL
            return "qwen/qwen3.8-27b"
        return settings.LLM_MODEL

    def configure(self, provider: str, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """Allows dynamically switching provider, model, or API key (useful for tests and benchmarking)."""
        self.provider = provider
        if api_key:
            settings.GROQ_API_KEY = api_key
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self._resolve_model_name()
        self._llm = None
        self._init_client()

    def _init_client(self):
        """Attempts to initialize client with multiple provider fallbacks."""
        if self.provider == "ollama":
            # 1. Try langchain_ollama
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=0.1
                )
                logger.info(f"Initialized ChatOllama with model '{self.model_name}'")
                return
            except ImportError:
                logger.debug("langchain_ollama not installed, trying langchain_openai...")

            # 2. Try langchain_openai pointing to Ollama's local OpenAI-compatible endpoint
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.model_name,
                    base_url=f"{self.base_url}/v1",
                    api_key="ollama",
                    temperature=0.1,
                    timeout=3.0,
                    max_retries=1
                )
                logger.info(f"Initialized ChatOpenAI bridge to Ollama at {self.base_url}/v1")
                return
            except Exception as e:
                logger.debug(f"ChatOpenAI bridge initialization failed: {e}")

            # 3. Direct HTTP fallback will be used if both wrappers are unavailable
            logger.info("Using direct HTTP REST client for local Ollama daemon.")

        elif self.provider == "huggingface":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                from langchain_huggingface import HuggingFacePipeline
                import torch
                
                device = settings.get_torch_device()
                logger.info(f"Loading HuggingFace model {self.model_name} on device: {device}")
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.1
                )
                self._llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.warning(f"Could not initialize local HuggingFace LLM: {e}")
                self._llm = None

        elif self.provider == "groq":
            api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY is not set. Groq calls will fail until an API key is provided.")
                self._llm = None
                return

            # 1. Try langchain_groq
            try:
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(
                    model=self.model_name,
                    groq_api_key=api_key,
                    temperature=0.1,
                    max_tokens=settings.GROQ_MAX_TOKENS
                )
                logger.info(f"Initialized ChatGroq with model '{self.model_name}' for testing")
                return
            except Exception as e:
                logger.debug(f"ChatGroq initialization failed: {e}. Will use direct Groq client fallback.")
                self._llm = None

    def _invoke_direct_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Direct HTTP invocation to Ollama REST endpoint if LangChain packages are missing."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
            raise RuntimeError(f"Ollama API returned HTTP {resp.status_code}: {resp.text}")

    def _invoke_direct_groq(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Direct invocation to Groq REST endpoint or SDK if LangChain wrapper fails."""
        api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Please set GROQ_API_KEY in your .env or environment to use Groq."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1. Try Groq SDK
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=settings.GROQ_MAX_TOKENS
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Groq SDK invocation error: {e}. Trying direct HTTP fallback...")

        # 2. Direct HTTP REST fallback
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": settings.GROQ_MAX_TOKENS
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            raise RuntimeError(f"Groq API returned HTTP {resp.status_code}: {resp.text}")

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invokes the LLM with the provided prompt and optional system message."""
        # Method 1: LangChain Runnable
        if self._llm is not None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                response = self._llm.invoke(messages)
                if hasattr(response, "content"):
                    return str(response.content).strip()
                return str(response).strip()
            except Exception as e:
                logger.warning(f"LangChain LLM invocation failed ({e}), falling back to direct provider call...")

        # Method 2: Direct HTTP to Ollama
        if self.provider == "ollama":
            try:
                return self._invoke_direct_ollama(prompt, system_prompt)
            except Exception as e:
                logger.error(f"Direct Ollama call failed: {e}")
                raise RuntimeError(
                    f"Local Ollama is not responding at {self.base_url}. "
                    f"Please ensure 'ollama serve' is running and model '{self.model_name}' is pulled."
                ) from e

        # Method 3: Direct Groq API
        if self.provider == "groq":
            try:
                return self._invoke_direct_groq(prompt, system_prompt)
            except Exception as e:
                logger.error(f"Direct Groq call failed: {e}")
                raise RuntimeError(f"Groq API call failed: {e}") from e

        raise RuntimeError(f"LLM service is not operational for provider: {self.provider}.")

    def invoke_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Invokes the local LLM and guarantees an extracted JSON response with heuristic fallback."""
        full_system = (
            (system_prompt or "") + 
            "\n\nCRITICAL: Respond ONLY with a valid JSON object. Do not include markdown code fences or conversational filler."
        )
        try:
            raw_output = self.invoke(prompt, full_system)
            
            # Clean possible markdown formatting
            cleaned = re.sub(r"^```(?:json)?", "", raw_output.strip())
            cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

            json_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(cleaned)
        except Exception as e:
            logger.warning(f"LLM JSON parsing fallback triggered: {e}")
            # Fallback heuristic extraction
            words = [w for w in re.findall(r"\b\w{3,}\b", prompt) if w.lower() not in {"user", "question", "provide", "search", "strategy", "json"}]
            fallback_query = " ".join(words[:4]) if words else prompt[:50]
            return {
                "search_queries": [fallback_query],
                "intent": "Information retrieval"
            }

# Singleton instance
llm_service = LocalLLMService()
