"""
Test script for verifying Groq API connectivity and performance in NeuralFetch.

Usage:
    # 1. Using environment variable or .env:
    # Set GROQ_API_KEY in .env or:
    # $env:GROQ_API_KEY="gsk_..."
    python test_groq.py

    # 2. Passing API key directly via CLI:
    python test_groq.py --api-key gsk_your_key_here

    # 3. Test with a custom model:
    python test_groq.py --model llama-3.1-8b-instant

    # 4. Run an end-to-end test with the full NeuralFetch agent workflow:
    python test_groq.py --full-agent --query "What is AMD ROCm?"
"""

import sys
import os
import argparse
import asyncio
import time

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.config import settings
from app.services.llm_service import LocalLLMService

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Test Groq API integration in NeuralFetch")
    parser.add_argument("--api-key", type=str, default=None, help="Groq API key (defaults to GROQ_API_KEY from .env / env vars)")
    parser.add_argument("--model", type=str, default=None, help="Groq model (default: llama-3.3-70b-versatile or llama-3.1-8b-instant)")
    parser.add_argument("--prompt", type=str, default="What are three key strengths of AMD GPUs for AI acceleration? Answer in 3 concise bullet points.", help="Test prompt")
    parser.add_argument("--full-agent", action="store_true", help="Also run the full NeuralFetch LangGraph agent workflow using Groq")
    parser.add_argument("--query", type=str, default="What are the latest architecture features of AMD ROCm?", help="Query for full agent test")

    args = parser.parse_args()

    api_key = args.api_key or settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    model = args.model or settings.GROQ_MODEL or "llama-3.3-70b-versatile"

    print("=" * 65)
    print("       NeuralFetch - Groq API Integration Test")
    print("=" * 65)
    print(f"Target Model:  {model}")
    print(f"API Key:       {'[Provided via CLI]' if args.api_key else ('[Found in Config/Env: ' + api_key[:7] + '...]' if api_key else '[MISSING]')}")
    print("-" * 65)

    if not api_key:
        print("\n[!] WARNING: No GROQ_API_KEY detected.")
        print("To test Groq API:")
        print("  1. Get a free API key at: https://console.groq.com/keys")
        print("  2. Run:")
        print("       python test_groq.py --api-key gsk_your_key_here")
        print("     OR add to .env:")
        print("       GROQ_API_KEY=gsk_your_key_here")
        print("       LLM_PROVIDER=groq")
        print("\nExiting diagnostic without calling remote API.")
        return 1

    # Initialize service with Groq configuration
    llm = LocalLLMService()
    llm.configure(provider="groq", model_name=model, api_key=api_key)

    # Test 1: Direct text invocation
    print("\n[Test 1/3] Testing standard prompt generation (llm.invoke)...")
    print(f"Prompt: \"{args.prompt}\"")
    t0 = time.time()
    try:
        reply = llm.invoke(args.prompt)
        elapsed = round(time.time() - t0, 2)
        print(f"\nResponse (received in {elapsed}s):\n")
        print(reply)
        print("\n[PASSED] Test 1 completed successfully.")
    except Exception as e:
        print(f"\n[FAILED] Test 1 failed: {e}")
        return 1

    # Test 2: Structured JSON extraction (used by Query Analyzer node)
    print("\n" + "-" * 65)
    print("[Test 2/3] Testing JSON extraction (llm.invoke_json)...")
    json_prompt = (
        "User Question: What are the best open source LLMs for AMD ROCm?\n\n"
        "Provide a JSON response with keys 'search_queries' (list of strings) and 'intent' (string)."
    )
    t0 = time.time()
    try:
        json_res = llm.invoke_json(json_prompt)
        elapsed = round(time.time() - t0, 2)
        print(f"\nExtracted JSON (received in {elapsed}s):")
        print(json_res)
        assert isinstance(json_res, dict), "Result should be a dictionary"
        print("\n[PASSED] Test 2 completed successfully.")
    except Exception as e:
        print(f"\n[FAILED] Test 2 failed: {e}")
        return 1

    # Test 3: Optional Full Agent workflow
    if args.full_agent:
        print("\n" + "-" * 65)
        print("[Test 3/3] Running full LangGraph agent workflow with Groq...")
        from app.services.llm_service import llm_service
        # Configure global singleton used by nodes
        llm_service.configure(provider="groq", model_name=model, api_key=api_key)
        
        from app.agent import run_agent
        t0 = time.time()
        try:
            print(f"Running agent on query: '{args.query}'")
            state = asyncio.run(run_agent(args.query))
            elapsed = round(time.time() - t0, 2)
            print(f"\nWorkflow finished in {elapsed}s with status: {state.get('status')}")
            print("\nSynthesized Answer:\n")
            print(state.get("answer"))
            print("\nCitations:")
            for cit in state.get("citations", []):
                print(f" - [{cit.get('score')}] {cit.get('url')}")
            print("\n[PASSED] Full agent test completed successfully.")
        except Exception as e:
            print(f"\n[FAILED] Full agent test failed: {e}")
            return 1
    else:
        print("\n" + "-" * 65)
        print("[Tip] To test the entire end-to-end NeuralFetch agent pipeline with Groq, run:")
        print(f"  python test_groq.py --full-agent --query \"{args.query}\"")

    print("\n" + "=" * 65)
    print("All tests passed! Groq is fully functional in NeuralFetch.")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(main())
