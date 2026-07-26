"""
Main entry point.

Runs the two-agent RAG pipeline (Report Generator -> Data Retriever, via
the agent-as-tool pattern) against a set of sample queries and prints the
final synthesized answers.

Usage:
    python main.py
    python main.py "What is the PTO policy?"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import (
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

load_dotenv()

# Tracing uploads run data to OpenAI's platform using the configured API
# key. Since we're routing through Gemini instead of a real OpenAI key,
set_tracing_disabled(True)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

gemini_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)

set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
os.environ["GEMINI_MODEL"] = GEMINI_MODEL

# Agent modules read GEMINI_MODEL from the environment at import time to
from agents_def import report_generator_agent  
# ------------------------------------------------------------------------

DEFAULT_SAMPLE_QUERIES = [
    "What is the policy on international travel?",
    "How much PTO do employees get?",
    "What should I do if I lose my work laptop?",
    "What is the company's policy on remote dinosaurs?",  # no-match test
]


async def run_query(query: str) -> None:
    print("=" * 80)
    print(f"QUERY: {query}")
    print("-" * 80)

    result = Runner.run_streamed(report_generator_agent, query)

    async for event in result.stream_events():
        # Ignore raw token-delta events; we only want higher-level status.
        if event.type != "run_item_stream_event":
            continue

        item = event.item

        if item.type == "tool_call_item":
            tool_name = getattr(item.raw_item, "name", "a tool")
            if tool_name == "get_knowledge_base_snippets":
                print("🔍 Retrieving relevant snippets from the knowledge base...")
            else:
                print(f"🔧 Calling tool: {tool_name}...")

        elif item.type == "tool_call_output_item":
            print("✅ Snippets retrieved. Synthesizing the final answer...")

        elif item.type == "message_output_item":
            print("✍️  Drafting response...")

    print("-" * 80)
    print(result.final_output)
    print("=" * 80)
    print()

    
async def main() -> None:
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = DEFAULT_SAMPLE_QUERIES

    for query in queries:
        await run_query(query)


if __name__ == "__main__":
    asyncio.run(main())