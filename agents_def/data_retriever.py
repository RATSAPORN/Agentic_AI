"""
Data Retriever Agent
---------------------
Specializes in retrieving specific information from the local knowledge base
(knowledge_base.txt). It does NOT answer questions directly — it only
returns raw, relevant text snippets for the Report Generator agent to use.
"""

from agents import Agent, ModelSettings

from tools import search_knowledge_base
import os

MODEL = os.environ.get("GEMINI_MODEL")
MODEL_SETTINGS = ModelSettings(verbosity=None, reasoning=None)


INSTRUCTIONS = """\
You are an expert information retrieval agent.

Your ONLY job is to search the knowledge base for text snippets relevant to \
the user's request, using the `search_knowledge_base` tool.

Rules:
- Always call the `search_knowledge_base` tool with a concise query derived \
from the user's request.
- Do NOT answer the user's question yourself.
- Do NOT summarize, interpret, or rephrase the retrieved content.
- Return the raw, relevant snippets exactly as found, clearly separated.
- If no relevant snippets are found, say so explicitly instead of guessing.
"""

data_retriever_agent = Agent(
    name="Data Retriever",
    instructions=INSTRUCTIONS,
    tools=[search_knowledge_base],
    model=MODEL,
    model_settings=MODEL_SETTINGS,
)