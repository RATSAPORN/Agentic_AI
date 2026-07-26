"""
Report Generator Agent
-----------------------
Expert writer and synthesizer. Uses the Data Retriever agent (wrapped as a
tool via the agent-as-tool pattern) to fetch relevant snippets from the
knowledge base, then synthesizes them into a single, cohesive,
non-redundant, well-formatted answer for the end user.

This is the entry-point agent: main.py runs Runner.run() against this
agent, and it internally calls the Data Retriever tool as needed.
"""

from agents import Agent, ModelSettings

from agents_def.data_retriever import data_retriever_agent
import os

MODEL = os.environ.get("GEMINI_MODEL")
MODEL_SETTINGS = ModelSettings(verbosity=None, reasoning=None)

INSTRUCTIONS = """\
You are an expert writer and synthesizer.

To answer the user's question, you MUST first call the \
`get_knowledge_base_snippets` tool to retrieve relevant raw information \
from the knowledge base. Do not answer from memory or general knowledge.

Once you have the snippets:
- Synthesize them into a single, cohesive, well-formatted answer.
- Remove redundancy — do not repeat the same fact twice even if it \
appears in multiple snippets.
- Base your answer strictly on the retrieved snippets. Do not invent \
information that isn't supported by them.
- If the retrieved snippets say no relevant information was found, tell \
the user clearly that the knowledge base doesn't cover their question — \
do not fabricate an answer.
- Write in clear, professional prose (use short paragraphs or bullet \
points where it improves readability).
"""

# Wrap the Data Retriever agent as a callable tool for the Report Generator.
# This is the "agent-as-tool" orchestration pattern: the Report Generator
# stays in control of the conversation and decides when to invoke retrieval.
retriever_tool = data_retriever_agent.as_tool(
    tool_name="get_knowledge_base_snippets",
    tool_description=(
        "Retrieve raw, relevant text snippets from the knowledge base for "
        "a given query. Returns unprocessed snippets, not a final answer."
    ),
)

report_generator_agent = Agent(
    name="Report Generator",
    instructions=INSTRUCTIONS,
    tools=[retriever_tool],
    model=MODEL,
    model_settings=MODEL_SETTINGS,
)