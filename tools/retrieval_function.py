"""
Retrieval Tool
--------------
A simple keyword-based search tool over a local knowledge_base.txt file.
Used by the Data Retriever agent to fetch relevant raw text snippets.

Approach:
- The knowledge base is split into paragraphs (blank-line separated).
- Each paragraph is scored against the query using keyword overlap
  (case-insensitive word matching), which is enough for the "simple
  keyword or basic semantic search" requirement without extra deps.
- The top-scoring, non-zero-score paragraphs are returned as snippets.
"""

import os
import re
from typing import List

from agents import function_tool

KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base.txt",
)


def _load_paragraphs(path: str = KNOWLEDGE_BASE_PATH) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split on blank lines, drop empties/whitespace-only chunks
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if p]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _score_paragraph(query_tokens: List[str], paragraph: str) -> int:
    paragraph_tokens = set(_tokenize(paragraph))
    return sum(1 for token in query_tokens if token in paragraph_tokens)


@function_tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search the local knowledge base for snippets relevant to a query.

    Args:
        query: The user's search query or topic to look up.
        top_k: Maximum number of relevant snippets to return (default 3).

    Returns:
        A string containing the top matching snippets, separated by
        '---', or a message indicating no relevant snippets were found.
    """
    paragraphs = _load_paragraphs()
    query_tokens = _tokenize(query)

    scored = [
        (paragraph, _score_paragraph(query_tokens, paragraph))
        for paragraph in paragraphs
    ]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)

    top_matches = [paragraph for paragraph, _ in scored[:top_k]]

    if not top_matches:
        return "No relevant snippets found in the knowledge base."

    return "\n---\n".join(top_matches)