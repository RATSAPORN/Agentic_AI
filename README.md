# Agentic RAG Test — OpenAI Agents SDK

A two-agent Retrieval-Augmented Generation (RAG) system built with the
**OpenAI Agents SDK**, using the **agent-as-tool** orchestration pattern.

## Architecture

```
User Query
    │
    ▼
Report Generator Agent  ──calls──►  Data Retriever Agent (as a tool)
    │                                       │
    │                                       ▼
    │                              search_knowledge_base(query)
    │                                       │
    │                                       ▼
    │                              knowledge_base.txt (raw snippets)
    │◄──────────────────────────── raw snippets returned
    ▼
Final synthesized answer
```

- **Data Retriever Agent** (`agents_def/data_retriever.py`) — only retrieves.
  It calls a custom tool (`tools/retrieval.py`) that performs keyword search
  over `knowledge_base.txt` and returns raw, relevant snippets. It never
  answers the question directly.
- **Report Generator Agent** (`agents_def/report_generator.py`) — the
  entry-point agent. It calls the Data Retriever (wrapped via
  `.as_tool(...)`) to fetch snippets, then synthesizes them into a single,
  non-redundant, well-formatted answer.

## Project Structure

```
ai-engineer-test/
├── .env                      # GEMINI_API_KEY (gitignored)
├── .gitignore
├── requirements.txt
├── README.md
├── knowledge_base.txt        # sample company policy knowledge base
├── main.py                   # entry point — runs sample queries
├── tools/
│   ├── __init__.py
│   └── retrieval.py          # custom keyword search tool
├── agents_def/
│   ├── __init__.py
│   ├── data_retriever.py     # Data Retriever agent
│   └── report_generator.py   # Report Generator agent
├── screenshots/               # output screenshots for submission
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy your GEMINI API key into `.env`:

```
GEMINI_API_KEY=...
```

## Running

Run the default sample queries:

```bash
python main.py
```

Run a single ad-hoc query:

```bash
python main.py "What is the PTO policy?"
```

## Sample Queries Used for Evaluation

- "What is the policy on international travel?"
- "How much PTO do employees get?"
- "What should I do if I lose my work laptop?"
- "What is the company's policy on remote dinosaurs?" (no-match / hallucination guard test)

## Design Notes

- **Orchestration pattern:** agent-as-tool. The Report Generator stays in
  control of the run and calls the Data Retriever as a tool
  (`get_knowledge_base_snippets`), rather than handing off control entirely —
  this fits the strictly sequential retrieve-then-synthesize workflow.
- **Retrieval method:** simple keyword overlap scoring over paragraph-level
  chunks of `knowledge_base.txt`, implemented with the standard library
  (no external vector DB or embedding model required).
- **Hallucination guard:** the Report Generator's instructions explicitly
  require it to say when the knowledge base doesn't cover a question rather
  than fabricate an answer — demonstrated by the "remote dinosaurs" query.
