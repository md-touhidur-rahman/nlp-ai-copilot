# NLP AI Copilot  
**Evidence-Grounded RAG, Data Reasoning, and Agent Workflows**

An AI Copilot built in **Python 3.9** using **Streamlit** that answers questions about **documents and datasets** in a strictly grounded way.

The system either:
- **answers with explicit evidence** (citations or computed values), or  
- **refuses** when the required evidence does not exist.

This project is designed to demonstrate how large language models can act as **interfaces to reliable systems**, not as free-form guessers.

---

## Core Capabilities

### 1. Document Question Answering (RAG)
- Ingests PDF / text documents
- Chunks content with metadata
- Embeds and indexes chunks (FAISS)
- Retrieves relevant evidence for a query
- Generates answers **only from retrieved chunks**
- Refuses to answer when evidence is missing
- UI displays retrieved chunks as an evidence panel

### 2. Data Reasoning (Language → Computation)
- Loads a default CSV dataset or **any user-uploaded CSV**
- Detects schema and numeric columns automatically
- Parses natural-language questions into structured queries
- Executes **real pandas computations** (not LLM guessing)
- Produces tables and plots grounded in computed results

### 3. Unified Router
- Single chat interface for both documents and data
- Rule-based routing (DATA vs RAG)
- Manual override available in the UI for debugging and demos

### 4. Agent Workflows (LangGraph)
- Central agent entrypoint orchestrates execution
- Deterministic control flow
- State captures:
  - route decision
  - tool execution
  - evidence
  - answer
- Designed to be inspectable and extensible

### 5. Observability via Traces
- Every execution produces a JSON trace
- Stored under `data/traces/`
- Streamlit Trace Viewer page lets you inspect:
  - question
  - route and mode
  - node timings
  - router metadata
  - evidence summary

---


## Grounding Policy

This project enforces a strict grounding contract:

- **RAG mode**
  - Answers must cite retrieved document chunks
  - If no sufficient evidence is found, the system refuses

- **DATA mode**
  - Answers must reference computed values
  - Tables and plots are generated from real data
  - No speculative or fabricated numbers

If the system cannot justify an answer with evidence, it will say **“I don’t know.”**

---

## Project Structure (Key Parts)

src/
agent/
router.py # rule-based routing
tools.py # data + RAG execution tools
graph.py # agent orchestration & tracing
app/
streamlit_app.py # Streamlit entrypoint
pages/
chat.py # unified chat UI
data.py # dataset exploration
traces.py # trace viewer
data/
loader/, filter/, aggregate/, nl_parser.py

Copy code
data/
datasets/tabular/ # default CSV datasets (not committed)
traces/ # execution traces (not committed)

yaml
Copy code

---

## Setup Instructions

### 1. Environment
```bash
cd ~/Projects/nlp-ai-copilot
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2. Run the App
Always run from the project root:

bash
Copy code
PYTHONPATH=. streamlit run src/app/streamlit_app.py
Default Dataset
The project expects a default dataset at:

bash
Copy code
data/datasets/tabular/store_sales_train.csv
You can alternatively upload any CSV via the sidebar, and DATA queries will operate on the uploaded file instead.

Example Queries
Data Questions (Store Sales)
Show daily sales for store 1 family Grocery I in jan 2013

Sales in march to june in 2013

Which month had maximum sales in 2016

Compare sales by family in 2014

Data Questions (Uploaded CSV)
max Log2_FC by Gene

average Adj_P_Value

top 10 rows by P_Value

Document Questions (RAG)
According to the document, what does X mean?

What does the PDF say about Y? Cite sources.

Evaluation
Unit and regression tests:

bash
Copy code
PYTHONPATH=. pytest -q
Optional evaluation script:

bash
Copy code
PYTHONPATH=. python eval/run_eval.py
Known Limitations
Mixed reasoning (docs + data in one answer) is scaffolded but not executed end-to-end

The router detects MIXED queries and the UI explains the limitation clearly

Uploaded CSV handling is heuristic

For unusual schemas, more explicit queries are required

These are deliberate design choices prioritizing determinism and transparency.

Roadmap
Mixed reasoning workflow (retrieve document facts → compute data window → compare)

Evaluation results UI in Streamlit

Chat → Trace deep-linking

More advanced schema selection for arbitrary datasets

Design Philosophy
This project focuses on one principle:

LLMs should interface with reliable systems, not replace them.

Every answer must be supported by either:

retrieved document evidence, or

computed data results.

License
MIT