NLP AI Copilot

A grounded AI system for document retrieval and data reasoning

Overview

This project implements a unified AI copilot that answers user questions by deterministically routing them to one of two grounded pipelines:

Document reasoning (RAG) using retrieval over indexed PDFs

Data reasoning using real computation over structured tabular data

The system is explicitly designed to avoid hallucinations:

Document answers must reference retrieved evidence

Data answers must be derived from actual computations and visualizations

The focus of this project is system design, grounding, and debuggability, not conversational fluency.

Key Capabilities

Single unified chat interface

Deterministic, explainable routing (rule-based)

Retrieval-augmented generation with evidence inspection

Real data analytics (filtering, aggregation, plotting)

Explicit grounding contracts per execution path

Routing override for debugging and demonstration

Clear extension path toward agent graphs and tracing


Architecture:

User Question
      |
      v
Rule-Based Router
      |
      +---------------------------+
      |                           |
      v                           v
Data Pipeline                RAG Pipeline
- Parse query                - Embed + retrieve
- Filter dataset             - Top-k FAISS search
- Aggregate metrics           - Local LLM (Ollama)
- Generate plots              - Evidence-based answer
      |                           |
      +-------------+-------------+
                    |
              Final Answer
           (with evidence)

Project Structure:
nlp-ai-copilot/
├── src/
│   ├── agent/
│   │   ├── router.py          # Rule-based routing logic
│   │   └── rag_answerer.py    # RAG execution
│   ├── data/
│   │   ├── loader.py
│   │   ├── filtering.py
│   │   ├── aggregate.py
│   │   └── nl_parser.py
│   ├── app/
│   │   └── pages/
│   │       └── chat.py        # Unified chat UI
│
├── data/                      # Local only (ignored by git)
│   └── .gitkeep
├── tests/
│   └── test_router.py
├── README.md
└── .gitignore

Running the Application
1. Environment setup

git clone https://github.com/md-touhidur-rahman/nlp-ai-copilot.git
cd nlp-ai-copilot
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Local dependencies

Python 3.9

Streamlit

Ollama running locally

Dataset placed at: data/datasets/tabular/store_sales_train.csv

3. Launch
PYTHONPATH=. streamlit run src/app/streamlit_app.py

Example Queries
Data reasoning (computed)

Which month sales maximum in 2016

Sales from January to April in 2013

Compare sales by family in 2014

Show daily sales for store 1 in Jan 2013

These queries result in:

Parsed structured query

Aggregated table

Time-series or categorical plot

Explicit numerical results (e.g., peak month)

Document reasoning (RAG)

According to the document, what is Industry 4.0?

Cite evidence about automation from the PDF

What does the report say about supply chain risks?

These queries result in:

Generated answer

Retrieved document chunks

Similarity scores

Explicit refusal if evidence is insufficient

Grounding Policy

The system enforces different grounding guarantees depending on execution mode:

Mode	        Guarantee
DATA	        Results are produced only via real computation
RAG	            Answers must reference retrieved document chunks
GENERAL	        Used only when routing confidence is low
MIXED	        Reserved for future phases

At no point does the system infer facts without evidence or computation.

Known Limitations

Routing is rule-based (no learned routing yet)

RAG operates on local documents only

Data metrics currently focus on sales-related measures

No persistent trace or execution viewer yet

No automated evaluation harness yet

These limitations are intentional and addressed in the roadmap.











Design Rationale

This project prioritizes:

Determinism over cleverness

Explainability over opacity

Grounded outputs over fluent text

System boundaries over monolithic prompting

It is intended as a foundation for production-grade AI tooling rather than a conversational demo.

License : MIT