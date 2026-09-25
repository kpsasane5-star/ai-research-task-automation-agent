# AI Research & Task Automation Agent 🤖

An enterprise-grade, autonomous **AI Research & Task Automation Agent** built with **Python 3.12**, **LLMs** (Gemini, OpenAI, Ollama), **Agentic AI** task planning loops (ReAct + DAG Goal Planner), **Hybrid Retrieval-Augmented Generation (RAG)**, and an interactive **Streamlit Dashboard** + **Rich CLI**.

---

## 🌟 Key Features

- 🧠 **Agentic Task Decomposition & ReAct Execution**:
  - Automatically breaks complex research queries into structured DAG sub-tasks.
  - Multi-step tool selection loop (Thought $\rightarrow$ Action $\rightarrow$ Observation $\rightarrow$ Synthesis).
  - Self-Reflection module for quality verification & critique.

- 📚 **Hybrid Retrieval-Augmented Generation (RAG)**:
  - Supports **PDF**, **Markdown**, **TXT**, and **CSV** document ingestion.
  - **ChromaDB** persistent vector database with local sentence embeddings.
  - **Hybrid Search**: Combines dense vector semantic similarity with BM25 sparse keyword scoring.

- 🛠️ **Extensible Tool Suite**:
  - **Web Search**: Real-time web intelligence via DuckDuckGo / scraper fallback.
  - **RAG Knowledge Retriever**: Extracts contextual passages from uploaded documents.
  - **Python Code Execution Sandbox**: Executes data analysis scripts, calculates math, and generates `matplotlib` plots.
  - **Report Exporter**: Exports markdown research reports and PDF documents (`fpdf2`).

- 🖥️ **Dual User Interfaces**:
  - **Streamlit Web Dashboard**: Interactive UI with live step progress, document management, code sandbox, and report download buttons.
  - **Rich Terminal CLI**: Command-line interface for automated batch tasks and server execution.

---

## 🏗️ Architecture Overview

```
d:\projects\ai-research-task-automation-agent\
├── app.py                      # Streamlit Web Dashboard
├── cli.py                      # Rich Terminal CLI Interface
├── config.py                   # Central Application & Environment Settings
├── requirements.txt            # Python Dependencies
├── README.md                   # System Documentation
├── src/
│   ├── agent/                  # Agent Core (Planner, ReAct Executor, Memory, Self-Reflection)
│   ├── rag/                    # RAG Engine (Document Ingestor, Vector Store, Hybrid Retriever)
│   ├── tools/                  # Extensible Tool Suite (Web Search, RAG Query, Python Code, File Export)
│   ├── llm/                    # Unified Multi-Provider LLM Client (Gemini, OpenAI, Ollama, Fallback)
│   └── utils/                  # Structured Logger & Report PDF Exporter
├── data/                       # Storage for Vector DB & Uploaded Documents
└── outputs/                    # Output Reports & Generated Charts
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
cd d:\projects\ai-research-task-automation-agent
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file (or copy `.env.example`):

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
```

*(Note: The agent contains an intelligent fallback runner, so it can run and execute tool loops even before setting up API keys!)*

---

## 💻 Running the Application

### Option A: Launch Streamlit Web Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Option B: Run via Terminal CLI

Execute an autonomous research task:

```bash
python cli.py --goal "Analyze recent breakthroughs in AI Agent frameworks and document RAG architectures"
```

Ingest a PDF or document into the RAG Vector Database:

```bash
python cli.py --ingest "path/to/document.pdf"
```

---

## 🧪 Automated Verification Suite

To run end-to-end tests across the LLM Provider, RAG Ingestor, Tool Execution, and Agentic Loop:

```bash
python -m unittest tests/test_agent.py
```

---

## 📄 License

MIT License. Developed for Advanced AI Research and Autonomous Task Automation.
