import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv, set_key

# Page setup
st.set_page_config(
    page_title="AI Research & Task Automation Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .stCard { background-color: #f8f9fa; border-radius: 10px; padding: 1.5rem; border: 1px solid #e9ecef; }
    .badge-success { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; }
    .badge-info { background-color: #17a2b8; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

from config import DATA_DIR, OUTPUTS_DIR, GEMINI_API_KEY, OPENAI_API_KEY
from src.llm.provider import LLMProvider
from src.agent.executor import AgentExecutor
from src.rag.ingest import DocumentIngestor
from src.rag.vector_store import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.tools.code_tool import PythonCodeExecutorTool

# Load persistent instances in Streamlit session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever(st.session_state.vector_store)
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Sidebar - Settings & Config
with st.sidebar:
    st.title("⚙️ Agent Settings")
    
    provider_choice = st.selectbox(
        "LLM Provider",
        ["gemini", "openai", "ollama"],
        index=0,
        help="Select the backend LLM engine"
    )

    api_key_input = ""
    if provider_choice == "gemini":
        api_key_input = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
    elif provider_choice == "openai":
        api_key_input = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")

    if st.button("Save API Configuration"):
        os.environ["LLM_PROVIDER"] = provider_choice
        if provider_choice == "gemini":
            os.environ["GEMINI_API_KEY"] = api_key_input
        elif provider_choice == "openai":
            os.environ["OPENAI_API_KEY"] = api_key_input
        st.success("Configuration updated!")

    st.markdown("---")
    st.markdown("### 📊 Vector Store Status")
    st.info(f"Storage Directory: `{DATA_DIR}`")
    st.info(f"Vector Database: **ChromaDB Active**")

# App Header
st.markdown('<div class="main-header">🤖 AI Research & Task Automation Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous Agentic AI, Hybrid RAG, Multi-Tool Execution Loop, and Report Synthesis</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Autonomous Research Agent",
    "📚 RAG Knowledge Base",
    "💻 Python Code Sandbox",
    "📄 Research Reports"
])

# TAB 1: Autonomous Agent
with tab1:
    st.subheader("Research Goal Execution")
    
    user_goal = st.text_area(
        "Enter your research prompt or task description:",
        height=100,
        placeholder="e.g. Analyze recent breakthroughs in AI Agent frameworks, extract key features, and generate a technical summary report."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Start Agent Execution", type="primary", use_container_width=True)
        
    if run_btn and user_goal.strip():
        llm = LLMProvider(provider=provider_choice, api_key=api_key_input)
        executor = AgentExecutor(llm_provider=llm)

        progress_bar = st.progress(0, text="Initializing Agent...")
        status_box = st.empty()
        plan_box = st.empty()
        feed_box = st.container()

        def stream_callback(event_type, data):
            if event_type == "status":
                status_box.info(f"🔄 **Agent Status**: {data}")
            elif event_type == "plan":
                progress_bar.progress(25, text="Task plan generated!")
                with plan_box.expander("📋 **Sub-Task DAG Execution Plan**", expanded=True):
                    for task in data:
                        st.markdown(f"- **Task {task['id']}**: `{task['title']}` (Tool: `{task['tool']}`)")
            elif event_type == "step":
                progress_bar.progress(60, text=f"Executed: {data['task']}")
                with feed_box.expander(f"⚙️ Step Executed: {data['task']}", expanded=False):
                    st.caption(f"Tool Used: `{data['tool']}`")
                    st.text(data["output"][:500])

        with st.spinner("Agent running autonomous loop..."):
            result = executor.run(user_goal, callback=stream_callback)
            st.session_state.last_result = result
            progress_bar.progress(100, text="Execution Complete!")
            st.success("✅ Research task completed successfully!")

    # Display Results if available
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        st.subheader("📝 Synthesized Research Report")
        
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            st.metric("Reflection Quality Score", f"{res['evaluation'].get('score', 0.9)*100:.0f}%")
        with col_eval2:
            st.caption(f"**Critique**: {res['evaluation'].get('critique', 'Passed verification')}")

        st.markdown(res["report_markdown"])
        
        if res.get("exports") and res["exports"].get("markdown_path"):
            st.download_button(
                "⬇️ Download Markdown Report",
                data=res["report_markdown"],
                file_name="research_report.md",
                mime="text/markdown"
            )

# TAB 2: RAG Knowledge Base
with tab2:
    st.subheader("Document Ingestion & Vector Indexing")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, Markdown, TXT, or CSV documents to index into RAG knowledge base:",
        type=["pdf", "md", "txt", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        ingestor = DocumentIngestor()
        total_chunks = 0
        
        for file in uploaded_files:
            file_path = DATA_DIR / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            chunks = ingestor.process_document(file_path)
            st.session_state.vector_store.add_chunks(chunks)
            total_chunks += len(chunks)
            
        st.success(f"Indexed {len(uploaded_files)} document(s) with {total_chunks} total text chunks into ChromaDB!")

    st.markdown("---")
    st.subheader("🔍 Test RAG Knowledge Search")
    rag_query = st.text_input("Query internal document knowledge base:")
    
    if rag_query:
        passages = st.session_state.retriever.retrieve(rag_query, top_k=4)
        if passages:
            st.markdown(f"**Found {len(passages)} relevant passages:**")
            for idx, p in enumerate(passages):
                with st.expander(f"Passage {idx+1} (Relevance Score: {p['score']:.3f})"):
                    st.write(p["content"])
                    st.caption(f"Source: {p['metadata'].get('source', 'Unknown')}")
        else:
            st.warning("No matching passages found.")

# TAB 3: Code Sandbox
with tab3:
    st.subheader("Python Code Analysis Sandbox")
    st.markdown("Execute Python snippets safely to compute math, process tables, or generate plots.")
    
    sample_code = """import matplotlib.pyplot as plt
import numpy as np

# Sample data plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label='Sin(x)', color='blue')
plt.title('Agent Generated Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)
plt.legend()
plt.savefig('d:/projects/ai-research-task-automation-agent/outputs/plot.png')
print("Plot saved to outputs/plot.png successfully!")
"""

    code_input = st.text_area("Python Code:", value=sample_code, height=220)
    
    if st.button("Run Code"):
        executor_tool = PythonCodeExecutorTool()
        res = executor_tool.execute(code_input)
        
        if res["status"] == "success":
            st.success("Code Executed Successfully!")
            st.code(res["output"])
            
            # Check if image generated
            plot_path = OUTPUTS_DIR / "plot.png"
            if plot_path.exists():
                st.image(str(plot_path), caption="Generated Plot")
        else:
            st.error("Execution Error:")
            st.code(res["output"])

# TAB 4: Reports Viewer
with tab4:
    st.subheader("Generated Research Reports")
    output_files = list(OUTPUTS_DIR.glob("*.*"))
    
    if output_files:
        for f in output_files:
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1:
                st.markdown(f"📄 **{f.name}** ({f.stat().st_size / 1024:.1f} KB)")
            with col_f2:
                with open(f, "rb") as file_bytes:
                    st.download_button(f"Download", data=file_bytes, file_name=f.name)
    else:
        st.info("No saved reports found yet. Run an autonomous research agent task to generate reports.")
