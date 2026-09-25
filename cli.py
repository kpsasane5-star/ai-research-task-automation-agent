import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from config import DEFAULT_LLM_PROVIDER
from src.llm.provider import LLMProvider
from src.agent.executor import AgentExecutor
from src.rag.ingest import DocumentIngestor
from src.rag.vector_store import VectorStoreManager

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console()

def main():
    parser = argparse.ArgumentParser(description="AI Research & Task Automation Agent CLI")
    parser.add_argument("--goal", "-g", type=str, help="Research goal or task description to execute")
    parser.add_argument("--ingest", "-i", type=str, help="Path to document file (PDF, TXT, MD) to ingest into RAG")
    parser.add_argument("--provider", "-p", type=str, default=DEFAULT_LLM_PROVIDER, help="LLM provider: gemini, openai, ollama")

    args = parser.parse_args()

    console.print(Panel.fit("[bold blue]🤖 AI Research & Task Automation Agent CLI[/bold blue]", border_style="blue"))

    # Document Ingestion
    if args.ingest:
        console.print(f"[yellow]Ingesting document into RAG database:[/yellow] {args.ingest}")
        try:
            ingestor = DocumentIngestor()
            chunks = ingestor.process_document(args.ingest)
            vector_store = VectorStoreManager()
            vector_store.add_chunks(chunks)
            console.print(f"[bold green]Successfully indexed {len(chunks)} text chunks into ChromaDB![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to ingest document:[/bold red] {e}")

    # Goal Execution
    if args.goal:
        console.print(f"\n[bold green]Goal:[/bold green] {args.goal}")
        
        llm = LLMProvider(provider=args.provider)
        executor = AgentExecutor(llm_provider=llm)

        def cli_callback(event_type, data):
            if event_type == "status":
                console.print(f"[dim font red]► {data}[/dim font red]")
            elif event_type == "plan":
                console.print("\n[bold cyan]📋 Sub-Task Plan:[/bold cyan]")
                for t in data:
                    console.print(f"  • Task {t['id']}: [bold]{t['title']}[/bold] (Tool: [yellow]{t['tool']}[/yellow])")
                console.print("")

        with console.status("[bold green]Executing agent sub-tasks...[/bold green]"):
            result = executor.run(args.goal, callback=cli_callback)

        console.print("\n" + "="*80)
        console.print(Panel("[bold green]Synthesized Research Report[/bold green]"))
        console.print(Markdown(result["report_markdown"]))
        console.print("="*80)
        
        if result.get("exports") and result["exports"].get("markdown_path"):
            console.print(f"\n[bold yellow]Report saved to:[/bold yellow] {result['exports']['markdown_path']}")

    if not args.goal and not args.ingest:
        parser.print_help()

if __name__ == "__main__":
    main()
