
# Pipeline Data Analysis Agent - Main Entry Point : A sophisticated AI agent for analyzing pipeline data through natural language queries.


import asyncio
import sys
import os
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_agent.core.agent import PipelineDataAgent
from data_agent.utils.logger import setup_logger

console = Console()
logger = setup_logger()

@click.command()
@click.option(
    "--config-path",
    "-c",
    help="Path to configuration file",
    default="config/config.yaml",
    type=click.Path(),
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(config_path, verbose):
    """
    Pipeline Data Analysis Agent

    An AI-powered agent for analyzing pipeline data through natural language queries.
    Supports pattern recognition, anomaly detection, and causal analysis.
    """

    # Setup logging level
    if verbose:
        logger.setLevel("DEBUG")

    # Display welcome message
    welcome_text = Text()
    welcome_text.append("🤖 Pipeline Data Agent", style="bold blue")
    welcome_text.append("\n\nAn AI-powered assistant for pipeline data analysis")
    welcome_text.append("\nSupports: Pattern Recognition • Anomaly Detection • Causal Analysis")

    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                "[red]❌ ANTHROPIC_API_KEY environment variable not found!\n\n"
                "Please set your API key:\n"
                "export ANTHROPIC_API_KEY='your-api-key-here'",
                title="Configuration Error",
                border_style="red",
            )
        )
        sys.exit(1)

    try:
        asyncio.run(run_agent(config_path))
    except KeyboardInterrupt:
        console.print("\n\n[yellow]👋 Goodbye! Thanks for using Pipeline Data Agent.[/yellow]")
    except Exception as e:
        logger.error(f"Application error: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


async def run_agent(config_path):
    """Initialize and run the main agent loop."""

    agent = PipelineDataAgent(config_path=config_path)

    # Always prompt user for dataset path
    dataset_path = None
    while not dataset_path or not Path(dataset_path).exists():
        dataset_path = console.input(
            "[yellow]📁 Please enter the path to your pipeline dataset (CSV or Parquet): [/yellow]"
        ).strip()
        if not Path(dataset_path).exists():
            console.print("[red]❌ Dataset file not found! Please try again.[/red]")

    console.print(f"[yellow]📁 Loading dataset from {dataset_path}...[/yellow]")
    await agent.setup_dataset(dataset_path=dataset_path)

    console.print("[green]✅ Dataset loaded successfully![/green]")

    # Display dataset info
    info = await agent.get_dataset_info()
    console.print(Panel(info, title="Dataset Information", border_style="green"))

    # Interactive chat loop
    console.print(
        Panel(
            "[cyan]🎯 Ready for your questions![/cyan]\n\n"
            "Try asking:\n"
            "• 'Find anomalies in scheduled quantities'\n"
            "• 'Show correlations between location and quantities'\n"
            "• 'Cluster pipelines by characteristics'\n"
            "• 'What patterns do you see in the data?'\n\n"
            "[dim]Type 'help' for more examples, 'quit' to exit[/dim]",
            title="Getting Started",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if user_input.lower() == "help":
                show_help()
                continue

            if user_input.lower() == "clear":
                console.clear()
                continue

            console.print("\n[yellow]🤔 Thinking...[/yellow]")
            response = await agent.process_query(user_input)
            console.print(f"\n[bold green]🤖 Agent:[/bold green]\n{response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            console.print(f"\n[red]❌ Error processing query: {e}[/red]")


def show_help():
    """Display help information."""
    help_text = """[bold]Available Commands:[/bold]

🔍 [cyan]Query Examples:[/cyan]
    • "How many pipelines are in Texas?"
    • "Find outliers in scheduled quantities"
    • "Show correlation between state and delivery sign"
    • "Cluster pipelines by location and category"
    • "What's unusual about the data from 2024?"
    • "Analyze patterns in gas deliveries by day"

📊 [cyan]Analysis Types:[/cyan]
    • Pattern Recognition: trends, correlations, clustering
    • Anomaly Detection: outliers, unusual patterns
    • Causal Analysis: explanations with evidence
    • Statistical Analysis: counts, averages, distributions

⌨️ [cyan]Commands:[/cyan]
    • help - Show this help
    • clear - Clear the screen
    • quit/exit/q - Exit the agent

💡 [dim]Tip: Be specific in your questions for better results![/dim]
    """
    console.print(Panel(help_text, title="Help", border_style="blue"))


if __name__ == "__main__":
    main()
