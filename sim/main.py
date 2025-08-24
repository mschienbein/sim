"""
Main entry point for the LLM Agent Simulation.
"""

import asyncio
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import signal
import sys

from .orchestration.engine import SimulationEngine
from .config.settings import settings

# Initialize Typer app and Rich console
app = typer.Typer(
    name="sim",
    help="LLM Agent Simulation Framework - Watch AI agents develop personalities through interaction"
)
console = Console()

# Global engine for signal handling
engine: Optional[SimulationEngine] = None

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    console.print("\n[red]Received interrupt signal. Stopping simulation...[/red]")
    if engine:
        asyncio.create_task(engine.stop())
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

@app.command()
def run(
    agents: int = typer.Option(5, "--agents", "-a", help="Number of agents to simulate"),
    days: int = typer.Option(10, "--days", "-d", help="Number of days to simulate"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    no_graphiti: bool = typer.Option(False, "--no-graphiti", help="Use basic memory instead of Graphiti")
):
    """
    Run the agent simulation.
    
    Example:
        sim run --agents 5 --days 10
    """
    global engine
    
    console.print(Panel.fit(
        "[bold cyan]LLM Agent Simulation Framework[/bold cyan]\n"
        "Watch autonomous agents develop personalities through interaction",
        title="🎭 Welcome",
        border_style="cyan"
    ))
    
    # Load configuration
    config = {}
    if config_file and config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
    
    # Override with CLI parameters
    config["max_agents"] = agents
    config["max_days"] = days
    config["debug"] = debug
    config["use_graphiti"] = not no_graphiti
    
    # Run simulation
    console.print(f"\n[green]Starting simulation with {agents} agents for {days} days...[/green]")
    
    try:
        asyncio.run(run_simulation(config))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if debug:
            console.print_exception()

async def run_simulation(config: dict):
    """Run the simulation with given configuration"""
    global engine
    
    # Create engine
    engine = SimulationEngine(config)
    
    # Initialize with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        init_task = progress.add_task("Initializing simulation...", total=None)
        await engine.initialize()
        progress.update(init_task, completed=True)
    
    # Run simulation
    await engine.run_simulation(days=config["max_days"])
    
    # Cleanup
    await engine.stop()

@app.command()
def view(
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Checkpoint file to view"),
    latest: bool = typer.Option(False, "--latest", "-l", help="View latest checkpoint")
):
    """
    View simulation state from a checkpoint.
    
    Example:
        sim view --latest
    """
    import json
    
    # Find checkpoint file
    if latest:
        checkpoints = list(settings.checkpoints_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            console.print("[red]No checkpoints found[/red]")
            return
        checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    elif not checkpoint or not checkpoint.exists():
        console.print("[red]Checkpoint file not found[/red]")
        return
    
    # Load checkpoint
    with open(checkpoint) as f:
        data = json.load(f)
    
    # Display summary
    console.print(Panel.fit(
        f"[bold]Checkpoint: Day {data['day'] + 1}[/bold]\n"
        f"Tick: {data['tick']}",
        title="📊 Simulation State",
        border_style="blue"
    ))
    
    # Agent table
    table = Table(title="Agent Status")
    table.add_column("Agent", style="cyan")
    table.add_column("Role", style="magenta")
    table.add_column("Energy", style="green")
    table.add_column("Gold", style="yellow")
    table.add_column("Location", style="blue")
    
    for agent_id, state in data["agents"].items():
        table.add_row(
            state["name"],
            state["role"],
            f"{state['stats']['energy']:.0f}/100",
            str(state["stats"]["gold"]),
            str(state["location"])
        )
    
    console.print(table)
    
    # Metrics
    metrics = data.get("metrics", {})
    console.print(Panel(
        f"Conversations: {metrics.get('total_conversations', 0)}\n"
        f"Trades: {metrics.get('total_trades', 0)}\n"
        f"Reflections: {metrics.get('total_reflections', 0)}\n"
        f"Tokens Used: {metrics.get('tokens_used', 0):,}",
        title="📈 Metrics",
        border_style="green"
    ))

@app.command()
def world():
    """
    Display the world map.
    
    Example:
        sim world
    """
    from .world.grid import WorldGrid
    
    world = WorldGrid()
    console.print(Panel(
        world.visualize_grid(show_agents=False),
        title="🗺️ World Map",
        border_style="cyan"
    ))

@app.command()
def interview(
    agent_name: str = typer.Argument(..., help="Name of agent to interview"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Checkpoint file"),
    latest: bool = typer.Option(False, "--latest", "-l", help="Use latest checkpoint")
):
    """
    Interview an agent about their experiences.
    
    Example:
        sim interview Aldric --latest
    """
    import json
    
    # Find checkpoint
    if latest:
        checkpoints = list(settings.checkpoints_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            console.print("[red]No checkpoints found[/red]")
            return
        checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    elif not checkpoint or not checkpoint.exists():
        console.print("[red]Checkpoint file required[/red]")
        return
    
    # Load checkpoint
    with open(checkpoint) as f:
        data = json.load(f)
    
    # Find agent
    agent_state = None
    for agent_id, state in data["agents"].items():
        if state["name"].lower() == agent_name.lower():
            agent_state = state
            break
    
    if not agent_state:
        console.print(f"[red]Agent '{agent_name}' not found[/red]")
        return
    
    # Display agent info
    console.print(Panel.fit(
        f"[bold]{agent_state['name']}[/bold] - {agent_state['role'].title()}\n"
        f"{agent_state.get('backstory', 'No backstory available')}",
        title="🎭 Agent Profile",
        border_style="cyan"
    ))
    
    # Personality traits
    personality = agent_state.get("personality", {})
    console.print("\n[bold]Personality Traits:[/bold]")
    for trait, value in personality.items():
        bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
        console.print(f"  {trait:20} {bar} {value:.2f}")
    
    # Relationships
    relationships = agent_state.get("relationships", {})
    if relationships:
        console.print("\n[bold]Relationships:[/bold]")
        for other_id, rel in relationships.items():
            trust = rel.get("trust", 0)
            friendship = rel.get("friendship", 0)
            console.print(f"  {other_id:20} Trust: {trust:.2f}, Friendship: {friendship:.2f}")
    
    # Goals
    goals = agent_state["stats"].get("short_term_goals", [])
    if goals:
        console.print("\n[bold]Current Goals:[/bold]")
        for goal in goals:
            console.print(f"  • {goal}")
    
    console.print("\n[yellow]Note: Full interview mode with TTS will be available in future versions[/yellow]")

@app.command()
def stats():
    """
    Display simulation statistics.
    
    Example:
        sim stats
    """
    import json
    from datetime import datetime
    
    # Find all checkpoints
    checkpoints = list(settings.checkpoints_dir.glob("checkpoint_*.json"))
    if not checkpoints:
        console.print("[red]No simulation data found[/red]")
        return
    
    # Load latest checkpoint
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        data = json.load(f)
    
    # Token usage files
    token_files = list(settings.logs_dir.glob("token_usage_*.json"))
    total_tokens = 0
    total_cost = 0
    
    for file in token_files:
        with open(file) as f:
            usage = json.load(f)
            total_tokens += usage.get("total_tokens", 0)
            total_cost += usage.get("cost_estimate", 0)
    
    # Display statistics
    console.print(Panel.fit(
        f"[bold]Simulation Statistics[/bold]\n"
        f"Total Checkpoints: {len(checkpoints)}\n"
        f"Latest Day: {data['day'] + 1}\n"
        f"Total Ticks: {data['tick']}",
        title="📊 Overview",
        border_style="blue"
    ))
    
    # Metrics table
    metrics = data.get("metrics", {})
    table = Table(title="Interaction Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Conversations", str(metrics.get("total_conversations", 0)))
    table.add_row("Trades", str(metrics.get("total_trades", 0)))
    table.add_row("Reflections", str(metrics.get("total_reflections", 0)))
    table.add_row("Knowledge Propagated", str(metrics.get("knowledge_propagated", 0)))
    
    console.print(table)
    
    # Cost analysis
    console.print(Panel(
        f"Total Tokens Used: {total_tokens:,}\n"
        f"Estimated Cost: ${total_cost:.4f}\n"
        f"Average Tokens/Day: {total_tokens // max(1, data['day'] + 1):,}",
        title="💰 Cost Analysis",
        border_style="yellow"
    ))

@app.command()
def config():
    """
    Display current configuration.
    
    Example:
        sim config
    """
    console.print(Panel.fit(
        "[bold]Current Configuration[/bold]",
        title="⚙️ Settings",
        border_style="blue"
    ))
    
    # Display key settings
    config_items = [
        ("Max Agents", settings.simulation.max_agents),
        ("Max Days", settings.simulation.max_days),
        ("Ticks per Day", settings.simulation.ticks_per_day),
        ("World Size", f"{settings.simulation.world_size[0]}x{settings.simulation.world_size[1]}"),
        ("Daily Token Budget", f"{settings.rate_limit.daily_token_budget:,}"),
        ("Per Agent Token Limit", f"{settings.rate_limit.per_agent_token_limit:,}"),
        ("Sage Search Limit", f"{settings.SAGE_SEARCH_LIMIT_PER_DAY}/day"),
        ("Neo4j URI", settings.neo4j.uri),
        ("OpenAI Model", settings.llm.openai_model_id),
    ]
    
    for name, value in config_items:
        console.print(f"  {name:25} {value}")
    
    console.print("\n[dim]Edit .env file to change configuration[/dim]")

@app.command()
def clean():
    """
    Clean up logs and checkpoints.
    
    Example:
        sim clean
    """
    import shutil
    
    if typer.confirm("This will delete all logs and checkpoints. Continue?"):
        # Clean logs
        if settings.logs_dir.exists():
            shutil.rmtree(settings.logs_dir)
            settings.logs_dir.mkdir()
        
        # Clean checkpoints
        if settings.checkpoints_dir.exists():
            shutil.rmtree(settings.checkpoints_dir)
            settings.checkpoints_dir.mkdir()
        
        console.print("[green]✓ Cleaned logs and checkpoints[/green]")
    else:
        console.print("[yellow]Cancelled[/yellow]")

if __name__ == "__main__":
    app()