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

from src.orchestration.engine import SimulationEngine
from src.config.settings import settings

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
    
    # Run simulation with live dashboard if not in debug mode
    if config.get("debug"):
        # Simple run without live display
        await engine.run_simulation(days=config["max_days"])
    else:
        # Run with live dashboard
        await run_with_dashboard(engine, config["max_days"])
    
    # Cleanup
    await engine.stop()

async def run_with_dashboard(engine: SimulationEngine, days: int):
    """Run simulation with live dashboard display"""
    
    def make_layout() -> Layout:
        """Create the dashboard layout"""
        layout = Layout(name="root")
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=4)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="center"),
            Layout(name="right")
        )
        
        return layout
    
    def generate_dashboard() -> Layout:
        """Generate dashboard content"""
        layout = make_layout()
        
        # Header
        layout["header"].update(Panel(
            f"[bold cyan]🎭 LLM Agent Simulation[/bold cyan] | Day {engine.current_day + 1}/{days} | Tick {engine.current_tick}",
            border_style="cyan"
        ))
        
        # Agent Status Table
        agent_table = Table(title="Agent Status", expand=True)
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Energy", style="green")
        agent_table.add_column("Location", style="blue")
        agent_table.add_column("Action", style="yellow")
        
        for agent_id, agent in engine.agents.items():
            location = engine.world.get_location_by_position(agent.location)
            location_name = location.name if location else str(agent.location)
            current_action = agent.current_action or "idle"
            
            agent_table.add_row(
                agent.name,
                f"{agent.stats.energy:.0f}/100",
                location_name,
                current_action
            )
        
        layout["left"].update(Panel(agent_table, title="👥 Agents", border_style="green"))
        
        # Mini World Map
        world_lines = []
        for y in range(min(10, engine.world.height)):
            row = ""
            for x in range(min(10, engine.world.width)):
                # Check if any agent is at this position
                agent_here = None
                for agent_id, agent in engine.agents.items():
                    if agent.location == (x, y):
                        agent_here = agent.name[0]  # First letter of name
                        break
                
                if agent_here:
                    row += f"[bold yellow]{agent_here}[/bold yellow] "
                else:
                    loc = engine.world.get_location(x, y)
                    if loc:
                        symbol = settings.world.locations.get(loc.location_type.value, {}).get("symbol", ".")
                        row += f"[dim]{symbol}[/dim] "
                    else:
                        row += ". "
            world_lines.append(row)
        
        world_text = "\n".join(world_lines)
        layout["center"].update(Panel(world_text, title="🗺️ World", border_style="magenta"))
        
        # Metrics Panel
        metrics = engine.metrics
        metrics_text = (
            f"[bold]Interactions[/bold]\n"
            f"Conversations: {metrics.get('total_conversations', 0)}\n"
            f"Trades: {metrics.get('total_trades', 0)}\n"
            f"Reflections: {metrics.get('total_reflections', 0)}\n\n"
            f"[bold]Resources[/bold]\n"
            f"Tokens Used: {metrics.get('tokens_used', 0):,}\n"
            f"Est. Cost: ${metrics.get('tokens_used', 0) * 0.00001:.4f}"
        )
        
        layout["right"].update(Panel(metrics_text, title="📊 Metrics", border_style="blue"))
        
        # Recent Events in Footer
        recent_events = []
        if engine.event_log:
            for event in engine.event_log[-3:]:  # Last 3 events
                agent_name = engine.agents[event['agent']].name
                action = event['action']
                recent_events.append(f"[dim]{agent_name}[/dim] → {action}")
        
        events_text = "\n".join(recent_events) if recent_events else "No recent events"
        layout["footer"].update(Panel(events_text, title="📝 Recent Activity", border_style="yellow"))
        
        return layout
    
    # Create live display
    with Live(generate_dashboard(), refresh_per_second=2, console=console) as live:
        # Store reference for updates
        engine.live_display = live
        
        # Run simulation
        original_run = engine.run_simulation
        
        async def run_with_updates(days: int):
            """Wrapper to update display during simulation"""
            engine.running = True
            
            for day in range(days):
                if not engine.running:
                    break
                
                engine.current_day = day
                
                for tick in range(settings.simulation.ticks_per_day):
                    if not engine.running or engine.paused:
                        await engine._handle_pause()
                        if not engine.running:
                            break
                    
                    engine.current_tick = day * settings.simulation.ticks_per_day + tick
                    
                    # Get world state
                    world_state = engine.world.get_world_state(engine.current_tick)
                    
                    # Run simulation phases
                    perceptions = await engine._gather_perceptions(world_state)
                    decisions = await engine._make_decisions(perceptions)
                    await engine._resolve_actions(decisions)
                    await engine._handle_interactions()
                    engine._update_world_state()
                    engine._update_agent_states(tick)
                    
                    if tick % 6 == 0:
                        await engine._trigger_reflections()
                    
                    engine._record_metrics()
                    
                    # Update live display
                    live.update(generate_dashboard())
                    
                    # Small delay
                    await asyncio.sleep(settings.simulation.tick_duration_ms / 1000)
                
                # End of day processing
                await engine._end_of_day_processing()
                
                # Update display with day summary
                live.update(generate_dashboard())
            
            # Final update
            await engine._print_final_summary()
        
        # Run with display updates
        await run_with_updates(days)

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
    from src.world.grid import WorldGrid
    
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