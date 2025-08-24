#!/usr/bin/env python3
"""Monitor the full simulation run in real-time."""

import time
import subprocess
import os
from datetime import datetime, timedelta
from neo4j import GraphDatabase

def get_process_status(pid):
    """Check if process is running."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid,vsz,rss,pcpu,pmem,etime,command"],
            capture_output=True,
            text=True
        )
        if str(pid) in result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
    except:
        pass
    return None

def get_neo4j_stats():
    """Get current graph statistics."""
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "simulation123")
        )
        
        with driver.session(database="simulation") as session:
            # Count different node types
            result = session.run("""
                MATCH (n)
                RETURN 
                    labels(n)[0] as type,
                    count(n) as count
                ORDER BY count DESC
            """)
            node_stats = {record['type']: record['count'] for record in result}
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            
            # Get simulation run info
            result = session.run("""
                MATCH (s:SimulationRun)
                RETURN s.id as run_id, s.created_at as created
                LIMIT 1
            """)
            run_info = result.single()
            
        driver.close()
        
        return {
            "nodes": node_stats,
            "relationships": rel_count,
            "run_id": run_info['run_id'] if run_info else None
        }
    except Exception as e:
        return {"error": str(e)}

def count_log_events(log_file):
    """Count key events in log file."""
    if not os.path.exists(log_file):
        return {}
    
    counts = {
        "episodes": 0,
        "llm_calls": 0,
        "embeddings": 0,
        "errors": 0,
        "warnings": 0,
        "agents": set(),
        "last_tick": 0
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "add_episode" in line:
                    counts["episodes"] += 1
                if "responses" in line and "POST" in line:
                    counts["llm_calls"] += 1
                if "embeddings" in line and "POST" in line:
                    counts["embeddings"] += 1
                if "[ERROR]" in line:
                    counts["errors"] += 1
                if "[WARNING]" in line:
                    counts["warnings"] += 1
                if "Tick" in line:
                    try:
                        # Extract tick number
                        parts = line.split("Tick")[1].split()[0]
                        tick = int(parts.strip(':'))
                        counts["last_tick"] = max(counts["last_tick"], tick)
                    except:
                        pass
                # Track unique agents
                if "agent_" in line:
                    import re
                    agents = re.findall(r'agent_\w+', line)
                    counts["agents"].update(agents)
    except:
        pass
    
    counts["unique_agents"] = len(counts["agents"])
    del counts["agents"]
    
    return counts

def monitor(pid, start_time):
    """Main monitoring loop."""
    print("=" * 80)
    print("FULL SIMULATION RUN MONITOR")
    print("=" * 80)
    print(f"Started: {start_time}")
    print(f"Process ID: {pid}")
    print("-" * 80)
    
    while True:
        # Check process
        process_info = get_process_status(pid)
        
        if not process_info:
            print("\n" + "=" * 80)
            print("SIMULATION COMPLETED!")
            elapsed = datetime.now() - start_time
            print(f"Total runtime: {elapsed}")
            
            # Final stats
            neo4j_stats = get_neo4j_stats()
            log_stats = count_log_events("full_simulation_run.log")
            
            print("\nFinal Statistics:")
            print(f"  Total ticks: {log_stats.get('last_tick', 0)}")
            print(f"  Total episodes: {log_stats.get('episodes', 0)}")
            print(f"  LLM calls: {log_stats.get('llm_calls', 0)}")
            print(f"  Embedding calls: {log_stats.get('embeddings', 0)}")
            print(f"  Errors: {log_stats.get('errors', 0)}")
            
            if "nodes" in neo4j_stats:
                print("\nGraph Database:")
                for node_type, count in neo4j_stats["nodes"].items():
                    print(f"    {node_type}: {count}")
                print(f"    Total relationships: {neo4j_stats['relationships']}")
            
            print("=" * 80)
            break
        
        # Current status
        elapsed = datetime.now() - start_time
        print(f"\r[{elapsed}] Running... ", end="")
        
        # Periodic detailed update every 30 seconds
        if int(elapsed.total_seconds()) % 30 == 0:
            neo4j_stats = get_neo4j_stats()
            log_stats = count_log_events("full_simulation_run.log")
            
            print(f"\n  Tick: {log_stats.get('last_tick', 0)}/120")
            print(f"  Episodes: {log_stats.get('episodes', 0)}")
            print(f"  API calls: {log_stats.get('llm_calls', 0)}")
            
            if "nodes" in neo4j_stats:
                total_nodes = sum(neo4j_stats["nodes"].values())
                print(f"  Graph nodes: {total_nodes}")
                print(f"  Relationships: {neo4j_stats.get('relationships', 0)}")
            
            # Process memory
            parts = process_info.split()
            if len(parts) > 3:
                cpu = parts[3]
                mem = parts[4]
                print(f"  CPU: {cpu}% | Memory: {mem}%")
        
        time.sleep(1)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Try to find PID from log
        try:
            with open("full_simulation_run.log", "r") as f:
                for line in f:
                    if "Simulation PID:" in line:
                        pid = int(line.split(":")[-1].strip())
                        break
        except:
            print("Usage: python monitor_full_run.py <PID>")
            sys.exit(1)
    else:
        pid = int(sys.argv[1])
    
    start_time = datetime.now()
    
    # Get actual start time from log if available
    try:
        with open("full_simulation_run.log", "r") as f:
            for line in f:
                if "Start time:" in line:
                    # Parse the date
                    date_str = line.split("Start time:")[-1].strip()
                    # Simple parse - adjust format as needed
                    break
    except:
        pass
    
    monitor(pid, start_time)