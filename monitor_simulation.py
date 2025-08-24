#!/usr/bin/env python3
"""Monitor running simulation and report timing metrics."""

import time
import subprocess
import os
from datetime import datetime, timedelta

def get_process_info():
    """Get info about running simulation process."""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'src.main run' in line and 'grep' not in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                return pid, cpu, mem
    except:
        pass
    return None, None, None

def count_neo4j_nodes():
    """Count nodes in Neo4j for current simulation."""
    try:
        from py2neo import Graph
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "your_password"))
        
        # Count different node types
        entity_count = graph.run("MATCH (n:Entity) WHERE n.group_id STARTS WITH 'sim_run_5fffb' RETURN count(n) as count").data()[0]['count']
        episode_count = graph.run("MATCH (n:Episodic) WHERE n.group_id STARTS WITH 'sim_run_5fffb' RETURN count(n) as count").data()[0]['count']
        edge_count = graph.run("MATCH ()-[r:RELATES_TO]->() WHERE r.group_id STARTS WITH 'sim_run_5fffb' RETURN count(r) as count").data()[0]['count']
        
        return entity_count, episode_count, edge_count
    except Exception as e:
        return 0, 0, 0

def monitor():
    """Monitor the simulation progress."""
    start_time = datetime(2025, 8, 24, 17, 51, 29)  # From log
    
    print("=" * 60)
    print("SIMULATION MONITORING REPORT")
    print("=" * 60)
    print(f"Start Time: {start_time}")
    print(f"Current Time: {datetime.now()}")
    
    elapsed = datetime.now() - start_time
    print(f"Elapsed Time: {elapsed}")
    print("-" * 60)
    
    # Check process
    pid, cpu, mem = get_process_info()
    if pid:
        print(f"Process Status: RUNNING")
        print(f"  PID: {pid}")
        print(f"  CPU Usage: {cpu}%")
        print(f"  Memory Usage: {mem}%")
    else:
        print(f"Process Status: NOT RUNNING")
    
    print("-" * 60)
    
    # Check logs
    if os.path.exists("simulation_timing.log"):
        size = os.path.getsize("simulation_timing.log")
        print(f"Log File Size: {size:,} bytes")
        
        # Count key events
        with open("simulation_timing.log", "r") as f:
            content = f.read()
            embedding_calls = content.count("embeddings")
            llm_calls = content.count("/v1/responses")
            neo4j_queries = content.count("neo4j")
            
            print(f"API Calls:")
            print(f"  Embedding API calls: {embedding_calls}")
            print(f"  LLM API calls: {llm_calls}")
            print(f"  Neo4j operations: {neo4j_queries}")
    
    print("-" * 60)
    
    # Try to count Neo4j nodes
    entities, episodes, edges = count_neo4j_nodes()
    if entities or episodes or edges:
        print(f"Graph Database Stats:")
        print(f"  Entity nodes: {entities}")
        print(f"  Episode nodes: {episodes}")
        print(f"  Relationship edges: {edges}")
        print("-" * 60)
    
    # Performance metrics
    if elapsed.total_seconds() > 0:
        print(f"Performance Metrics:")
        if embedding_calls > 0:
            print(f"  Embeddings/minute: {embedding_calls * 60 / elapsed.total_seconds():.2f}")
        if llm_calls > 0:
            print(f"  LLM calls/minute: {llm_calls * 60 / elapsed.total_seconds():.2f}")
        print(f"  Average time per iteration: ~{elapsed.total_seconds() / max(1, llm_calls):.2f}s")
    
    print("=" * 60)

if __name__ == "__main__":
    monitor()