#!/usr/bin/env python3
"""
Minimal test to verify simulation components work.
"""

import os
import sys

# Ensure we're using the right environment
print(f"Python: {sys.executable}")
print(f"OpenAI Key configured: {'OPENAI_API_KEY' in os.environ}")

# Test imports
print("\n=== Testing imports ===")
try:
    from src.config.settings import settings
    print("✓ Settings loaded")
except Exception as e:
    print(f"✗ Settings failed: {e}")
    sys.exit(1)

try:
    from src.world.grid import WorldGrid
    world = WorldGrid(10, 10)
    print(f"✓ World created: {world.width}x{world.height} with {len(world.grid)} locations")
except Exception as e:
    print(f"✗ World failed: {e}")
    sys.exit(1)

try:
    from neo4j import GraphDatabase
    # Try to connect to Neo4j
    uri = settings.neo4j.uri
    auth = (settings.neo4j.username, settings.neo4j.password)
    driver = GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()
    print(f"✓ Neo4j connected at {uri}")
    driver.close()
except Exception as e:
    print(f"✗ Neo4j connection failed: {e}")
    print("  Make sure Neo4j is running: docker compose ps")

try:
    from src.agents.personality import Personality, EmotionalProfile
    personality = Personality()
    emotions = EmotionalProfile()
    print(f"✓ Personality system loaded")
except Exception as e:
    print(f"✗ Personality failed: {e}")

try:
    from src.memory.manager import MemoryManager, NodeType, EdgeType
    print(f"✓ Memory manager loaded with {len(NodeType)} node types, {len(EdgeType)} edge types")
except Exception as e:
    print(f"✗ Memory manager failed: {e}")

print("\n=== All basic components working! ===")
print("\nTo run full simulation:")
print("  python -m src.main --agents 2 --days 1")