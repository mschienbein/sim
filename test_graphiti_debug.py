#!/usr/bin/env python
"""Debug test for Graphiti Neo4j authentication"""

import asyncio
import logging
import sys
import os

# Add path and set env
sys.path.insert(0, '/Users/mooki/Code/sim')
os.chdir('/Users/mooki/Code/sim')

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.memory.graphiti_manager import GraphitiMemoryManager
from src.config.settings import settings

async def test():
    
    print(f"Neo4j URI: {settings.neo4j.uri}")
    print(f"Neo4j Database: {settings.neo4j.database}")
    print(f"Neo4j Username: {settings.neo4j.username}")
    
    manager = GraphitiMemoryManager()
    
    print("Initializing Graphiti...")
    await manager.initialize()
    print('✓ Initialized successfully')
    
    # Try a simple operation
    print("Ingesting test observation...")
    await manager.ingest_observation(
        agent_id='test_agent',
        observation='Test observation',
        location='test_location'
    )
    print('✓ Ingested observation successfully')

if __name__ == "__main__":
    asyncio.run(test())