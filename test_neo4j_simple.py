#!/usr/bin/env python
"""Simple test of Neo4j connection"""

import asyncio
from neo4j import AsyncGraphDatabase

async def test():
    # Test direct connection
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "simulation123")
    )
    
    try:
        # Test with simulation database
        result = await driver.execute_query(
            "RETURN 1 as test",
            database_="simulation"
        )
        print(f"✓ Connected to simulation database: {result}")
        
        # Try to query
        result2 = await driver.execute_query(
            "MATCH (n) RETURN count(n) as count",
            database_="simulation"
        )
        print(f"✓ Node count: {result2.records[0]['count']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await driver.close()

if __name__ == "__main__":
    asyncio.run(test())