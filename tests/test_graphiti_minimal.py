import asyncio
import os
import pytest

from src.config.settings import settings
from src.memory.graphiti_manager import GraphitiMemoryManager
from neo4j import AsyncGraphDatabase


pytestmark = pytest.mark.asyncio


async def _neo4j_ready() -> bool:
    try:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.username, settings.neo4j.password),
        )
        # Simple ping on the configured database
        await driver.execute_query("RETURN 1 AS ok", database_=settings.neo4j.database)
        await driver.close()
        return True
    except Exception:
        return False


async def test_initialize_and_ingest_observation():
    if not await _neo4j_ready():
        pytest.skip("Neo4j is not reachable. Ensure docker-compose neo4j is running.")
    manager = GraphitiMemoryManager()
    await manager.initialize()

    await manager.ingest_observation(
        agent_id="agent_minimal",
        observation="Saw a bird at the market",
        location="market",
        importance=0.7,
    )

    # If no exception was raised, ingestion worked to at least submit to Neo4j
    assert manager.initialized is True


async def test_create_relationship_edge_and_verify():
    if not await _neo4j_ready():
        pytest.skip("Neo4j is not reachable. Ensure docker-compose neo4j is running.")
    manager = GraphitiMemoryManager()
    await manager.initialize()

    a, b = "agent_a", "agent_b"
    await manager.create_relationship_edge(
        from_agent_id=a,
        to_agent_id=b,
        edge_type="SPOKE_WITH",
        attributes={"timestamp": "2025-01-01T00:00:00", "sentiment": 0.5},
    )

    # Verify the relationship exists in the 'simulation' database
    driver = AsyncGraphDatabase.driver(
        settings.neo4j.uri, auth=(settings.neo4j.username, settings.neo4j.password)
    )
    async with driver.session(database=settings.neo4j.database) as session:
        result = await session.run(
            """
            MATCH (a:Agent {id: $from_id})-[r:SPOKE_WITH]->(b:Agent {id: $to_id})
            RETURN count(r) AS c
            """,
            from_id=a,
            to_id=b,
        )
        record = await result.single()
        assert record is not None
        assert record["c"] >= 1

    await driver.close()
