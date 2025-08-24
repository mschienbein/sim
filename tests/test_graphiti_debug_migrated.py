import pytest
from neo4j import AsyncGraphDatabase
from src.config.settings import settings
from src.memory.graphiti_manager import GraphitiMemoryManager

pytestmark = pytest.mark.asyncio


async def _neo4j_ready() -> bool:
    try:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.username, settings.neo4j.password),
        )
        await driver.execute_query("RETURN 1 AS ok", database_=settings.neo4j.database)
        await driver.close()
        return True
    except Exception:
        return False


async def test_debug_initialize_and_ingest():
    if not await _neo4j_ready():
        pytest.skip("Neo4j is not reachable. Ensure docker-compose neo4j is running.")

    manager = GraphitiMemoryManager()
    await manager.initialize()

    await manager.ingest_observation(
        agent_id="debug_agent",
        observation="Debug observation",
        location="market",
        importance=0.5,
    )

    assert manager.initialized is True
