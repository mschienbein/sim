import pytest
from neo4j import AsyncGraphDatabase
from src.config.settings import settings

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


async def test_simple_connection_and_count():
    if not await _neo4j_ready():
        pytest.skip("Neo4j is not reachable. Ensure docker-compose neo4j is running.")

    driver = AsyncGraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.username, settings.neo4j.password),
    )

    try:
        result = await driver.execute_query("RETURN 1 as test", database_=settings.neo4j.database)
        assert result.records[0]["test"] == 1

        result2 = await driver.execute_query(
            "MATCH (n) RETURN count(n) as count",
            database_=settings.neo4j.database,
        )
        assert result2.records[0]["count"] >= 0
    finally:
        await driver.close()
