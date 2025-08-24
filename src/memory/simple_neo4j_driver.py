"""
Simple Neo4j driver for Graphiti that works with 'simulation' database.
"""

from typing import Any
from typing_extensions import LiteralString
from neo4j import AsyncGraphDatabase, EagerResult
from graphiti_core.driver.driver import GraphDriver, GraphProvider
import logging

logger = logging.getLogger(__name__)

class SimulationNeo4jDriver(GraphDriver):
    """Neo4j driver that works with simulation database"""
    
    def __init__(self, uri: str, user: str | None, password: str | None):
        # Initialize GraphDriver base
        super().__init__()
        self.provider = GraphProvider.NEO4J
        
        # Create driver directly
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user, password)
        )
        self._database = 'simulation'
        logger.info("SimulationNeo4jDriver initialized with simulation database")
    
    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        """Execute query with simulation database"""
        # Get parameters from kwargs
        params = {}
        
        # Graphiti passes these as kwargs, not in params
        for key in list(kwargs.keys()):
            if key not in ['database_', 'parameters_', 'params']:
                params[key] = kwargs.pop(key)
        
        # Also merge any params dict
        if 'params' in kwargs:
            params.update(kwargs.pop('params'))
        if 'parameters_' in kwargs:
            params.update(kwargs.pop('parameters_'))
        
        # Remove database_ from params if it ended up there
        params.pop('database_', None)
        
        try:
            # Execute with simulation database
            result = await self.client.execute_query(
                cypher_query_, 
                parameters_=params,
                database_='simulation'  # Always use simulation
            )
            return result
        except Exception as e:
            logger.error(f'Error executing Neo4j query: {e}')
            raise
    
    def session(self, database: str | None = None):
        """Create a session with simulation database"""
        return self.client.session(database=self._database)
    
    async def close(self) -> None:
        """Close the driver"""
        return await self.client.close()
    
    def delete_all_indexes(self):
        """Delete all indexes - not implemented for safety"""
        logger.warning("delete_all_indexes called but not implemented")
        pass