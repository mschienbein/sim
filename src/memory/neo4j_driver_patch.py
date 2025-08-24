"""
Patched Neo4j driver for Graphiti that correctly passes database parameter.
"""

from typing import Any
from typing_extensions import LiteralString
from neo4j import AsyncGraphDatabase, EagerResult
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.driver.driver import GraphDriver, GraphProvider
import logging

logger = logging.getLogger(__name__)

class PatchedNeo4jDriver(Neo4jDriver):
    """Fixed Neo4j driver that correctly uses the database parameter"""
    
    def __init__(self, uri: str, user: str | None, password: str | None, database: str = 'neo4j'):
        # Don't call parent __init__ - we'll do it ourselves
        GraphDriver.__init__(self)  # Call grandparent's init
        
        # Create the driver ourselves with proper auth
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user, password)
        )
        self._database = database
        self.provider = GraphProvider.NEO4J
        
        logger.info(f"Patched Neo4j driver initialized with database: {database}")

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        """Execute query with database parameter passed correctly"""
        logger.debug(f"execute_query called with kwargs: {kwargs}")
        
        # Graphiti passes these as direct kwargs, not in params
        # Extract them to build parameters_ dict
        params = {}
        
        # These are the expected query parameters from Graphiti
        query_params = ['reference_time', 'source', 'num_episodes', 'group_ids', 
                       'last_n', 'entity_id', 'uuid', 'limit']
        
        for param in query_params:
            if param in kwargs:
                params[param] = kwargs.pop(param)
        
        # Also check for params or parameters_ dict
        if 'params' in kwargs:
            params.update(kwargs.pop('params'))
        if 'parameters_' in kwargs:
            params.update(kwargs.pop('parameters_'))
            
        logger.debug(f"Executing query with database={self._database}, params: {params}")
        
        try:
            # Pass database_ as a keyword arg, parameters as parameters_
            result = await self.client.execute_query(
                cypher_query_, 
                parameters_=params,
                database_=self._database,
                **kwargs  # Pass any remaining kwargs
            )
            return result
        except Exception as e:
            logger.error(f'Error executing Neo4j query: {e}\n{cypher_query_}\nDatabase: {self._database}\nParams: {params}')
            raise