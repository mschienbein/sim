"""
Performance optimizations for Graphiti memory management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class GraphitiOptimizer:
    """Optimizations for Graphiti operations"""
    
    @staticmethod
    async def batch_add_episodes(graphiti_instance, episodes: List[Dict[str, Any]], group_id: str):
        """
        Batch multiple episodes together to reduce API calls.
        Process episodes in smaller chunks to avoid JSON truncation.
        """
        BATCH_SIZE = 3  # Process 3 episodes at a time to avoid truncation
        results = []
        
        for i in range(0, len(episodes), BATCH_SIZE):
            batch = episodes[i:i + BATCH_SIZE]
            batch_tasks = []
            
            for episode in batch:
                # Truncate content if too long to avoid JSON issues
                content = episode.get("content", "")
                if len(content) > 2000:
                    content = content[:1997] + "..."
                    
                task = graphiti_instance.add_episode(
                    content=content,
                    source_description=episode.get("source_description", "Agent interaction"),
                    source=episode.get("source", "simulation"),
                    group_id=group_id,
                    metadata=episode.get("metadata", {})
                )
                batch_tasks.append(task)
            
            # Process batch in parallel
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing episode {i+j}: {result}")
                    else:
                        results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                
        return results
    
    @staticmethod
    def truncate_json_response(data: Dict[str, Any], max_size: int = 15000) -> Dict[str, Any]:
        """
        Truncate JSON data to prevent parsing errors.
        Prioritizes keeping complete entity resolutions.
        """
        json_str = json.dumps(data)
        if len(json_str) <= max_size:
            return data
            
        # Truncate entity resolutions if needed
        if "entity_resolutions" in data:
            resolutions = data["entity_resolutions"]
            # Keep first N resolutions that fit
            truncated_resolutions = []
            current_size = 100  # Base JSON overhead
            
            for resolution in resolutions:
                res_size = len(json.dumps(resolution))
                if current_size + res_size < max_size:
                    truncated_resolutions.append(resolution)
                    current_size += res_size
                else:
                    break
                    
            data["entity_resolutions"] = truncated_resolutions
            
        return data
    
    @staticmethod
    async def parallel_search(graphiti_instance, queries: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute multiple search queries in parallel.
        """
        search_tasks = []
        
        for query in queries:
            task = graphiti_instance.search(
                query=query.get("query", ""),
                group_id=query.get("group_id"),
                search_type=query.get("search_type", "unstructured"),
                limit=query.get("limit", 10)
            )
            search_tasks.append(task)
            
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search query {i} failed: {result}")
                valid_results.append([])
            else:
                valid_results.append(result)
                
        return valid_results


# Commented out as it goes against the point of Graphiti's incremental updates
# class MemoryCompressor:
#     """Compress and consolidate memories to reduce storage"""
#     
#     @staticmethod
#     async def compress_old_memories(graphiti_instance, group_id: str, days_old: int = 7):
#         """
#         Compress memories older than specified days into summary facts.
#         """
#         cutoff_date = datetime.now() - timedelta(days=days_old)
#         
#         # Search for old episodic memories
#         old_memories = await graphiti_instance.search(
#             query="",
#             group_id=group_id,
#             search_type="structured",
#             limit=100
#         )
#         
#         # Group by agent and create summaries
#         agent_memories = {}
#         for memory in old_memories:
#             if hasattr(memory, 'created_at') and memory.created_at < cutoff_date:
#                 agent_id = memory.metadata.get('agent_id', 'unknown')
#                 if agent_id not in agent_memories:
#                     agent_memories[agent_id] = []
#                 agent_memories[agent_id].append(memory)
#         
#         # Create compressed summaries
#         for agent_id, memories in agent_memories.items():
#             if len(memories) > 10:
#                 summary = f"Historical summary for {agent_id}: "
#                 summary += f"Participated in {len(memories)} events. "
#                 
#                 # Extract key themes
#                 themes = set()
#                 for mem in memories[:20]:  # Sample first 20
#                     if hasattr(mem, 'content'):
#                         if 'trade' in mem.content.lower():
#                             themes.add('trading')
#                         if 'work' in mem.content.lower():
#                             themes.add('working')
#                         if 'speak' in mem.content.lower():
#                             themes.add('socializing')
#                             
#                 summary += f"Main activities: {', '.join(themes)}."
#                 
#                 # Add as compressed fact
#                 await graphiti_instance.add_episode(
#                     content=summary,
#                     source_description="Memory compression",
#                     source="system",
#                     group_id=group_id,
#                     metadata={
#                         "type": "compressed_memory",
#                         "agent_id": agent_id,
#                         "original_count": len(memories),
#                         "compression_date": datetime.now().isoformat()
#                     }
#                 )
#                 
#                 logger.info(f"Compressed {len(memories)} memories for {agent_id}")


class QueryOptimizer:
    """Optimize Neo4j queries for better performance"""
    
    @staticmethod
    def get_optimized_context_query(agent_id: str, group_id: str) -> str:
        """
        Get an optimized Cypher query for agent context.
        Uses indexes and limits for better performance.
        """
        return f"""
        MATCH (agent:Entity {{name: '{agent_id}', group_id: '{group_id}'}})
        OPTIONAL MATCH (agent)-[r:RELATES_TO]-(other:Entity)
        WHERE other.group_id = '{group_id}'
        WITH agent, COLLECT(DISTINCT {{
            entity: other.name,
            relationship: r.fact,
            strength: r.strength
        }}) AS relationships
        
        OPTIONAL MATCH (episode:Episodic)
        WHERE episode.group_id = '{group_id}'
        AND episode.content CONTAINS '{agent_id}'
        AND episode.created_at > datetime() - duration('P1D')
        WITH agent, relationships, COLLECT(DISTINCT {{
            content: episode.content,
            time: episode.created_at
        }}) AS recent_episodes
        
        RETURN {{
            agent: agent.name,
            summary: agent.summary,
            relationships: relationships[..10],
            recent_episodes: recent_episodes[..20]
        }} AS context
        """
    
    @staticmethod
    def get_batch_relationship_query(agent_ids: List[str], group_id: str) -> str:
        """
        Get relationships for multiple agents in one query.
        """
        agent_list = "','".join(agent_ids)
        return f"""
        MATCH (a1:Entity)-[r:RELATES_TO]-(a2:Entity)
        WHERE a1.name IN ['{agent_list}']
        AND a2.name IN ['{agent_list}']
        AND a1.group_id = '{group_id}'
        AND a2.group_id = '{group_id}'
        RETURN a1.name AS agent1, a2.name AS agent2, 
               r.fact AS relationship, r.strength AS strength
        """


class RequestHandler:
    """Handle API requests with timeout and retry logic"""
    
    @staticmethod
    async def with_timeout(coro, timeout_seconds: int = 30):
        """
        Execute coroutine with timeout.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout_seconds} seconds")
            return None
    
    @staticmethod
    async def with_retry(coro_factory, max_retries: int = 3, backoff: float = 1.0):
        """
        Retry a coroutine with exponential backoff.
        coro_factory should be a function that returns a new coroutine.
        """
        for attempt in range(max_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        return None