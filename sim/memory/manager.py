"""
Memory management system using Neo4j and Graphiti for temporal knowledge graphs.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import json
import numpy as np
from enum import Enum

from neo4j import AsyncGraphDatabase
from sentence_transformers import SentenceTransformer
import asyncio

from ..config.settings import settings

class MemoryType(Enum):
    """Types of memories"""
    OBSERVATION = "observation"
    CONVERSATION = "conversation"
    REFLECTION = "reflection"
    ACTION = "action"
    LEARNED = "learned"
    RUMOR = "rumor"
    FACT = "fact"

@dataclass
class Memory:
    """Individual memory instance"""
    id: str
    agent_id: str
    memory_type: MemoryType
    content: str
    importance: float
    timestamp: datetime
    t_valid: datetime  # When the event happened
    t_invalid: Optional[datetime]  # When it becomes invalid
    t_ingested: datetime  # When stored in system
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    references: List[str]  # IDs of related entities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "t_valid": self.t_valid.isoformat(),
            "t_invalid": self.t_invalid.isoformat() if self.t_invalid else None,
            "t_ingested": self.t_ingested.isoformat(),
            "metadata": json.dumps(self.metadata),
            "references": self.references
        }

class MemoryManager:
    """Manages agent memories in Neo4j with Graphiti-style temporal awareness"""
    
    def __init__(self):
        self.driver = None
        self.embedding_model = None
        self.memory_cache = {}  # Simple cache for recent memories
        self.initialize_embedding_model()
    
    def initialize_embedding_model(self):
        """Initialize the sentence transformer for embeddings"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def connect(self):
        """Connect to Neo4j database"""
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.username, settings.neo4j.password)
        )
        
        # Initialize schema
        await self.initialize_schema()
    
    async def initialize_schema(self):
        """Create indexes and constraints in Neo4j"""
        async with self.driver.session() as session:
            queries = [
                # Create constraints
                "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT knowledge_id IF NOT EXISTS FOR (k:Knowledge) REQUIRE k.id IS UNIQUE",
                "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
                
                # Create indexes for performance
                "CREATE INDEX memory_agent IF NOT EXISTS FOR (m:Memory) ON (m.agent_id)",
                "CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
                "CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
                "CREATE INDEX memory_importance IF NOT EXISTS FOR (m:Memory) ON (m.importance)",
                
                # Full-text search index
                "CREATE FULLTEXT INDEX memory_content IF NOT EXISTS FOR (m:Memory) ON EACH [m.content]"
            ]
            
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    print(f"Schema query failed (may already exist): {e}")
    
    async def store_memory(
        self,
        agent_id: str,
        memory_type: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        references: Optional[List[str]] = None
    ) -> Memory:
        """Store a new memory in the graph"""
        
        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            memory_type=MemoryType(memory_type),
            content=content,
            importance=importance,
            timestamp=datetime.now(),
            t_valid=datetime.now(),
            t_invalid=None,
            t_ingested=datetime.now(),
            embedding=self.embedding_model.encode(content),
            metadata=metadata or {},
            references=references or []
        )
        
        # Store in Neo4j
        async with self.driver.session() as session:
            query = """
            MERGE (a:Agent {id: $agent_id})
            CREATE (m:Memory {
                id: $memory_id,
                agent_id: $agent_id,
                memory_type: $memory_type,
                content: $content,
                importance: $importance,
                timestamp: datetime($timestamp),
                t_valid: datetime($t_valid),
                t_ingested: datetime($t_ingested),
                metadata: $metadata,
                embedding: $embedding
            })
            CREATE (a)-[:REMEMBERS {strength: $importance}]->(m)
            RETURN m
            """
            
            await session.run(
                query,
                agent_id=agent_id,
                memory_id=memory.id,
                memory_type=memory.memory_type.value,
                content=content,
                importance=importance,
                timestamp=memory.timestamp.isoformat(),
                t_valid=memory.t_valid.isoformat(),
                t_ingested=memory.t_ingested.isoformat(),
                metadata=json.dumps(metadata or {}),
                embedding=memory.embedding.tolist()
            )
            
            # Create reference relationships
            if references:
                for ref_id in references:
                    ref_query = """
                    MATCH (m:Memory {id: $memory_id})
                    MATCH (e) WHERE e.id = $ref_id
                    CREATE (m)-[:REFERENCES]->(e)
                    """
                    await session.run(
                        ref_query,
                        memory_id=memory.id,
                        ref_id=ref_id
                    )
        
        # Cache recent memory
        if agent_id not in self.memory_cache:
            self.memory_cache[agent_id] = []
        self.memory_cache[agent_id].append(memory)
        if len(self.memory_cache[agent_id]) > 20:
            self.memory_cache[agent_id].pop(0)
        
        return memory
    
    async def retrieve_memories(
        self,
        agent_id: str,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """Retrieve relevant memories using hybrid search"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        async with self.driver.session() as session:
            # Build the cypher query
            cypher = """
            MATCH (a:Agent {id: $agent_id})-[:REMEMBERS]->(m:Memory)
            WHERE m.importance >= $min_importance
            """
            
            params = {
                "agent_id": agent_id,
                "min_importance": min_importance,
                "query_embedding": query_embedding.tolist(),
                "k": k
            }
            
            # Add type filter if specified
            if memory_types:
                type_values = [mt.value for mt in memory_types]
                cypher += " AND m.memory_type IN $memory_types"
                params["memory_types"] = type_values
            
            # Add time range filter if specified
            if time_range:
                cypher += " AND m.timestamp >= datetime($start_time) AND m.timestamp <= datetime($end_time)"
                params["start_time"] = time_range[0].isoformat()
                params["end_time"] = time_range[1].isoformat()
            
            # Calculate similarity and return top k
            cypher += """
            WITH m, a,
                 gds.similarity.cosine(m.embedding, $query_embedding) AS similarity
            ORDER BY similarity DESC, m.importance DESC, m.timestamp DESC
            LIMIT $k
            RETURN m, similarity
            """
            
            result = await session.run(cypher, **params)
            memories = []
            
            async for record in result:
                m = record["m"]
                memory = Memory(
                    id=m["id"],
                    agent_id=m["agent_id"],
                    memory_type=MemoryType(m["memory_type"]),
                    content=m["content"],
                    importance=m["importance"],
                    timestamp=datetime.fromisoformat(str(m["timestamp"])),
                    t_valid=datetime.fromisoformat(str(m["t_valid"])),
                    t_invalid=datetime.fromisoformat(str(m["t_invalid"])) if m.get("t_invalid") else None,
                    t_ingested=datetime.fromisoformat(str(m["t_ingested"])),
                    embedding=np.array(m["embedding"]) if m.get("embedding") else None,
                    metadata=json.loads(m["metadata"]) if m.get("metadata") else {},
                    references=[]
                )
                memories.append(memory)
            
            return memories
    
    async def store_knowledge(
        self,
        agent_id: str,
        fact: str,
        source: str,
        confidence: float = 0.5,
        verified: bool = False
    ) -> str:
        """Store a piece of knowledge (fact, belief, or rumor)"""
        
        knowledge_id = str(uuid.uuid4())
        
        async with self.driver.session() as session:
            query = """
            MERGE (a:Agent {id: $agent_id})
            CREATE (k:Knowledge {
                id: $knowledge_id,
                fact: $fact,
                source: $source,
                confidence: $confidence,
                verified: $verified,
                timestamp: datetime($timestamp)
            })
            CREATE (a)-[:KNOWS {confidence: $confidence}]->(k)
            RETURN k
            """
            
            await session.run(
                query,
                agent_id=agent_id,
                knowledge_id=knowledge_id,
                fact=fact,
                source=source,
                confidence=confidence,
                verified=verified,
                timestamp=datetime.now().isoformat()
            )
        
        # Also store as a memory
        await self.store_memory(
            agent_id=agent_id,
            memory_type="learned" if verified else "rumor",
            content=f"Learned: {fact}",
            importance=confidence,
            metadata={"source": source, "verified": verified}
        )
        
        return knowledge_id
    
    async def update_relationship(
        self,
        agent_a_id: str,
        agent_b_id: str,
        changes: Dict[str, float]
    ):
        """Update relationship metrics between two agents"""
        
        async with self.driver.session() as session:
            # Check if relationship exists
            check_query = """
            MATCH (a:Agent {id: $agent_a_id})
            MATCH (b:Agent {id: $agent_b_id})
            MATCH (a)-[r:RELATIONSHIP]->(b)
            RETURN r
            """
            
            result = await session.run(
                check_query,
                agent_a_id=agent_a_id,
                agent_b_id=agent_b_id
            )
            
            existing = await result.single()
            
            if existing:
                # Update existing relationship
                set_clauses = []
                for metric, value in changes.items():
                    set_clauses.append(f"r.{metric} = COALESCE(r.{metric}, 0) + ${metric}")
                
                update_query = f"""
                MATCH (a:Agent {{id: $agent_a_id}})
                MATCH (b:Agent {{id: $agent_b_id}})
                MATCH (a)-[r:RELATIONSHIP]->(b)
                SET {', '.join(set_clauses)},
                    r.last_interaction = datetime($timestamp),
                    r.interaction_count = COALESCE(r.interaction_count, 0) + 1
                RETURN r
                """
                
                params = {
                    "agent_a_id": agent_a_id,
                    "agent_b_id": agent_b_id,
                    "timestamp": datetime.now().isoformat(),
                    **changes
                }
                
                await session.run(update_query, **params)
            else:
                # Create new relationship
                create_query = """
                MERGE (a:Agent {id: $agent_a_id})
                MERGE (b:Agent {id: $agent_b_id})
                CREATE (a)-[r:RELATIONSHIP {
                    trust: $trust,
                    friendship: $friendship,
                    respect: $respect,
                    familiarity: $familiarity,
                    last_interaction: datetime($timestamp),
                    interaction_count: 1
                }]->(b)
                RETURN r
                """
                
                await session.run(
                    create_query,
                    agent_a_id=agent_a_id,
                    agent_b_id=agent_b_id,
                    trust=changes.get("trust", 0),
                    friendship=changes.get("friendship", 0),
                    respect=changes.get("respect", 0),
                    familiarity=changes.get("familiarity", 0),
                    timestamp=datetime.now().isoformat()
                )
    
    async def reflect_and_compress(
        self,
        agent_id: str,
        max_memories: int = 100
    ) -> str:
        """Compress and reflect on memories to manage growth"""
        
        async with self.driver.session() as session:
            # Get recent memories
            query = """
            MATCH (a:Agent {id: $agent_id})-[:REMEMBERS]->(m:Memory)
            WHERE m.memory_type <> 'reflection'
            RETURN m
            ORDER BY m.timestamp DESC
            LIMIT $limit
            """
            
            result = await session.run(
                query,
                agent_id=agent_id,
                limit=max_memories
            )
            
            memories = []
            async for record in result:
                memories.append(record["m"]["content"])
            
            if len(memories) < 10:
                return "Not enough memories to reflect on"
            
            # Create reflection summary (simplified - would use LLM)
            reflection_content = f"Reflected on {len(memories)} recent experiences. "
            
            # Identify patterns (simplified)
            common_topics = {}
            for memory in memories:
                words = memory.lower().split()
                for word in words:
                    if len(word) > 4:  # Skip short words
                        common_topics[word] = common_topics.get(word, 0) + 1
            
            top_topics = sorted(common_topics.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_topics:
                reflection_content += f"Frequently thinking about: {', '.join([t[0] for t in top_topics])}"
            
            # Store reflection
            await self.store_memory(
                agent_id=agent_id,
                memory_type="reflection",
                content=reflection_content,
                importance=0.8,
                metadata={"memory_count": len(memories)}
            )
            
            # Mark old memories as less important
            decay_query = """
            MATCH (a:Agent {id: $agent_id})-[:REMEMBERS]->(m:Memory)
            WHERE m.timestamp < datetime($cutoff_time)
            AND m.memory_type <> 'reflection'
            SET m.importance = m.importance * 0.9
            """
            
            cutoff = datetime.now() - timedelta(hours=24)
            await session.run(
                decay_query,
                agent_id=agent_id,
                cutoff_time=cutoff.isoformat()
            )
            
            return reflection_content
    
    async def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get a summary of an agent's memories and knowledge"""
        
        async with self.driver.session() as session:
            query = """
            MATCH (a:Agent {id: $agent_id})
            OPTIONAL MATCH (a)-[:REMEMBERS]->(m:Memory)
            OPTIONAL MATCH (a)-[:KNOWS]->(k:Knowledge)
            OPTIONAL MATCH (a)-[:RELATIONSHIP]->(other:Agent)
            RETURN 
                COUNT(DISTINCT m) as memory_count,
                COUNT(DISTINCT k) as knowledge_count,
                COUNT(DISTINCT other) as relationship_count,
                AVG(m.importance) as avg_memory_importance,
                COLLECT(DISTINCT m.memory_type) as memory_types
            """
            
            result = await session.run(query, agent_id=agent_id)
            record = await result.single()
            
            if record:
                return {
                    "memory_count": record["memory_count"],
                    "knowledge_count": record["knowledge_count"],
                    "relationship_count": record["relationship_count"],
                    "avg_memory_importance": record["avg_memory_importance"],
                    "memory_types": record["memory_types"]
                }
            
            return {
                "memory_count": 0,
                "knowledge_count": 0,
                "relationship_count": 0,
                "avg_memory_importance": 0,
                "memory_types": []
            }
    
    async def close(self):
        """Close database connection"""
        if self.driver:
            await self.driver.close()