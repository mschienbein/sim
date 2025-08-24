"""
Graphiti-based memory management with temporal knowledge graphs.
Using Graphiti framework with Neo4j for real-time, incremental memory updates.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import uuid
import json
from enum import Enum

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embeddings import OpenAIEmbedding
from graphiti_core.llm import OpenAIClient
from graphiti_core.search import SearchConfig, SearchMethod

from ..config.settings import settings

class GraphitiMemoryManager:
    """
    Memory management using Graphiti's temporal knowledge graph.
    Provides real-time, incremental updates without batch recomputation.
    """
    
    def __init__(self):
        """Initialize Graphiti with Neo4j backend"""
        self.graphiti = None
        self.initialized = False
        
    async def initialize(self):
        """Set up Graphiti connection and schema"""
        # Initialize Graphiti with Neo4j
        self.graphiti = Graphiti(
            neo4j_uri=settings.neo4j.uri,
            neo4j_user=settings.neo4j.username,
            neo4j_password=settings.neo4j.password,
            neo4j_database=settings.neo4j.database,
            
            # LLM config (can override with Bedrock)
            llm_client=OpenAIClient() if not settings.aws.bedrock_model_id else None,
            embedding_client=OpenAIEmbedding(),
            
            # Graphiti config
            build_indices=True,
            use_reranking=True
        )
        
        await self.graphiti.build_indices()
        self.initialized = True
        
        # Define our simulation-specific node and edge types
        await self._register_simulation_schema()
    
    async def _register_simulation_schema(self):
        """Register custom node and edge types for the simulation"""
        # Custom node types for our simulation
        node_types = [
            "Agent",       # Simulation agents
            "Location",    # Places in the world
            "Item",        # Tradeable items, books, scrolls
            "Memory",      # Event snippets
            "Fact",        # Verified knowledge
            "Rumor",       # Unverified claims with confidence
            "Community",   # Social groups/topics
            "Skill",       # Learned abilities
            "Contract"     # Trade agreements, IOUs
        ]
        
        # Custom edge types with temporal + weight attributes
        edge_types = [
            ("SPOKE_WITH", {"timestamp": "datetime", "sentiment": "float"}),
            ("OBSERVED", {"timestamp": "datetime", "importance": "float"}),
            ("REMEMBERS", {"timestamp": "datetime", "strength": "float"}),
            ("LEARNED_FROM", {"timestamp": "datetime", "confidence": "float"}),
            ("TRUSTS", {"weight": "float", "last_updated": "datetime"}),
            ("OWES", {"amount": "float", "due_date": "datetime"}),
            ("AT_LOCATION", {"since": "datetime", "purpose": "string"}),
            ("TRADED", {"timestamp": "datetime", "items": "json"}),
            ("WROTE", {"timestamp": "datetime", "content_type": "string"}),
            ("READ", {"timestamp": "datetime", "comprehension": "float"}),
            ("BELIEVES", {"confidence": "float", "source": "string"}),
            ("RUMOR_OF", {"confidence": "float", "hops": "int"}),
            ("KNOWS_SKILL", {"level": "float", "learned_date": "datetime"}),
            ("PART_OF", {"role": "string", "joined": "datetime"})
        ]
        
        # Note: Graphiti handles schema dynamically, but we document our types
        self.node_types = node_types
        self.edge_types = edge_types
    
    async def ingest_conversation(
        self,
        agent_a_id: str,
        agent_b_id: str,
        dialogue: List[Dict[str, str]],
        location: str,
        timestamp: Optional[datetime] = None
    ):
        """
        Ingest a conversation between two agents.
        Creates memory nodes and updates relationship edges.
        """
        if not timestamp:
            timestamp = datetime.now()
        
        # Build episode for Graphiti
        episode = {
            "name": f"conversation_{agent_a_id}_{agent_b_id}_{timestamp.isoformat()}",
            "content": json.dumps(dialogue),
            "timestamp": timestamp,
            "source": "conversation",
            "metadata": {
                "participants": [agent_a_id, agent_b_id],
                "location": location,
                "turn_count": len(dialogue)
            }
        }
        
        # Ingest into Graphiti - it will extract entities and relationships
        await self.graphiti.add_episode(
            name=episode["name"],
            episode_body=episode["content"],
            source_description=f"Conversation between {agent_a_id} and {agent_b_id}",
            reference_time=timestamp,
            metadata=episode["metadata"]
        )
        
        # Update SPOKE_WITH relationship
        await self._update_relationship(
            from_id=agent_a_id,
            to_id=agent_b_id,
            edge_type="SPOKE_WITH",
            properties={
                "timestamp": timestamp,
                "sentiment": self._analyze_sentiment(dialogue)
            }
        )
    
    async def ingest_observation(
        self,
        agent_id: str,
        observation: str,
        location: str,
        importance: float = 0.5,
        timestamp: Optional[datetime] = None
    ):
        """
        Ingest an agent's observation of the environment.
        """
        if not timestamp:
            timestamp = datetime.now()
        
        episode = {
            "name": f"observation_{agent_id}_{timestamp.isoformat()}",
            "content": observation,
            "timestamp": timestamp,
            "source": "observation",
            "metadata": {
                "agent": agent_id,
                "location": location,
                "importance": importance
            }
        }
        
        await self.graphiti.add_episode(
            name=episode["name"],
            episode_body=observation,
            source_description=f"{agent_id} observed",
            reference_time=timestamp,
            metadata=episode["metadata"]
        )
    
    async def ingest_trade(
        self,
        from_agent: str,
        to_agent: str,
        given_items: Dict[str, int],
        received_items: Dict[str, int],
        location: str,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a trade transaction between agents.
        """
        if not timestamp:
            timestamp = datetime.now()
        
        trade_record = {
            "from": from_agent,
            "to": to_agent,
            "given": given_items,
            "received": received_items,
            "location": location,
            "timestamp": timestamp
        }
        
        # Create trade episode
        await self.graphiti.add_episode(
            name=f"trade_{from_agent}_{to_agent}_{timestamp.isoformat()}",
            episode_body=json.dumps(trade_record),
            source_description=f"Trade between {from_agent} and {to_agent}",
            reference_time=timestamp,
            metadata=trade_record
        )
        
        # Update TRADED edge
        await self._update_relationship(
            from_id=from_agent,
            to_id=to_agent,
            edge_type="TRADED",
            properties={
                "timestamp": timestamp,
                "items": json.dumps({"given": given_items, "received": received_items})
            }
        )
    
    async def ingest_knowledge(
        self,
        agent_id: str,
        fact: str,
        source: str,
        confidence: float = 0.5,
        verified: bool = False,
        timestamp: Optional[datetime] = None
    ):
        """
        Store learned knowledge as Fact or Rumor node.
        """
        if not timestamp:
            timestamp = datetime.now()
        
        knowledge_type = "Fact" if verified else "Rumor"
        
        # Create knowledge node via episode
        await self.graphiti.add_episode(
            name=f"{knowledge_type.lower()}_{agent_id}_{timestamp.isoformat()}",
            episode_body=fact,
            source_description=f"Knowledge from {source}",
            reference_time=timestamp,
            metadata={
                "agent": agent_id,
                "type": knowledge_type,
                "source": source,
                "confidence": confidence,
                "verified": verified
            }
        )
        
        # Create BELIEVES or KNOWS edge
        edge_type = "KNOWS" if verified else "BELIEVES"
        await self._update_relationship(
            from_id=agent_id,
            to_id=f"{knowledge_type}_{fact[:50]}",  # Truncated fact as ID
            edge_type=edge_type,
            properties={
                "confidence": confidence,
                "source": source,
                "timestamp": timestamp
            }
        )
    
    async def propagate_rumor(
        self,
        from_agent: str,
        to_agent: str,
        rumor: str,
        confidence_decay: float = 0.9,
        timestamp: Optional[datetime] = None
    ):
        """
        Propagate a rumor from one agent to another with confidence decay.
        """
        if not timestamp:
            timestamp = datetime.now()
        
        # Get current rumor confidence if it exists
        existing = await self.search_temporal(
            query=rumor,
            agent_id=from_agent,
            memory_types=["Rumor"],
            limit=1
        )
        
        base_confidence = existing[0].get("confidence", 0.5) if existing else 0.5
        new_confidence = base_confidence * confidence_decay
        
        # Ingest rumor for receiving agent
        await self.ingest_knowledge(
            agent_id=to_agent,
            fact=rumor,
            source=f"heard_from_{from_agent}",
            confidence=new_confidence,
            verified=False,
            timestamp=timestamp
        )
        
        # Track rumor propagation
        await self._update_relationship(
            from_id=from_agent,
            to_id=to_agent,
            edge_type="RUMOR_OF",
            properties={
                "confidence": new_confidence,
                "hops": existing[0].get("hops", 0) + 1 if existing else 1,
                "timestamp": timestamp
            }
        )
    
    async def search_temporal(
        self,
        query: str,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Temporal + semantic search for memories/facts.
        Uses Graphiti's fusion of temporal, full-text, semantic, and graph queries.
        """
        search_config = SearchConfig(
            query=query,
            limit=limit,
            search_methods=[
                SearchMethod.SEMANTIC,
                SearchMethod.FULL_TEXT,
                SearchMethod.GRAPH_TRAVERSAL
            ],
            rerank=True
        )
        
        # Add temporal filters
        filters = {}
        if since:
            filters["timestamp_gte"] = since.isoformat()
        if until:
            filters["timestamp_lte"] = until.isoformat()
        if agent_id:
            filters["agent"] = agent_id
        if memory_types:
            filters["type"] = {"$in": memory_types}
        
        # Execute search via Graphiti
        results = await self.graphiti.search(
            query=query,
            config=search_config,
            filters=filters
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.content,
                "timestamp": result.created_at,
                "importance": result.metadata.get("importance", 0.5),
                "type": result.metadata.get("type", "Memory"),
                "confidence": result.metadata.get("confidence", 1.0),
                "source": result.metadata.get("source", "unknown"),
                "relevance_score": result.relevance_score
            })
        
        return formatted_results
    
    async def get_related_entities(
        self,
        entity_id: str,
        hop_count: int = 2,
        edge_types: Optional[List[str]] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop graph traversal to find related entities.
        Perfect for social networks and rumor propagation tracking.
        """
        # Use Graphiti's graph traversal
        results = await self.graphiti.traverse(
            start_node=entity_id,
            max_hops=hop_count,
            edge_filters=edge_types,
            node_filters=filters
        )
        
        return [
            {
                "id": node.uuid,
                "name": node.name,
                "type": node.labels[0] if node.labels else "Entity",
                "distance": result.distance,
                "path": result.path
            }
            for result in results
            for node in [result.end_node]
        ]
    
    async def get_agent_context(
        self,
        agent_id: str,
        location: str,
        nearby_agents: List[str],
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for an agent's turn.
        Combines temporal memories, relationships, location facts, and goals.
        """
        context = {
            "recent_memories": [],
            "relationships": {},
            "location_facts": [],
            "active_rumors": [],
            "personal_goals": [],
            "pending_contracts": []
        }
        
        # Recent memories (last 24 hours)
        since = datetime.now() - timedelta(hours=24)
        context["recent_memories"] = await self.search_temporal(
            query=f"agent {agent_id} recent events",
            agent_id=agent_id,
            since=since,
            limit=10
        )
        
        # Relationships with nearby agents
        for other_agent in nearby_agents:
            rel = await self._get_relationship_strength(agent_id, other_agent)
            context["relationships"][other_agent] = rel
        
        # Location-specific facts
        context["location_facts"] = await self.search_temporal(
            query=f"location {location}",
            memory_types=["Fact", "Observation"],
            limit=5
        )
        
        # Active rumors the agent believes
        context["active_rumors"] = await self.search_temporal(
            query="rumors gossip news",
            agent_id=agent_id,
            memory_types=["Rumor"],
            limit=5
        )
        
        # Personal goals (stored as special facts)
        context["personal_goals"] = await self.search_temporal(
            query=f"goals objectives plans",
            agent_id=agent_id,
            memory_types=["Fact"],
            limit=3
        )
        
        return context
    
    async def reflect_and_compress(
        self,
        agent_id: str,
        reflection_prompt: Optional[str] = None
    ) -> str:
        """
        Nightly reflection: summarize day's events into higher-level facts.
        Uses Graphiti's incremental update capability.
        """
        # Get today's memories
        since = datetime.now() - timedelta(hours=24)
        recent_memories = await self.search_temporal(
            query=f"all events",
            agent_id=agent_id,
            since=since,
            limit=50
        )
        
        if len(recent_memories) < 5:
            return "Not enough events to reflect upon."
        
        # Create reflection episode
        reflection = f"Daily reflection for {agent_id}: "
        reflection += f"Experienced {len(recent_memories)} events. "
        
        # Extract themes (simplified - would use LLM)
        themes = self._extract_themes(recent_memories)
        reflection += f"Key themes: {', '.join(themes)}. "
        
        # Store reflection as high-level fact
        await self.graphiti.add_episode(
            name=f"reflection_{agent_id}_{datetime.now().isoformat()}",
            episode_body=reflection,
            source_description=f"{agent_id}'s daily reflection",
            reference_time=datetime.now(),
            metadata={
                "agent": agent_id,
                "type": "Reflection",
                "memory_count": len(recent_memories),
                "themes": themes
            }
        )
        
        # Prune low-importance memories (Graphiti handles this efficiently)
        # The framework automatically manages memory importance decay
        
        return reflection
    
    async def _update_relationship(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        properties: Dict[str, Any]
    ):
        """Update or create an edge between entities"""
        # Graphiti handles incremental edge updates automatically
        edge = EntityEdge(
            source_node_uuid=from_id,
            target_node_uuid=to_id,
            name=edge_type,
            fact=json.dumps(properties),
            created_at=properties.get("timestamp", datetime.now()),
            metadata=properties
        )
        
        await self.graphiti.add_edge(edge)
    
    async def _get_relationship_strength(
        self,
        agent_a: str,
        agent_b: str
    ) -> Dict[str, float]:
        """Get relationship metrics between two agents"""
        # Query for edges between agents
        edges = await self.graphiti.get_edges_between(agent_a, agent_b)
        
        relationship = {
            "trust": 0.0,
            "friendship": 0.0,
            "familiarity": 0.0,
            "last_interaction": None
        }
        
        for edge in edges:
            if edge.name == "TRUSTS":
                relationship["trust"] = edge.metadata.get("weight", 0.0)
            elif edge.name == "SPOKE_WITH":
                relationship["familiarity"] += 0.1
                relationship["last_interaction"] = edge.created_at
        
        return relationship
    
    def _analyze_sentiment(self, dialogue: List[Dict[str, str]]) -> float:
        """Simple sentiment analysis (would use LLM in production)"""
        positive_words = {"happy", "good", "great", "wonderful", "thank", "please", "friend"}
        negative_words = {"angry", "bad", "terrible", "hate", "dislike", "enemy"}
        
        text = " ".join([turn["message"] for turn in dialogue])
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_themes(self, memories: List[Dict]) -> List[str]:
        """Extract themes from memories (simplified)"""
        # In production, use LLM or clustering
        word_freq = {}
        for memory in memories:
            words = memory["content"].lower().split()
            for word in words:
                if len(word) > 5:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top themes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:3]]
    
    async def close(self):
        """Clean up Graphiti connection"""
        if self.graphiti:
            await self.graphiti.close()