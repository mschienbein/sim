"""
Graphiti-based memory management with temporal knowledge graphs.
Using Graphiti framework with Neo4j for real-time, incremental memory updates.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import json
import logging

logger = logging.getLogger(__name__)

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embeddings import OpenAIEmbedding
from graphiti_core.llm import OpenAIClient
from graphiti_core.search import SearchConfig, SearchMethod
from neo4j import AsyncGraphDatabase

from src.config.settings import settings

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
            # Core Entities
            "Agent",           # Simulation agents with personalities
            "Location",        # Places in the world (market, temple, etc.)
            "Building",        # Specific structures (houses, shops, library)
            
            # Items & Objects
            "Item",            # Physical objects that can be owned/traded
            "Tool",            # Items with specific uses (hammer, quill, etc.)
            "Food",            # Consumable items for energy
            "Book",            # Written works that contain knowledge
            "Scroll",          # Written messages or knowledge artifacts
            "Artifact",        # Special/magical items with unique properties
            "Currency",        # Money or tokens of value
            
            # Knowledge & Information
            "Memory",          # Event snippets and experiences
            "Fact",            # Verified knowledge
            "Rumor",           # Unverified claims with confidence scores
            "Secret",          # Hidden knowledge known to few
            "Recipe",          # Instructions for creating/doing something
            "Story",           # Narratives and tales
            "Prophecy",        # Future predictions or warnings
            "Law",             # Rules and regulations
            
            # Social Structures
            "Community",       # Social groups/topics
            "Guild",           # Professional organizations
            "Family",          # Family units and lineages
            "Faction",         # Political or ideological groups
            "Relationship",    # Explicit relationship records
            
            # Skills & Abilities
            "Skill",           # Learned abilities (cooking, smithing, etc.)
            "Talent",          # Natural aptitudes
            "Profession",      # Job roles and careers
            "Title",           # Earned or bestowed honors
            
            # Economic & Trade
            "Contract",        # Trade agreements, IOUs
            "Service",         # Services that can be provided
            "Quest",           # Tasks or missions to complete
            "Debt",            # Money owed between parties
            "Shop",            # Commercial establishments
            "Market",          # Trading venues
            
            # Events & Activities
            "Event",           # Significant occurrences
            "Festival",        # Celebrations and gatherings
            "Ritual",          # Religious or cultural ceremonies
            "Meeting",         # Planned gatherings
            "Conflict",        # Disputes and fights
            "Achievement",     # Accomplishments and milestones
            
            # Environmental
            "Weather",         # Weather conditions and patterns
            "Season",          # Time periods (spring, harvest, etc.)
            "Resource",        # Natural resources (wood, stone, water)
            "Landmark",        # Notable geographic features
            
            # Emotional & Mental States
            "Emotion",         # Emotional states and feelings
            "Mood",            # Longer-term emotional conditions
            "Dream",           # Dreams and visions
            "Fear",            # Specific fears and phobias
            "Desire",          # Wants and goals
            "Belief",          # Core beliefs and values
            
            # Health & Status
            "Illness",         # Diseases and ailments
            "Injury",          # Physical wounds
            "Blessing",        # Positive supernatural effects
            "Curse",           # Negative supernatural effects
            "Status",          # Social standing and reputation
            
            # Communication
            "Message",         # Communications between agents
            "Announcement",    # Public declarations
            "Gossip",          # Informal information spreading
            "Letter",          # Written correspondence
            "Song",            # Musical compositions
            "Poem",            # Poetic works
            
            # Relationships & Connections (for edge consistency)
            "Trust",           # Trust relationships
            "Friendship",      # Friend connections
            "Romance",         # Romantic relationships
            "Rivalry",         # Competitive relationships
            "Mentorship",      # Teaching relationships
            "Alliance",        # Cooperative bonds
            "Trade",           # Commercial relationships
            "Employment"       # Work relationships
        ]
        
        # Custom edge types with temporal + weight attributes
        edge_types = [
            # Social Interactions
            ("SPOKE_WITH", {"timestamp": "datetime", "sentiment": "float"}),
            ("OBSERVED", {"timestamp": "datetime", "importance": "float"}),
            ("GREETED", {"timestamp": "datetime", "warmth": "float"}),
            ("ARGUED_WITH", {"timestamp": "datetime", "intensity": "float", "topic": "string"}),
            ("HELPED", {"timestamp": "datetime", "task": "string", "gratitude": "float"}),
            ("TAUGHT", {"timestamp": "datetime", "subject": "string", "effectiveness": "float"}),
            ("COLLABORATED_WITH", {"timestamp": "datetime", "project": "string", "harmony": "float"}),
            
            # Emotional Relationships
            ("LIKES", {"intensity": "float", "since": "datetime", "reason": "string"}),
            ("DISLIKES", {"intensity": "float", "since": "datetime", "reason": "string"}),
            ("LOVES", {"intensity": "float", "since": "datetime", "type": "string"}),  # romantic, platonic, familial
            ("FEARS", {"intensity": "float", "since": "datetime", "trigger": "string"}),
            ("ADMIRES", {"aspect": "string", "intensity": "float", "since": "datetime"}),
            ("ENVIES", {"reason": "string", "intensity": "float", "since": "datetime"}),
            ("RESPECTS", {"level": "float", "reason": "string", "since": "datetime"}),
            ("FEELS_SORRY_FOR", {"reason": "string", "intensity": "float", "timestamp": "datetime"}),
            
            # Trust & Social Standing
            ("TRUSTS", {"weight": "float", "last_updated": "datetime"}),
            ("DISTRUSTS", {"reason": "string", "intensity": "float", "since": "datetime"}),
            ("BETRAYED", {"timestamp": "datetime", "severity": "float", "forgiven": "boolean"}),
            ("FORGAVE", {"timestamp": "datetime", "offense": "string"}),
            ("ALLIED_WITH", {"strength": "float", "purpose": "string", "since": "datetime"}),
            ("RIVALS_WITH", {"domain": "string", "intensity": "float", "since": "datetime"}),
            
            # Memory & Knowledge
            ("REMEMBERS", {"timestamp": "datetime", "strength": "float", "emotion": "string"}),
            ("LEARNED_FROM", {"timestamp": "datetime", "confidence": "float"}),
            ("FORGOT", {"timestamp": "datetime", "importance": "float"}),
            ("REMINDS_OF", {"similarity": "float", "emotional_weight": "float"}),
            ("CONTRADICTS", {"confidence": "float", "timestamp": "datetime"}),
            
            # Economic & Trade
            ("OWES", {"amount": "float", "due_date": "datetime", "currency": "string"}),
            ("TRADED", {"timestamp": "datetime", "items": "json", "satisfaction": "float"}),
            ("GIFTED", {"timestamp": "datetime", "item": "string", "occasion": "string"}),
            ("STOLE_FROM", {"timestamp": "datetime", "item": "string", "discovered": "boolean"}),
            ("EMPLOYED_BY", {"since": "datetime", "role": "string", "satisfaction": "float"}),
            ("COMMISSIONED", {"timestamp": "datetime", "task": "string", "payment": "float"}),
            
            # Activities & Actions
            ("AT_LOCATION", {"since": "datetime", "purpose": "string", "mood": "string"}),
            ("TRAVELED_TO", {"timestamp": "datetime", "from": "string", "reason": "string"}),
            ("AVOIDED", {"timestamp": "datetime", "reason": "string"}),
            ("VISITED", {"timestamp": "datetime", "duration": "float", "purpose": "string"}),
            ("PERFORMED_FOR", {"timestamp": "datetime", "type": "string", "reception": "float"}),
            ("COMPETED_WITH", {"timestamp": "datetime", "contest": "string", "outcome": "string"}),
            
            # Information & Communication
            ("WROTE", {"timestamp": "datetime", "content_type": "string", "length": "int"}),
            ("READ", {"timestamp": "datetime", "comprehension": "float", "enjoyment": "float"}),
            ("HEARD_FROM", {"timestamp": "datetime", "credibility": "float", "topic": "string"}),
            ("GOSSIPED_ABOUT", {"timestamp": "datetime", "topic": "string", "malicious": "boolean"}),
            ("SHARED_SECRET", {"timestamp": "datetime", "trust_level": "float"}),
            ("LIED_TO", {"timestamp": "datetime", "topic": "string", "discovered": "boolean"}),
            
            # Beliefs & Opinions
            ("BELIEVES", {"confidence": "float", "source": "string", "since": "datetime"}),
            ("DOUBTS", {"reason": "string", "strength": "float", "since": "datetime"}),
            ("AGREES_WITH", {"topic": "string", "strength": "float", "timestamp": "datetime"}),
            ("DISAGREES_WITH", {"topic": "string", "strength": "float", "timestamp": "datetime"}),
            ("RUMOR_OF", {"confidence": "float", "hops": "int", "timestamp": "datetime"}),
            
            # Skills & Development
            ("KNOWS_SKILL", {"level": "float", "learned_date": "datetime", "teacher": "string"}),
            ("MENTORED_BY", {"since": "datetime", "domain": "string", "progress": "float"}),
            ("INSPIRED_BY", {"timestamp": "datetime", "aspect": "string", "impact": "float"}),
            ("LEARNED_RECIPE", {"timestamp": "datetime", "dish": "string", "mastery": "float"}),
            
            # Community & Groups
            ("PART_OF", {"role": "string", "joined": "datetime", "status": "string"}),
            ("LEADS", {"since": "datetime", "approval": "float", "style": "string"}),
            ("FOLLOWS", {"since": "datetime", "loyalty": "float", "reason": "string"}),
            ("BANISHED_FROM", {"timestamp": "datetime", "reason": "string", "duration": "float"}),
            ("WELCOMED_BY", {"timestamp": "datetime", "warmth": "float"}),
            
            # Conflicts & Resolutions
            ("FOUGHT_WITH", {"timestamp": "datetime", "reason": "string", "outcome": "string"}),
            ("MADE_PEACE_WITH", {"timestamp": "datetime", "mediator": "string"}),
            ("CHALLENGED", {"timestamp": "datetime", "type": "string", "accepted": "boolean"}),
            ("DEFENDED", {"timestamp": "datetime", "from": "string", "success": "boolean"}),
            
            # Life Events
            ("WITNESSED", {"timestamp": "datetime", "event": "string", "impact": "float"}),
            ("CELEBRATED_WITH", {"timestamp": "datetime", "occasion": "string", "joy": "float"}),
            ("MOURNED_WITH", {"timestamp": "datetime", "loss": "string", "grief": "float"}),
            ("BLESSED_BY", {"timestamp": "datetime", "type": "string", "power": "float"}),
            ("CURSED_BY", {"timestamp": "datetime", "type": "string", "severity": "float"})
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
        await self.create_relationship_edge(
            from_agent_id=agent_a_id,
            to_agent_id=agent_b_id,
            edge_type="SPOKE_WITH",
            attributes={
                "timestamp": timestamp.isoformat(),
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
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type="TRADED",
            attributes={
                "timestamp": timestamp.isoformat(),
                "items": json.dumps({"given": given_items, "received": received_items}),
                "satisfaction": 0.8  # Default satisfaction
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
        
        # Store the knowledge as an entity node first
        knowledge_node = await self.create_entity_node(
            node_type=knowledge_type,
            node_id=str(uuid.uuid4()),
            name=fact[:100],  # Truncated for name
            properties={
                "fact": fact,
                "confidence": confidence,
                "source": source,
                "verified": verified,
                "timestamp": timestamp.isoformat()
            }
        )
        
        # Create BELIEVES or KNOWS edge from agent to knowledge
        edge_type = "KNOWS" if verified else "BELIEVES"
        await self.create_relationship_edge(
            from_agent_id=agent_id,
            to_agent_id=knowledge_node.properties["id"],
            edge_type=edge_type,
            attributes={
                "confidence": confidence,
                "source": source,
                "since": timestamp.isoformat()
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
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type="HEARD_FROM",
            attributes={
                "timestamp": timestamp.isoformat(),
                "credibility": new_confidence,
                "topic": rumor[:100]
            }
        )
        
        # Also track the rumor belief
        await self.create_relationship_edge(
            from_agent_id=to_agent,
            to_agent_id=from_agent,
            edge_type="RUMOR_OF",
            attributes={
                "confidence": new_confidence,
                "hops": existing[0].get("hops", 0) + 1 if existing else 1,
                "timestamp": timestamp.isoformat()
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
    
    async def create_relationship_edge(
        self,
        from_agent_id: str,
        to_agent_id: str,
        edge_type: str,
        attributes: Dict[str, Any],
        bidirectional: bool = False
    ):
        """
        Create or update a specific relationship edge between agents.
        Uses Neo4j directly for custom edge types.
        """
        async with AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.username, settings.neo4j.password)
        ).session() as session:
            
            # Ensure both agents exist as nodes
            await session.run(
                "MERGE (a:Agent {id: $from_id}) "
                "MERGE (b:Agent {id: $to_id})",
                from_id=from_agent_id,
                to_id=to_agent_id
            )
            
            # Create the relationship with properties
            query = f"""
            MATCH (a:Agent {{id: $from_id}})
            MATCH (b:Agent {{id: $to_id}})
            MERGE (a)-[r:{edge_type}]->(b)
            SET r += $properties
            RETURN r
            """
            
            await session.run(
                query,
                from_id=from_agent_id,
                to_id=to_agent_id,
                properties=attributes
            )
            
            # Create reverse relationship if bidirectional
            if bidirectional:
                reverse_query = f"""
                MATCH (a:Agent {{id: $to_id}})
                MATCH (b:Agent {{id: $from_id}})
                MERGE (a)-[r:{edge_type}]->(b)
                SET r += $properties
                RETURN r
                """
                
                await session.run(
                    reverse_query,
                    from_id=from_agent_id,
                    to_id=to_agent_id,
                    properties=attributes
                )
    
    async def create_entity_node(
        self,
        node_type: str,
        node_id: str,
        name: str,
        properties: Dict[str, Any]
    ) -> EntityNode:
        """
        Create a custom entity node (Agent, Location, Item, etc.)
        """
        node = EntityNode(
            name=name,
            node_type=node_type,
            properties={
                "id": node_id,
                **properties,
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Store in Neo4j via Graphiti
        await self.graphiti.add_node(node)
        return node
    
    async def create_episodic_node(
        self,
        content: str,
        timestamp: datetime,
        agent_id: str,
        node_type: str = "Memory",
        metadata: Optional[Dict[str, Any]] = None
    ) -> EpisodicNode:
        """
        Create an episodic memory node.
        """
        node = EpisodicNode(
            content=content,
            timestamp=timestamp,
            source_id=agent_id,
            metadata={
                "node_type": node_type,
                **(metadata or {})
            }
        )
        
        # Store via Graphiti
        await self.graphiti.add_node(node)
        return node
    
    async def create_entity_edge(
        self,
        from_node: EntityNode,
        to_node: EntityNode,
        edge_type: str,
        properties: Dict[str, Any]
    ) -> EntityEdge:
        """
        Create an edge between entity nodes.
        """
        edge = EntityEdge(
            source=from_node,
            target=to_node,
            edge_type=edge_type,
            properties=properties
        )
        
        await self.graphiti.add_edge(edge)
        return edge
    
    async def create_episodic_edge(
        self,
        from_node: EpisodicNode,
        to_node: EntityNode,
        edge_type: str,
        properties: Dict[str, Any]
    ) -> EpisodicEdge:
        """
        Create an edge from episodic memory to entity.
        """
        edge = EpisodicEdge(
            source=from_node,
            target=to_node,
            edge_type=edge_type,
            properties=properties
        )
        
        await self.graphiti.add_edge(edge)
        return edge
    
    # Specific relationship creation methods
    async def create_likes_relationship(
        self,
        from_agent: str,
        to_agent: str,
        intensity: float,
        reason: str
    ):
        """Create a LIKES relationship."""
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type="LIKES",
            attributes={
                "intensity": intensity,
                "since": datetime.now().isoformat(),
                "reason": reason
            }
        )
    
    async def create_trust_relationship(
        self,
        from_agent: str,
        to_agent: str,
        weight: float
    ):
        """Create or update TRUSTS relationship."""
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type="TRUSTS",
            attributes={
                "weight": weight,
                "last_updated": datetime.now().isoformat()
            }
        )
    
    async def create_trade_relationship(
        self,
        from_agent: str,
        to_agent: str,
        items: Dict[str, Any],
        satisfaction: float
    ):
        """Record a trade between agents."""
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type="TRADED",
            attributes={
                "timestamp": datetime.now().isoformat(),
                "items": json.dumps(items),
                "satisfaction": satisfaction
            }
        )
    
    async def create_conflict_relationship(
        self,
        agent_a: str,
        agent_b: str,
        conflict_type: str,
        intensity: float,
        topic: str
    ):
        """Create conflict relationship (ARGUED_WITH, FOUGHT_WITH, etc.)"""
        edge_type = "ARGUED_WITH" if conflict_type == "verbal" else "FOUGHT_WITH"
        
        await self.create_relationship_edge(
            from_agent_id=agent_a,
            to_agent_id=agent_b,
            edge_type=edge_type,
            attributes={
                "timestamp": datetime.now().isoformat(),
                "intensity": intensity,
                "topic": topic
            },
            bidirectional=True  # Conflicts are usually mutual
        )
    
    async def create_emotional_bond(
        self,
        from_agent: str,
        to_agent: str,
        emotion_type: str,
        intensity: float,
        reason: str
    ):
        """Create emotional relationships (LOVES, FEARS, ADMIRES, ENVIES, etc.)"""
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type=emotion_type.upper(),
            attributes={
                "intensity": intensity,
                "since": datetime.now().isoformat(),
                "reason": reason,
                "type": "emotional"
            }
        )
    
    async def create_social_interaction(
        self,
        from_agent: str,
        to_agent: str,
        interaction_type: str,
        metadata: Dict[str, Any]
    ):
        """Create social interaction edges (HELPED, TAUGHT, COLLABORATED_WITH, etc.)"""
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type=interaction_type.upper(),
            attributes={
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
        )
    
    async def update_relationship_strength(
        self,
        from_agent: str,
        to_agent: str,
        edge_type: str,
        delta: float
    ):
        """Update the strength/intensity of an existing relationship."""
        # First get the current relationship
        async with AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.username, settings.neo4j.password)
        ).session() as session:
            
            # Get current value
            query = f"""
            MATCH (a:Agent {{id: $from_id}})-[r:{edge_type}]->(b:Agent {{id: $to_id}})
            RETURN r.intensity as intensity, r.weight as weight
            """
            
            result = await session.run(
                query,
                from_id=from_agent,
                to_id=to_agent
            )
            
            record = await result.single()
            current_value = 0.0
            value_field = "intensity"
            
            if record:
                current_value = record.get("intensity") or record.get("weight") or 0.0
                value_field = "intensity" if record.get("intensity") is not None else "weight"
            
            # Update with new value
            new_value = max(0.0, min(1.0, current_value + delta))  # Clamp between 0 and 1
            
            update_query = f"""
            MATCH (a:Agent {{id: $from_id}})
            MATCH (b:Agent {{id: $to_id}})
            MERGE (a)-[r:{edge_type}]->(b)
            SET r.{value_field} = $new_value,
                r.last_updated = $timestamp
            RETURN r
            """
            
            await session.run(
                update_query,
                from_id=from_agent,
                to_id=to_agent,
                new_value=new_value,
                timestamp=datetime.now().isoformat()
            )
    
    # Convenience methods for common relationships
    async def agent_likes(
        self,
        from_agent: str,
        to_agent: str,
        intensity: float = 0.5,
        reason: str = "general affinity"
    ):
        """Agent develops positive feelings."""
        await self.create_likes_relationship(from_agent, to_agent, intensity, reason)
    
    async def agent_dislikes(
        self,
        from_agent: str,
        to_agent: str,
        intensity: float = 0.5,
        reason: str = "personality clash"
    ):
        """Agent develops negative feelings."""
        await self.create_relationship_edge(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            edge_type="DISLIKES",
            attributes={
                "intensity": intensity,
                "since": datetime.now().isoformat(),
                "reason": reason
            }
        )
    
    async def agent_helps(
        self,
        helper: str,
        helped: str,
        task: str,
        gratitude: float = 0.7
    ):
        """Record one agent helping another."""
        await self.create_social_interaction(
            from_agent=helper,
            to_agent=helped,
            interaction_type="HELPED",
            metadata={
                "task": task,
                "gratitude": gratitude
            }
        )
        
        # Helping often increases liking
        await self.update_relationship_strength(helped, helper, "LIKES", 0.1)
        await self.update_relationship_strength(helped, helper, "TRUSTS", 0.05)
    
    async def agent_teaches(
        self,
        teacher: str,
        student: str,
        subject: str,
        effectiveness: float = 0.7
    ):
        """Record teaching interaction."""
        await self.create_social_interaction(
            from_agent=teacher,
            to_agent=student,
            interaction_type="TAUGHT",
            metadata={
                "subject": subject,
                "effectiveness": effectiveness
            }
        )
        
        # Teaching creates mentorship bond
        await self.create_relationship_edge(
            from_agent_id=student,
            to_agent_id=teacher,
            edge_type="MENTORED_BY",
            attributes={
                "since": datetime.now().isoformat(),
                "domain": subject,
                "progress": effectiveness
            }
        )
    
    async def agent_argues(
        self,
        agent_a: str,
        agent_b: str,
        topic: str,
        intensity: float = 0.6
    ):
        """Record an argument between agents."""
        await self.create_conflict_relationship(
            agent_a=agent_a,
            agent_b=agent_b,
            conflict_type="verbal",
            intensity=intensity,
            topic=topic
        )
        
        # Arguments often decrease liking and trust
        await self.update_relationship_strength(agent_a, agent_b, "LIKES", -intensity * 0.2)
        await self.update_relationship_strength(agent_b, agent_a, "LIKES", -intensity * 0.2)
        await self.update_relationship_strength(agent_a, agent_b, "TRUSTS", -intensity * 0.1)
        await self.update_relationship_strength(agent_b, agent_a, "TRUSTS", -intensity * 0.1)
    
    async def agent_celebrates_with(
        self,
        agent_a: str,
        agent_b: str,
        occasion: str,
        joy: float = 0.8
    ):
        """Record shared celebration."""
        await self.create_relationship_edge(
            from_agent_id=agent_a,
            to_agent_id=agent_b,
            edge_type="CELEBRATED_WITH",
            attributes={
                "timestamp": datetime.now().isoformat(),
                "occasion": occasion,
                "joy": joy
            },
            bidirectional=True
        )
        
        # Celebrations increase bonds
        await self.update_relationship_strength(agent_a, agent_b, "LIKES", joy * 0.1)
        await self.update_relationship_strength(agent_b, agent_a, "LIKES", joy * 0.1)
    
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