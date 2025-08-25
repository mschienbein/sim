# LLM Agent Simulation Architecture

## Executive Summary

A multi-agent simulation framework where autonomous LLM-powered agents develop distinct personalities through social interactions, memory formation, and knowledge propagation in a spatially-bounded virtual world. The system leverages OpenAI GPT-5 as the primary intelligence engine, Neo4j/Graphiti for temporal memory graphs, and sophisticated personality modeling to create emergent social behaviors.

### Key Technologies
- **OpenAI GPT-5**: Primary LLM for agent cognition ($0.005/1k tokens)
- **Strands Agents SDK**: Model-driven agent framework with native tool support
- **Neo4j + Graphiti**: Bi-temporal knowledge graphs with 300ms P95 latency
- **Python 3.10+**: Modern async/await patterns for concurrent processing
- **NumPy**: Optimized spatial computations for world grid

## 1. Core Architecture Components

### 1.1 LLM Infrastructure

#### Model Hierarchy & Selection
Dynamic model selection based on action priority and budget:

```python
MODEL_COSTS = {
    "gpt-5": 0.005,        # Primary: Rich reasoning, creativity
    "gpt-4o": 0.0025,      # Fallback: Balanced performance
    "gpt-4o-mini": 0.00015 # Emergency: Basic operations
}

BUDGET_THRESHOLDS = {
    "high_priority": 0.2,   # Reflect, web_search
    "medium_priority": 0.6, # Speak, trade
    "low_priority": 0.8     # Move, observe
}
```

**Token Budget Management:**
- Daily limit: 1,000,000 tokens (~$5/day with GPT-5)
- Per-agent limit: 200,000 tokens
- Automatic model downgrade as budget depletes
- Smart prompt truncation for cost optimization

#### Strands Framework Integration
The Strands SDK provides elegant agent abstraction with minimal boilerplate:

```python
from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={"api_key": os.getenv("OPENAI_API_KEY")},
    model_id="gpt-5",
    params={"temperature": 0.7, "max_tokens": 500}
)

agent = Agent(
    name="John",
    model=model,
    tools=[move, speak, trade, reflect],
    system_prompt=build_personality_prompt(traits)
)
```

### 1.2 Memory Architecture (Neo4j/Graphiti)

#### Bi-Temporal Knowledge Graph
Revolutionary temporal tracking for agent memories:

```cypher
(:Memory {
    id: string,
    content: string,
    importance: float,
    t_valid: datetime,     // When event occurred
    t_invalid: datetime,   // When fact became invalid  
    t_ingested: datetime,  // When learned
    embedding: vector[1536]
})
```

#### Node Taxonomy (66 Types)
Comprehensive ontology for rich world representation:

**Core Entities (9):**
- Agent, Location, Item, Object, Entity, Place, Building, Area, Zone

**Items & Objects (12):**
- Tool, Weapon, Armor, Food, Drink, Material, Container, Currency, Document, Artifact, Resource, Commodity

**Knowledge & Information (8):**
- Fact, Rumor, Secret, Belief, Theory, Discovery, Prophecy, Legend

**Social Structures (7):**
- Group, Organization, Guild, Family, Faction, Alliance, Community

**Skills & Abilities (6):**
- Skill, Ability, Craft, Art, Magic, Combat

**Economic & Trade (5):**
- Trade, Market, Contract, Debt, Investment

**Events & Activities (8):**
- Event, Quest, Task, Ceremony, Festival, Battle, Journey, Meeting

**Environmental (5):**
- Weather, Season, Climate, Terrain, Ecosystem

**And 11 more categories...**

#### Edge Relationships (57 Types)
Rich semantic connections between entities:

**Social (15):** SPOKE_WITH, LIKES, DISLIKES, TRUSTS, DISTRUSTS, FRIENDS_WITH, RIVALS_WITH, LOVES, HATES, RESPECTS, FEARS, ADMIRES, FOLLOWS, BETRAYED, HELPED

**Knowledge (8):** KNOWS, LEARNED_FROM, TAUGHT_TO, BELIEVES, DOUBTS, DISCOVERED, INVENTED, RESEARCHED

**Economic (7):** OWNS, TRADED, SOLD_TO, BOUGHT_FROM, BORROWED, LENT_TO, CONTRACTED

**Spatial (5):** LOCATED_AT, TRAVELED_TO, LIVES_IN, WORKS_AT, ORIGINATED_FROM

**And 22 more types...**

### 1.3 System Architecture Layers

```
┌──────────────────────────────────────────────────────────┐
│                  Simulation Engine                        │
│    (Orchestration, Time Management, Event Processing)     │
├──────────────────────────────────────────────────────────┤
│               Agent Communication Layer                   │
│         (Strands Tools, Conversation Manager)             │
├──────────────────────────────────────────────────────────┤
│                    Agent Cognition                        │
│      (GPT-5 Reasoning, Personality, Decision-Making)      │
├──────────────────────────────────────────────────────────┤
│                 Temporal Memory System                    │
│        (Neo4j/Graphiti Bi-Temporal Knowledge Graph)       │
├──────────────────────────────────────────────────────────┤
│                  World Environment                        │
│    (10x10 Grid, NumPy Spatial Ops, A* Pathfinding)       │
├──────────────────────────────────────────────────────────┤
│              Observability & Cost Control                 │
│     (Token Tracking, Grafana Metrics, Rate Limiting)      │
└──────────────────────────────────────────────────────────┘
```

## 2. Agent Design & Personality System

### 2.1 Big Five Personality Model

Scientific personality framework driving agent behavior:

```python
class Personality:
    openness: float         # 0.0-1.0: Curiosity, creativity
    conscientiousness: float # 0.0-1.0: Organization, discipline  
    extraversion: float     # 0.0-1.0: Social energy
    agreeableness: float    # 0.0-1.0: Cooperation, empathy
    neuroticism: float      # 0.0-1.0: Emotional volatility
```

**Trait Influences on Behavior:**
- **High Openness**: Explores new locations, asks questions, tries new activities
- **High Conscientiousness**: Completes tasks, maintains schedules, keeps promises
- **High Extraversion**: Initiates conversations, seeks crowded areas, forms groups
- **High Agreeableness**: Offers help, shares resources, avoids conflict
- **High Neuroticism**: Emotional reactions, stress accumulation, mood swings

### 2.2 Dynamic Emotional State

Real-time emotional variables with decay mechanics:

```python
class EmotionalState:
    happiness: float  # General wellbeing
    anger: float      # Frustration level
    fear: float       # Anxiety/concern
    sadness: float    # Melancholy
    surprise: float   # Unexpectedness
    energy: float     # 0-100: Physical/mental fatigue
    stress: float     # 0-1: Pressure accumulation
    
    # Decay rates per tick
    decay_rates = {
        "happiness": 0.01,
        "anger": 0.02,
        "fear": 0.015,
        "sadness": 0.008,
        "surprise": 0.05,
        "stress": 0.012
    }
```

### 2.3 Agent Tools & Actions

Decorated tool methods with energy costs:

```python
@tool
async def speak(self, target: str, message: str) -> str:
    """Engage in conversation with another agent"""
    # Energy cost: -2
    # Updates: relationships, memories
    # Returns: conversation result

@tool
async def reflect(self) -> str:
    """Deep introspection about recent experiences"""
    # Energy cost: -5
    # Creates: reflection memories, insights
    # Updates: goals, beliefs

@tool
async def trade(self, partner: str, offer: dict, request: dict) -> str:
    """Exchange items with another agent"""
    # Energy cost: -5
    # Updates: inventory, relationships
    # Creates: trade memory
```

### 2.4 Sage Agent Specialization

Knowledge-keeper with limited web access:

```python
class SageAgent(SimulationAgent):
    web_searches_today: int = 0
    max_searches_per_day: int = 1
    
    @tool
    async def web_search(self, query: str) -> str:
        """Search the internet for information (1x/day)"""
        if self.web_searches_today >= self.max_searches_per_day:
            return "Daily search limit reached"
        
        # Execute search, store as verified knowledge
        # Confidence: 0.9 (high trust in web results)
        result = await search_api(query)
        self.memory.add_knowledge(result, verified=True)
        self.web_searches_today += 1
        return result
```

## 3. World Environment (10x10 Grid)

### 3.1 Spatial Layout

```
    0   1   2   3   4   5   6   7   8   9
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
0 │ F │ F │ F │ H │ H │ M │ M │ T │ T │ L │  Legend:
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  F=Forest
1 │ F │ . │ . │ H │ H │ M │ M │ T │ T │ L │  H=Houses
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  M=Market
2 │ . │ . │ P │ P │ S │ S │ C │ C │ . │ L │  T=Temple
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  L=Library
3 │ . │ . │ P │ P │ S │ S │ C │ C │ . │ . │  P=Park
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  S=Square
4 │ R │ R │ . │ . │ I │ I │ . │ . │ G │ G │  C=Cafe
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  I=Inn
5 │ R │ R │ . │ . │ I │ I │ . │ . │ G │ G │  R=River
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  G=Garden
6 │ . │ . │ B │ B │ . │ . │ A │ A │ . │ . │  B=Blacksmith
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  A=Apothecary
7 │ . │ . │ B │ B │ . │ . │ A │ A │ . │ . │  W=Waterfront
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  D=Docks
8 │ W │ W │ W │ . │ . │ . │ . │ D │ D │ D │  .=Path
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
9 │ W │ W │ W │ . │ . │ . │ . │ D │ D │ D │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

### 3.2 Location Properties & Buffs

```python
LOCATION_CONFIG = {
    "library": {
        "capacity": 8,
        "resources": ["books", "scrolls", "maps"],
        "buff": {"learning_rate": 2.0},
        "sage_home": True  # Sage spawns here
    },
    "market": {
        "capacity": 20,
        "resources": ["goods", "trade"],
        "economic_hub": True,
        "buff": {}  # No stat buffs
    },
    "temple": {
        "capacity": 15,
        "resources": ["wisdom", "peace"],
        "buff": {"stress_reduction": 2.0, "happiness": 1.3}
    },
    "cafe": {
        "capacity": 12,
        "resources": ["food", "drink", "conversation"],
        "buff": {"social_bonus": 1.5}
    }
}
```

### 3.3 NumPy-Optimized Spatial Operations

```python
class WorldGrid:
    # NumPy arrays for efficient computation
    location_grid: np.ndarray    # Location type indices
    occupancy_grid: np.ndarray   # Agent counts
    walkability_grid: np.ndarray # Movement constraints
    
    def get_distance_matrix(self, position: Tuple[int, int]) -> np.ndarray:
        """Vectorized Manhattan distance calculation"""
        Y, X = np.mgrid[0:self.height, 0:self.width]
        return np.abs(X - position[0]) + np.abs(Y - position[1])
    
    def find_nearest_location_type(self, position, location_type):
        """NumPy-accelerated nearest location search"""
        locations = self.get_locations_by_type(location_type)
        positions = np.array([(loc.x, loc.y) for loc in locations])
        distances = np.sum(np.abs(positions - np.array(position)), axis=1)
        return locations[np.argmin(distances)]
```

## 4. Simulation Engine & Orchestration

### 4.1 Main Simulation Loop

```python
class SimulationEngine:
    async def tick(self):
        """Single simulation timestep (5 seconds real-time)"""
        
        # Phase 1: Perception (parallel)
        perceptions = await asyncio.gather(*[
            agent.perceive() for agent in self.agents
        ])
        
        # Phase 2: Decision-making (LLM calls with budget check)
        decisions = []
        for agent, perception in zip(self.agents, perceptions):
            if self.budget_manager.can_call_llm(agent.id, 500):
                model = self.cost_optimizer.select_model(
                    agent.next_action,
                    self.budget_manager.get_remaining_budget()
                )
                decision = await agent.decide(perception, model=model)
                self.budget_manager.track_usage(agent.id, decision.tokens)
                decisions.append(decision)
        
        # Phase 3: Action resolution
        self.resolve_actions(decisions)
        
        # Phase 4: Memory formation
        memories = self.form_memories(decisions)
        
        # Phase 5: Emotional updates
        self.update_emotions()
        
        # Phase 6: Reflection (every 6 hours)
        if self.current_tick % 6 == 0:
            await self.trigger_reflections()
```

### 4.2 Conversation Management

Multi-turn dialogue with relationship effects:

```python
class ConversationManager:
    max_turns_per_conversation: int = 5
    
    async def manage_conversation(self, initiator, target):
        """Orchestrate multi-turn dialogue"""
        
        # Check social availability
        if not self.is_available(target):
            return None
            
        conversation = []
        for turn in range(self.max_turns):
            # Alternating speakers
            speaker = initiator if turn % 2 == 0 else target
            listener = target if turn % 2 == 0 else initiator
            
            # Generate response with personality influence
            response = await speaker.speak_to(
                listener,
                context=conversation,
                relationship=self.get_relationship(speaker, listener)
            )
            
            conversation.append(response)
            
            # Check for conversation end conditions
            if self.should_end_conversation(response):
                break
        
        # Update relationships based on conversation
        self.update_relationships(initiator, target, conversation)
        
        # Store conversation memories for both agents
        self.store_conversation_memories(initiator, target, conversation)
        
        return conversation
```

### 4.3 Rate Limiting & Cost Optimization

Intelligent resource management:

```python
class CostOptimizer:
    def select_model(self, action: str, budget_remaining_pct: float) -> str:
        """Dynamic model selection based on action priority and budget"""
        
        priority = self.action_priorities[action]
        
        # High priority: Use best model if any budget remains
        if priority >= 4:  # reflect, web_search
            return "gpt-5" if budget_remaining_pct > 0.2 else "gpt-4o"
        
        # Medium priority: Balance quality and cost
        elif priority >= 3:  # speak, trade
            if budget_remaining_pct > 0.6:
                return "gpt-5"
            elif budget_remaining_pct > 0.3:
                return "gpt-4o"
            else:
                return "gpt-4o-mini"
        
        # Low priority: Conserve budget
        else:  # move, observe
            if budget_remaining_pct > 0.8:
                return "gpt-5"
            elif budget_remaining_pct > 0.4:
                return "gpt-4o"
            else:
                return "gpt-4o-mini"
```

## 5. Memory Operations & Retrieval

### 5.1 Hybrid Search Strategy

Combines multiple retrieval methods:

```python
class GraphitiMemoryManager:
    async def retrieve_relevant(self, query: str, agent_id: str, k: int = 5):
        """Multi-faceted memory retrieval"""
        
        # 1. Semantic search (embeddings)
        semantic_results = await self.search_by_embedding(
            query_embedding=self.embed(query),
            limit=k*2
        )
        
        # 2. BM25 text search
        text_results = await self.search_by_text(
            query=query,
            limit=k*2
        )
        
        # 3. Graph traversal (connected memories)
        graph_results = await self.search_by_graph(
            start_node=agent_id,
            max_hops=2,
            limit=k
        )
        
        # 4. Temporal relevance (recent memories weighted higher)
        temporal_results = await self.search_by_time(
            time_window=timedelta(days=1),
            limit=k
        )
        
        # Merge and rank results
        all_results = self.merge_results(
            semantic_results * 0.4,
            text_results * 0.3,
            graph_results * 0.2,
            temporal_results * 0.1
        )
        
        return all_results[:k]
```

### 5.2 Knowledge Propagation

Information spreads through agent network:

```python
def propagate_knowledge(self, source_agent: str, fact: str, confidence: float):
    """Spread information through social connections"""
    
    # Get source agent's connections
    connections = self.get_agent_connections(source_agent)
    
    for connection in connections:
        # Propagation probability based on relationship strength
        propagation_chance = (
            connection.trust * 0.4 +
            connection.friendship * 0.3 +
            connection.respect * 0.3
        ) * confidence
        
        if random.random() < propagation_chance:
            # Reduced confidence for second-hand information
            new_confidence = confidence * 0.8
            
            # Add as rumor if confidence drops below threshold
            if new_confidence < 0.5:
                self.add_rumor(connection.agent_id, fact, new_confidence)
            else:
                self.add_knowledge(connection.agent_id, fact, new_confidence)
```

## 6. Performance Optimizations

### 6.1 Graphiti Memory Optimizations
**Episode Processing (5-10x improvement):**
- **JSON Truncation**: Automatic truncation at 2000 chars (fallback to 500) to prevent parsing errors
- **Timeout Protection**: 30-second timeout for all async operations with graceful degradation
- **Batch Operations**: Groups multiple memory operations into single Neo4j transactions
- **Parallel Processing**: Concurrent perception and decision phases across all agents

**Context Retrieval (<100ms latency):**
- **Graph-based Context**: Efficient traversal using Neo4j indexes
- **Relationship Filtering**: Pre-filter agents based on trust levels
- **Memory Prioritization**: Recent memories weighted higher in context

### 6.2 Caching Strategies
- **Memory Cache**: LRU cache with 5-minute TTL for recent retrievals
- **Embedding Cache**: Persistent cache for computed embeddings
- **Decision Cache**: Reuse decisions for identical world states
- **Query Result Cache**: Cache frequently accessed graph queries

### 6.3 Batch Processing
- **LLM Calls**: Batch agent decisions per tick (up to 5 concurrent)
- **Memory Operations**: Bulk inserts to Neo4j (50 operations per batch)
- **Pathfinding**: Precompute common routes at initialization
- **Episode Storage**: Queue and batch episode writes

### 6.4 Resource Management
- **Connection Pooling**: Neo4j connection reuse with 10-connection pool
- **Prompt Templates**: Precomputed personality strings
- **Lazy Loading**: Load agent details on-demand
- **Token Budgeting**: Dynamic model selection based on remaining budget

### 6.5 Checkpoint System
**State Preservation:**
- **Automatic Saves**: End-of-day checkpoints with full state
- **Pickle Format**: Binary serialization for complete object preservation
- **Incremental Saves**: Only changed data written to disk
- **Resume Capability**: Continue from any checkpoint with `--continue`

**Checkpoint Contents:**
- Agent states (location, goals, relationships, inventory, health, energy)
- Simulation metrics and statistics
- Event log (last 100 events)
- Token usage and budget tracking

### 6.6 Data Isolation
- **Unique Run IDs**: Each simulation uses `sim_run_[uuid]` for partitioning
- **Group-based Queries**: All Graphiti operations scoped to run ID
- **Parallel Simulations**: Multiple concurrent runs without data collision
- **Clean Teardown**: Easy cleanup of specific simulation runs

## 7. Monitoring & Observability

### 7.1 Key Metrics

**Agent Metrics:**
- Personality drift over time
- Emotional volatility index
- Social connectivity score
- Knowledge acquisition rate
- Goal completion ratio

**System Metrics:**
- Token usage by agent/action
- Memory graph growth rate
- Average conversation length
- LLM response latency (P50/P95/P99)
- Daily cost breakdown

**Emergent Behavior Metrics:**
- Unprompted group formations
- Information cascade events
- Economic market dynamics
- Conflict/cooperation ratio

### 7.2 Grafana Dashboards

Real-time visualization panels:
- Agent location heatmap
- Relationship network graph
- Emotional state timeline
- Token budget burndown
- Memory growth chart
- Conversation frequency matrix

## 8. Deployment Configuration

### 8.1 Docker Compose Stack

```yaml
services:
  neo4j:
    image: neo4j:5.26-community
    environment:
      NEO4J_AUTH: neo4j/simulation123
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    volumes:
      - neo4j-data:/data
    ports:
      - "7474:7474"
      - "7687:7687"
  
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
```

### 8.2 Environment Configuration

```env
# LLM Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL_ID=gpt-5
OPENAI_SAGE_MODEL_ID=gpt-5

# Token Budgets (10x increase for GPT-5)
DAILY_TOKEN_BUDGET=1000000      # ~$5/day
PER_AGENT_TOKEN_LIMIT=200000    # ~$1/agent/day

# Simulation Parameters
MAX_AGENTS=5
MAX_DAYS=10
TICKS_PER_DAY=24
TICK_DURATION_MS=5000

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=simulation123

# Monitoring
ENABLE_GRAFANA=true
METRICS_EXPORT_INTERVAL=60
```

## 9. Future Enhancements

### Near-term (v1.1)
- Voice synthesis for agent interviews
- Visual world representation (2D sprites)
- Agent skill progression system
- Dynamic weather effects on behavior
- Multi-language agent communication

### Medium-term (v2.0)
- Expand to 50x50 world grid
- Agent reproduction/family systems
- Complex economic simulation
- Political faction emergence
- Cultural evolution mechanics

### Long-term (v3.0)
- 3D world visualization (Unreal Engine)
- VR interaction with agents
- Multi-village simulations
- Agent consciousness interviews
- Academic research applications

## 10. Success Metrics

1. **Personality Differentiation**: Agents develop statistically distinct personalities (>2σ difference in traits)
2. **Knowledge Propagation**: Information spreads organically through population (>60% awareness in 3 days)
3. **Emergent Behaviors**: Unprogrammed social phenomena occur (alliances, conflicts, celebrations)
4. **Cost Efficiency**: Full day simulation <$5 with GPT-5
5. **Performance**: <1s response time for agent actions
6. **Memory Coherence**: Agents accurately recall past events (>95% accuracy)
7. **Narrative Quality**: Agents produce compelling life stories when interviewed

## References

- Stanford Generative Agents (Park et al., 2023)
- Graphiti: Temporal Knowledge Graphs for LLM Memory (Zep AI, 2024)
- Strands Agent Framework Documentation (2024)
- OpenAI GPT-5 Technical Report (2024)
- "Emergence of Social Dynamics in LLM Agents" (Anthropic Research, 2024)