# LLM Agent Simulation Architecture

## Executive Summary

A multi-agent simulation framework where autonomous LLM-powered agents develop distinct personalities through social interactions, memory formation, and limited knowledge propagation in a bounded virtual world. The system leverages cutting-edge technologies to create a living ecosystem of AI agents that exhibit emergent behaviors, form relationships, and evolve unique personalities over time.

### Key Technologies
- **Strands Agents SDK**: Model-driven agent framework with native OpenAI GPT-5 integration
- **Strands A2A Protocol**: Agent-to-agent communication enabling autonomous conversations
- **Neo4j + Graphiti**: Real-time temporal knowledge graphs with bi-temporal tracking
- **OpenAI GPT-5**: Primary LLM provider for agent cognition and decision-making
- **Python 3.10+**: Modern async/await patterns for concurrent agent processing

## 1. Core Architecture Components

### 1.1 Foundation Technologies

#### Strands Agents SDK (v1.0+)
The core agent framework providing model-driven development with minimal boilerplate. Strands takes a revolutionary approach where the LLM itself drives agent behavior rather than complex orchestration logic.

**Key Features:**
- **Model-First Design**: The LLM is the brain - all decisions flow through it
- **Native OpenAI Integration**: First-class support for GPT-5, GPT-3.5-turbo
- **Tool Ecosystem**: Decorators make any Python function an agent tool
- **A2A Protocol**: Agents can call each other as tools for complex interactions
- **Streaming Support**: Real-time token streaming for responsive agents
- **Type Safety**: Pydantic models for structured inputs/outputs

**Configuration Example:**
```python
from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={"api_key": "sk-..."},
    model_id="gpt-5",
    params={"temperature": 0.7, "max_tokens": 500}
)
agent = Agent(name="John", model=model, tools=[...])
```

#### Neo4j with Graphiti Framework
A revolutionary temporal knowledge graph system designed specifically for AI agent memory. Unlike traditional databases, Graphiti provides real-time, incremental updates without batch processing.

**Core Capabilities:**
- **Bi-Temporal Tracking**: Records both when events happened and when they were learned
- **Real-time Updates**: 300ms P95 latency for memory operations
- **Hybrid Search**: Combines semantic embeddings, BM25 text search, and graph traversal
- **Automatic Ontology**: Learns structure from data without predefined schemas
- **Memory Compression**: Reflection and summarization to manage growth
- **Proven Performance**: Outperforms MemGPT (94.8% vs 93.4% on DMR benchmark)

**Memory Operations:**
- **Ingestion**: Episodes, conversations, observations stored as temporal nodes
- **Retrieval**: Context-aware search considering time, relevance, and relationships
- **Reflection**: Periodic compression of memories into higher-level insights
- **Propagation**: Knowledge flows through relationship edges

#### OpenAI GPT-5 Integration
Primary LLM provider powering agent cognition, selected for its superior reasoning capabilities and extensive world knowledge.

**Model Selection:**
- **Regular Agents**: GPT-5 (balanced cost/performance)
- **Sage Agent**: GPT-5 with lower temperature (0.5) for accuracy
- **Fallback**: GPT-3.5-turbo for non-critical operations

**Token Management:**
- Automatic token counting for cost tracking
- Per-agent and daily budget limits
- Smart truncation for long contexts
- Response caching to reduce redundant calls

### 1.2 System Layers

```
┌──────────────────────────────────────────────────────────┐
│                    Simulation Engine                      │
│         (Orchestration, Time Management, Rules)           │
├──────────────────────────────────────────────────────────┤
│                  Agent Communication Layer                │
│              (Strands A2A Protocol, MCP Tools)           │
├──────────────────────────────────────────────────────────┤
│                      Agent Minds                          │
│     (LLM Reasoning, Personality Models, Decision)         │
├──────────────────────────────────────────────────────────┤
│                    Memory System                          │
│        (Neo4j/Graphiti Temporal Knowledge Graph)          │
├──────────────────────────────────────────────────────────┤
│                   World Environment                       │
│           (Grid System, Locations, Physics)               │
├──────────────────────────────────────────────────────────┤
│                  Observation & Metrics                    │
│         (CloudWatch, OpenTelemetry, Analytics)           │
└──────────────────────────────────────────────────────────┘
```

## 2. Agent Design & Personality System

The personality system is the heart of what makes each agent unique. Rather than scripted behaviors, agents develop genuine personalities through the interplay of base traits, emotional states, and accumulated experiences.

### 2.1 Core Personality Traits (Big Five Model)

The Big Five personality model from psychology provides a scientifically-grounded framework for agent personalities. Each trait influences decision-making, dialogue style, and relationship formation.

```python
personality_traits = {
    "openness": 0.0-1.0,          # Curiosity, creativity, imagination
    "conscientiousness": 0.0-1.0,  # Organization, discipline, reliability
    "extraversion": 0.0-1.0,       # Social energy, assertiveness, talkativeness
    "agreeableness": 0.0-1.0,      # Cooperation, trust, empathy
    "neuroticism": 0.0-1.0         # Emotional volatility, stress sensitivity
}
```

**Trait Influences:**
- **Openness**: High = seeks new experiences, asks questions, explores. Low = prefers routine, skeptical of change
- **Conscientiousness**: High = completes tasks, keeps promises. Low = spontaneous, flexible
- **Extraversion**: High = initiates conversations, seeks crowds. Low = prefers solitude, deeper connections
- **Agreeableness**: High = helpful, trusting, cooperative. Low = competitive, skeptical
- **Neuroticism**: High = emotionally reactive, stress-prone. Low = calm, stable

**Derived Traits:**
Additional personality dimensions calculated from the Big Five:
- **Curiosity**: Openness × 0.8 + random factor
- **Creativity**: Openness × 0.7 + conscientiousness × 0.3
- **Ambition**: Conscientiousness × 0.6 + extraversion × 0.4
- **Patience**: (1 - neuroticism) × 0.7
- **Humor**: Extraversion × 0.5 + openness × 0.3

### 2.2 Dynamic Emotional States

Real-time emotional variables that decay/recover over time:

```python
emotional_state = {
    "happiness": 0.0-1.0,    # General wellbeing
    "anger": 0.0-1.0,        # Frustration level
    "fear": 0.0-1.0,         # Anxiety/concern
    "sadness": 0.0-1.0,      # Melancholy
    "surprise": 0.0-1.0,     # Unexpectedness
    "energy": 0.0-1.0,       # Physical/mental fatigue
    "stress": 0.0-1.0        # Pressure accumulation
}
```

### 2.3 Social Relationship Matrix

Per-agent relationship tracking in Neo4j graph:

```cypher
(:Agent {id: "alice"})-[:RELATIONSHIP {
    trust: 0.5,
    friendship: 0.3,
    respect: 0.6,
    familiarity: 0.4,
    last_interaction: timestamp,
    interaction_count: 12
}]->(:Agent {id: "bob"})
```

### 2.4 Knowledge & Beliefs

```python
knowledge_system = {
    "facts": [],              # Verified information
    "beliefs": [],            # Unverified assumptions
    "rumors": [],             # Propagated hearsay
    "skills": {},             # Learned capabilities
    "secrets": [],            # Private information
    "goals": [],              # Personal objectives
    "values": []              # Core principles
}
```

### 2.5 Physical & Economic Stats

```python
physical_stats = {
    "health": 100,
    "energy": 100,
    "hunger": 0.0-1.0,
    "location": (x, y),
    "inventory": [],
    "wallet": {"gold": 50, "silver": 100},
    "reputation": 0.0-1.0
}
```

## 3. World Environment Design

### 3.1 Grid System (10x10 Initial)

```
    0   1   2   3   4   5   6   7   8   9
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
0 │ F │ F │ F │ H │ H │ M │ M │ T │ T │ L │  F=Forest
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
1 │ F │ . │ . │ H │ H │ M │ M │ T │ T │ L │  H=Houses
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
2 │ . │ . │ P │ P │ S │ S │ C │ C │ . │ L │  M=Market
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
3 │ . │ . │ P │ P │ S │ S │ C │ C │ . │ . │  T=Temple
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
4 │ R │ R │ . │ . │ I │ I │ . │ . │ G │ G │  L=Library
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
5 │ R │ R │ . │ . │ I │ I │ . │ . │ G │ G │  P=Park
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
6 │ . │ . │ B │ B │ . │ . │ A │ A │ . │ . │  S=Square
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
7 │ . │ . │ B │ B │ . │ . │ A │ A │ . │ . │  C=Cafe
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
8 │ W │ W │ W │ . │ . │ . │ . │ D │ D │ D │  I=Inn
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
9 │ W │ W │ W │ . │ . │ . │ . │ D │ D │ D │  R=River
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
    B=Blacksmith  A=Apothecary  G=Garden
    W=Waterfront  D=Docks  .=Path
```

### 3.2 Location Properties

```python
location_metadata = {
    "library": {
        "type": "building",
        "capacity": 5,
        "resources": ["books", "scrolls"],
        "buff": {"learning_rate": 1.5},
        "sage_home": True  # Sage agent starts here
    },
    "tavern": {
        "type": "building", 
        "capacity": 10,
        "resources": ["food", "drink", "gossip"],
        "buff": {"social_bonus": 1.2},
        "meeting_point": True
    },
    "market": {
        "type": "area",
        "capacity": 20,
        "resources": ["trade", "goods"],
        "economic_hub": True
    }
}
```

### 3.3 Time System

```python
time_config = {
    "ticks_per_day": 24,
    "tick_duration_ms": 5000,  # 5 seconds real-time
    "days_per_week": 7,
    "seasons": ["spring", "summer", "fall", "winter"],
    "events": {
        "dawn": 6,
        "noon": 12,
        "dusk": 18,
        "midnight": 0
    }
}
```

## 4. Memory Architecture (Neo4j/Graphiti)

### 4.1 Graph Schema

```cypher
// Core Entities
(:Agent {
    id: string,
    name: string,
    role: string,
    created_at: datetime,
    personality: json
})

(:Memory {
    id: string,
    type: "observation|conversation|reflection|learned",
    content: string,
    importance: float,
    timestamp: datetime,
    t_valid: datetime,    // Bi-temporal: event time
    t_invalid: datetime,  // Bi-temporal: expiry
    t_ingested: datetime, // Bi-temporal: system time
    embedding: vector
})

(:Location {
    id: string,
    name: string,
    x: int,
    y: int,
    type: string
})

(:Knowledge {
    id: string,
    fact: string,
    source: string,
    confidence: float,
    verified: boolean
})

(:Item {
    id: string,
    name: string,
    type: string,
    properties: json
})

// Relationships
(:Agent)-[:REMEMBERS {strength: float}]->(:Memory)
(:Agent)-[:KNOWS {confidence: float}]->(:Knowledge)
(:Agent)-[:LOCATED_AT]->(:Location)
(:Agent)-[:OWNS]->(:Item)
(:Agent)-[:RELATIONSHIP {trust, friendship, ...}]->(:Agent)
(:Memory)-[:REFERENCES]->(:Agent|:Location|:Item)
(:Knowledge)-[:DERIVED_FROM]->(:Memory)
(:Knowledge)-[:CONTRADICTS]->(:Knowledge)
```

### 4.2 Memory Operations

```python
class MemoryManager:
    def store_observation(agent_id, observation):
        # Create memory node with embedding
        # Link to agent and referenced entities
        # Update importance scores
        
    def retrieve_relevant(agent_id, context, k=5):
        # Hybrid search: embedding similarity + BM25 + graph
        # Filter by recency and importance
        # Return top-k memories
        
    def reflect(agent_id):
        # Aggregate recent memories
        # Generate high-level insights via LLM
        # Store reflection as new memory type
        
    def forget(agent_id, decay_rate=0.01):
        # Decay importance scores over time
        # Archive or delete low-importance memories
```

## 5. Agent Communication Protocol

### 5.1 Strands A2A Implementation

```python
from strands import Agent
from strands.multiagent.a2a import A2AServer, A2AClientToolProvider

class SimulationAgent(Agent):
    def __init__(self, persona, stats, memory_manager):
        super().__init__(
            name=persona.name,
            system_prompt=self._build_prompt(persona, stats),
            tools=[
                self.speak,
                self.move,
                self.trade,
                self.observe,
                self.reflect
            ]
        )
        self.memory = memory_manager
        self.stats = stats
        
    @tool
    def speak(self, target: str, message: str):
        """Initiate conversation with another agent"""
        # Route through A2A protocol
        # Update relationship scores
        # Store conversation memory
        
    @tool  
    def observe(self):
        """Perceive current environment"""
        # Get nearby agents and objects
        # Retrieve relevant memories
        # Generate observation memory
```

### 5.2 Sage Agent Special Tools

```python
class SageAgent(SimulationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_search_uses = 0
        self.last_search_day = 0
        
    @tool
    def web_search(self, query: str):
        """Search the web for information (1x per day limit)"""
        if self.web_search_uses >= 1 and current_day == self.last_search_day:
            return "Search limit reached for today"
            
        # Execute search via Bedrock or external API
        result = bedrock_search(query)
        
        # Store as verified knowledge
        knowledge = self.memory.store_knowledge(
            fact=result,
            source="web_search",
            confidence=0.9,
            verified=True
        )
        
        self.web_search_uses += 1
        self.last_search_day = current_day
        return knowledge
```

## 6. Simulation Engine

### 6.1 Main Loop

```python
class SimulationEngine:
    def __init__(self, config):
        self.agents = []
        self.world = WorldGrid(config.world_size)
        self.memory = MemoryManager(neo4j_connection)
        self.time = TimeManager(config.time)
        self.metrics = MetricsCollector()
        
    async def run_simulation(self, days=10):
        for day in range(days):
            for tick in range(self.time.ticks_per_day):
                # Phase 1: Perception
                perceptions = await self.gather_perceptions()
                
                # Phase 2: Decision (parallel)
                decisions = await asyncio.gather(*[
                    agent.decide(perceptions[agent.id])
                    for agent in self.agents
                ])
                
                # Phase 3: Action resolution
                self.resolve_actions(decisions)
                
                # Phase 4: State updates
                self.update_world_state()
                self.update_agent_stats()
                
                # Phase 5: Memory & reflection
                if tick % 6 == 0:  # Every 6 ticks
                    await self.trigger_reflections()
                    
                # Phase 6: Metrics
                self.metrics.record_tick(self.get_state())
                
            # End of day processing
            await self.end_of_day_processing()
```

### 6.2 Interaction Resolution

```python
def resolve_interactions(self, agents_in_location):
    """Handle multi-agent interactions in same location"""
    
    if len(agents_in_location) < 2:
        return
        
    # Probabilistic interaction based on personality
    for agent_a, agent_b in combinations(agents_in_location, 2):
        interaction_prob = calculate_interaction_probability(
            agent_a.stats.extraversion,
            agent_b.stats.extraversion,
            agent_a.get_relationship(agent_b.id),
            self.world.get_location_type()
        )
        
        if random.random() < interaction_prob:
            # Initiate conversation via A2A
            conversation = await self.run_conversation(agent_a, agent_b)
            
            # Update relationships
            self.update_relationships(agent_a, agent_b, conversation)
            
            # Store memories
            self.store_conversation_memories(conversation)
```

## 7. Tool & Inventory System

### 7.1 Core Tools Available

```python
AGENT_TOOLS = {
    # Movement & Navigation
    "move": {"cost": 1, "energy": -5},
    "plan_route": {"cost": 0, "energy": -1},
    
    # Social
    "speak": {"cost": 0, "energy": -2},
    "gossip": {"cost": 0, "energy": -3},
    "teach": {"cost": 0, "energy": -10},
    "learn": {"cost": 0, "energy": -8},
    
    # Economic
    "trade": {"cost": 0, "energy": -5},
    "craft": {"cost": 0, "energy": -15},
    "work": {"cost": 0, "energy": -20, "gold": +10},
    
    # Knowledge
    "read": {"cost": 0, "energy": -10, "knowledge": +1},
    "write_note": {"cost": 1, "energy": -15},
    "post_notice": {"cost": 2, "energy": -5},
    
    # Reflection
    "reflect": {"cost": 0, "energy": -5},
    "remember": {"cost": 0, "energy": -2}
}

SAGE_EXCLUSIVE_TOOLS = {
    "web_search": {"cost": 0, "energy": -20, "limit": "1/day"},
    "verify_knowledge": {"cost": 0, "energy": -10},
    "write_book": {"cost": 5, "energy": -50}
}
```

### 7.2 Inventory Management

```python
class Inventory:
    def __init__(self, capacity=20):
        self.items = {}
        self.capacity = capacity
        
    def add_item(self, item_id, quantity=1):
        if self.get_weight() + quantity > self.capacity:
            return False
        self.items[item_id] = self.items.get(item_id, 0) + quantity
        return True
        
    def trade(self, other_inventory, give_items, receive_items):
        # Validate trade feasibility
        # Execute atomic swap
        # Update both inventories
```

## 8. Rate Limiting & Cost Management

### 8.1 Token Budget System

```python
class TokenBudgetManager:
    def __init__(self):
        self.daily_budget = 100000  # tokens
        self.per_agent_limit = 20000
        self.used_today = 0
        self.agent_usage = {}
        
    def can_call_llm(self, agent_id, estimated_tokens):
        if self.used_today + estimated_tokens > self.daily_budget:
            return False
        if self.agent_usage.get(agent_id, 0) + estimated_tokens > self.per_agent_limit:
            return False
        return True
        
    def track_usage(self, agent_id, actual_tokens):
        self.used_today += actual_tokens
        self.agent_usage[agent_id] = self.agent_usage.get(agent_id, 0) + actual_tokens
```

### 8.2 Optimization Strategies

- **Conversation Batching**: Group agent decisions per tick
- **Memory Caching**: Reuse recent retrievals for 5 minutes
- **Prompt Templates**: Precomputed personality strings
- **Reflection Scheduling**: Staggered, not simultaneous
- **Model Routing**: Smaller models for simple decisions
- **Context Pruning**: Limit conversation history to 5 turns

## 9. Deployment Configuration

### 9.1 AgentCore Runtime Setup

```yaml
# agentcore-config.yaml
runtime:
  max_duration: 28800  # 8 hours
  session_isolation: true
  identity_provider: cognito
  
agents:
  - name: alice_farmer
    model: anthropic.claude-3-haiku
    memory_store: neo4j
    tools: [move, speak, trade, work]
    
  - name: sage_librarian
    model: anthropic.claude-3-sonnet  # Better model for sage
    memory_store: neo4j
    tools: [move, speak, teach, web_search]
    
observability:
  cloudwatch: true
  opentelemetry: true
  metrics_interval: 60
```

### 9.2 Infrastructure Requirements

```python
INFRASTRUCTURE = {
    "compute": {
        "agentcore_runtime": "managed",
        "neo4j": "m5.xlarge",
        "orchestrator": "t3.medium"
    },
    "storage": {
        "neo4j_volume": "100GB SSD",
        "logs": "S3 Standard"
    },
    "networking": {
        "vpc": "10.0.0.0/16",
        "private_subnets": ["10.0.1.0/24", "10.0.2.0/24"],
        "nat_gateway": True
    },
    "models": {
        "primary": "claude-3-haiku-20240307",
        "sage": "claude-3-sonnet-20240229",
        "fallback": "llama-3-8b"
    }
}
```

## 10. Metrics & Evaluation

### 10.1 Personality Evolution Metrics

- **Trait Drift**: Δ personality vectors over time
- **Relationship Density**: Average connections per agent
- **Knowledge Propagation**: Time for facts to spread
- **Emotional Variance**: Mood swing frequency/amplitude
- **Goal Achievement**: Completed vs abandoned objectives

### 10.2 System Performance Metrics

- **Token Usage**: Per agent, per day, per interaction
- **Memory Growth**: Nodes/edges added per tick
- **Retrieval Latency**: P50/P95/P99 for memory queries
- **Conversation Length**: Average turns per interaction
- **Emergent Behaviors**: Unprompted collaborative actions

## 11. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- Set up AWS Bedrock AgentCore
- Deploy Neo4j with Graphiti
- Implement basic Strands agents
- Create world grid system

### Phase 2: Agent Personalities (Week 2)
- Implement Big Five traits
- Add emotional state system
- Create 5 initial agent personas
- Basic movement and perception

### Phase 3: Memory & Communication (Week 3)
- Neo4j memory storage
- A2A conversation protocol
- Memory retrieval system
- Basic reflection mechanism

### Phase 4: Tools & Economy (Week 4)
- Inventory system
- Trading mechanics
- Sage web search tool
- Crafting and resources

### Phase 5: Optimization & Scale (Week 5)
- Rate limiting implementation
- Cost optimization
- Performance tuning
- Metrics dashboard

### Phase 6: Evaluation & Iteration (Week 6)
- Run extended simulations
- Analyze emergent behaviors
- Personality interviews via TTS
- Refine and document

## 12. Success Criteria

1. **Distinct Personalities**: Each agent develops unique characteristics measurably different from initial state
2. **Knowledge Propagation**: Information from sage spreads through population via conversation
3. **Emergent Behaviors**: Unprogrammed social structures or events arise (parties, alliances, conflicts)
4. **Cost Efficiency**: Full day simulation < $10 in LLM costs
5. **Performance**: Real-time interaction with <1s response time
6. **Memory Persistence**: Agents recall and reference past events correctly
7. **Narrative Coherence**: Agents can explain their experiences when interviewed

## References & Inspiration

- Stanford Smallville (Park et al., 2023)
- AWS Bedrock AgentCore Documentation (2025)
- Strands Agents SDK Guide
- Graphiti: Temporal Knowledge Graphs (Zep AI)
- Voyager: LLM-powered Embodied Agents
- MemGPT/Letta Framework
- AI Town Open Source Project