# Simulation Features Documentation

## Table of Contents
- [Memory System](#memory-system)
- [Performance Optimizations](#performance-optimizations)
- [Agent Capabilities](#agent-capabilities)
- [World System](#world-system)
- [Simulation Management](#simulation-management)
- [Monitoring & Analytics](#monitoring--analytics)

## Memory System

### Graphiti Integration
The simulation uses Graphiti, a temporal knowledge graph framework, for sophisticated memory management:

- **Bi-temporal Storage**: Episodes stored with both valid-time and transaction-time
- **Entity Extraction**: Automatic extraction of entities and relationships from conversations
- **Graph-based Retrieval**: Context-aware memory retrieval using graph traversal
- **Group Isolation**: Each simulation run uses a unique `group_id` for data partitioning

### Memory Types

#### Episodic Memory
- Stores complete interaction episodes with timestamps
- Automatic JSON truncation (2000 chars) to prevent parsing errors
- Fallback to 500 chars on initial failure
- Preserved in graph with full temporal context

#### Entity Memory
- Extracted entities from conversations (people, places, concepts)
- Linked relationships between entities
- Confidence scoring for entity resolution

#### Relationship Tracking
- Trust levels between agents
- Friendship scores
- Trade history
- Conversation topics

### Memory Operations

```python
# Store an episode (automatically truncated if needed)
await memory_manager.add_episode(
    episode_data,
    group_id=simulation_run_id,
    source_description="conversation"
)

# Retrieve context for decision-making
context = await memory_manager.get_context(
    agent_id,
    include_relationships=True,
    limit=10
)
```

## Performance Optimizations

### Graphiti Optimizer
Located in `src/memory/optimizations.py`, provides:

#### Request Handling
- **Timeout Protection**: 30-second default timeout for all async operations
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Recovery**: Graceful degradation on failures

#### Batch Processing
- Groups multiple memory operations into single transactions
- Reduces Neo4j round-trips by up to 80%
- Parallel processing of independent operations

#### Query Optimization
- **Caching Layer**: LRU cache for frequently accessed memories
- **Index Utilization**: Ensures proper Neo4j indexes for common queries
- **Query Batching**: Combines multiple queries into single Cypher statements

### Performance Metrics
- Episode processing: 5-8 seconds (down from 30+ seconds)
- Memory retrieval: <100ms for recent memories
- Batch operations: 10x improvement for bulk inserts

## Agent Capabilities

### Base Agent Features
All agents inherit from `SimulationAgent` with:

- **Personality Traits**: Big Five personality model
- **Energy System**: Actions consume energy, rest restores it
- **Health Tracking**: Impacts decision-making and actions
- **Goal Management**: Dynamic goal setting and achievement
- **Tool Usage**: Strands-based tool system for actions

### Available Tools

#### Communication
- `speak`: Engage in conversations with other agents
- `reflect`: Internal monologue and self-assessment
- `remember`: Store important information

#### Movement & Observation
- `move`: Navigate the world grid
- `observe`: Perceive environment and other agents

#### Interaction
- `trade`: Exchange resources with other agents
- `work`: Perform location-specific activities
- `rest`: Restore energy and health

### Special Agent: Sage

The Sage agent (`sage_librarian`) has unique capabilities:

- **Web Search**: Limited to 1 search per day
- **Knowledge Creation**: Can write scrolls with learned information
- **Teaching**: Shares knowledge with other agents
- **Text Consultation**: Access to special knowledge base

## World System

### Grid-based Environment
- **10x10 Grid**: Spatial representation of the village
- **Location Types**: Each with unique properties and symbols
  - Library (L): Knowledge and learning
  - Market (M): Trading hub
  - Temple (T): Spiritual activities
  - Houses (H): Agent homes
  - Forest (F): Resource gathering
  - Square (S): Social interactions

### Location Properties
```python
LOCATION_PROPERTIES = {
    'library': {
        'symbol': 'L',
        'color': 'cyan',
        'actions': ['read', 'write', 'research'],
        'resources': ['knowledge', 'scrolls']
    },
    # ... more locations
}
```

### Dynamic World State
- Time-based changes (day/night cycles)
- Weather effects on agent behavior
- Resource availability fluctuations
- Event triggers based on world conditions

## Simulation Management

### Checkpoint System

#### Automatic Saving
- Checkpoints created at end of each simulated day
- Stored as pickle files for complete state preservation
- Filename format: `checkpoint_day_N.pkl`

#### Checkpoint Contents
```python
{
    'current_day': int,
    'current_tick': int,
    'agents': {
        'agent_id': {
            'state': str,
            'location': tuple,
            'goals': list,
            'relationships': dict,
            'inventory': dict,
            'health': float,
            'energy': float
        }
    },
    'metrics': dict,
    'event_log': list,  # Last 100 events
    'token_usage': dict
}
```

#### Resuming Simulations
```bash
# Continue from specific checkpoint
sim run --continue checkpoints/checkpoint_day_3.pkl --days 2

# This restores:
# - All agent states
# - Simulation metrics
# - Event history
# - Token usage tracking
```

### Graceful Interruption
- CTRL+C handler saves current state
- Cleanup operations preserve data integrity
- Automatic checkpoint on unexpected termination

### Data Isolation
Each simulation run creates a unique identifier:
- Format: `sim_run_[uuid]`
- All memories tagged with this ID
- Enables parallel simulations without data collision
- Easy cleanup of specific runs

## Monitoring & Analytics

### Live Dashboard
Real-time CLI visualization showing:

#### Agent Panel
- Current locations on mini-map
- Health and energy bars
- Current action/state
- Active goals

#### World Map
- 10x10 grid visualization
- Agent positions marked with initials
- Location types with symbols
- Real-time updates

#### Metrics Panel
- Total conversations
- Trading activity
- Reflection counts
- Token usage and costs

#### Recent Events
- Last 3 agent actions
- Formatted activity log
- Timestamp tracking

### Token Management

#### Budget System
- **Daily Budget**: 1,000,000 tokens
- **Per-Agent Limit**: 200,000 tokens
- **Dynamic Model Selection**: Downgrades models as budget depletes

#### Model Hierarchy
1. **GPT-4** (Primary): Full-featured responses
2. **GPT-4o** (Fallback): When budget < 30%
3. **GPT-4o-mini** (Emergency): Minimal token usage

#### Priority-based Selection
```python
HIGH_PRIORITY = ['reflect', 'web_search']  # Always best model if possible
MEDIUM_PRIORITY = ['speak', 'trade']       # Balanced model selection
LOW_PRIORITY = ['move', 'observe']         # Conservative token usage
```

### Neo4j Monitoring

#### Graph Metrics
- Total nodes by type (Entity, Episodic, etc.)
- Relationship counts
- Memory formation rate
- Query performance stats

#### Example Queries
```cypher
// View agent memories
MATCH (e:Episodic)-[:MENTIONS]->(entity:Entity)
WHERE e.group_id = 'sim_run_xyz'
RETURN e, entity
LIMIT 100

// Analyze relationships
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE a.group_id = 'sim_run_xyz'
RETURN a.name, type(r), b.name, r.confidence
```

### Grafana Dashboards
Available at http://localhost:3000 with panels for:

- **Agent Activity**: Actions per tick, conversation rates
- **Memory Formation**: Episodes created, entities extracted
- **Performance Metrics**: API latency, token consumption
- **System Health**: CPU, memory, Neo4j statistics

## Advanced Features

### JSON Truncation System
Prevents parsing errors from oversized LLM responses:
```python
async def _add_episode_safe(self, content, **kwargs):
    try:
        # Try with 2000 char limit
        truncated = self._truncate_json(content, 2000)
        return await self.add_episode(truncated, **kwargs)
    except:
        # Fallback to 500 chars
        truncated = self._truncate_json(content, 500)
        return await self.add_episode(truncated, **kwargs)
```

### Context-Aware Decisions
Agents use graph context for informed choices:
```python
context = perception.get("context", {})
recent_memories = context.get("recent_memories", [])
relationships = context.get("relationships", {})

# Filter potential conversation partners
trusted_agents = [
    a for a in nearby_agents 
    if relationships.get(a.id, {}).get("trust", 0) > 0.5
]
```

### Parallel Processing
Simulation phases run concurrently where possible:
```python
# Gather all perceptions in parallel
perceptions = await asyncio.gather(*[
    agent.perceive(world_state) 
    for agent in agents
])

# Make decisions concurrently
decisions = await asyncio.gather(*[
    agent.decide(perception) 
    for agent, perception in zip(agents, perceptions)
])
```

## Configuration Options

### Environment Variables
```env
# Graphiti Settings
GRAPHITI_BATCH_SIZE=50
GRAPHITI_CACHE_TTL=300
GRAPHITI_TIMEOUT=30

# Performance Tuning
PARALLEL_AGENTS=true
MEMORY_BATCH_MODE=true
CONTEXT_CACHE_SIZE=100

# Debug Options
TRACE_MEMORY_OPS=false
LOG_GRAPH_QUERIES=false
PROFILE_PERFORMANCE=false
```

### Runtime Flags
```bash
# Debug modes
--debug         # General debug output
--trace         # Detailed per-tick logging
--profile       # Performance profiling

# Memory options
--no-graphiti   # Use basic memory instead
--memory-limit  # Set memory operation limit

# Performance
--parallel      # Enable parallel processing
--batch-size    # Set batch operation size
```