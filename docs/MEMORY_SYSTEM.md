# Memory System Documentation

This document describes how the memory system works in the simulation, integrating Neo4j, Graphiti, and temporal knowledge graphs.

## Overview

The simulation uses a sophisticated memory system that combines:
- **Neo4j**: Graph database for storing relationships and entities
- **Graphiti**: Temporal knowledge graph framework for bi-temporal tracking
- **Embeddings**: Semantic search capabilities using sentence transformers
- **Hybrid Search**: Combines semantic, full-text, and graph traversal

## Architecture

### Dual Memory Managers

#### 1. GraphitiMemoryManager (`memory/graphiti_manager.py`)
- Uses Graphiti framework for advanced temporal features
- Handles bi-temporal tracking (when events happened vs. when learned)
- Provides incremental updates without batch recomputation
- Manages 66 node types and 57 edge types

#### 2. MemoryManager (`memory/manager.py`)
- Direct Neo4j integration for simpler operations
- Handles basic CRUD operations
- Manages memory decay and reflection
- Provides relationship tracking

## Memory Types

### Core Memory Categories

| Type | Purpose | Example |
|------|---------|---------|
| `OBSERVATION` | Environmental perception | "Saw John at the market" |
| `CONVERSATION` | Dialogue records | "Discussed weather with Mary" |
| `REFLECTION` | Compressed insights | "I seem to enjoy mornings" |
| `ACTION` | Things done | "Planted seeds in garden" |
| `LEARNED` | Verified knowledge | "Wheat grows best in spring" |
| `RUMOR` | Unverified information | "Heard the merchant is leaving" |
| `FACT` | Confirmed truth | "The temple opens at dawn" |
| `DREAM` | Night visions | "Dreamed of flying over mountains" |
| `GOSSIP` | Social information | "Mary said John likes Sarah" |
| `SECRET` | Hidden knowledge | "The sage has a hidden scroll" |
| `PROPHECY` | Future predictions | "Rain will come in three days" |
| `STORY` | Narratives | "The tale of the founding" |

## Temporal Features

### Bi-Temporal Tracking
Every memory has multiple timestamps:
- **t_valid**: When the event actually happened
- **t_ingested**: When the agent learned about it
- **t_invalid**: When it becomes obsolete (optional)

### Example
```python
# Agent learns about a past event
memory = Memory(
    content="The merchant arrived yesterday",
    t_valid=datetime.now() - timedelta(days=1),  # Event happened yesterday
    t_ingested=datetime.now(),  # Agent learns now
    t_invalid=None  # Still valid
)
```

## Memory Storage Process

### 1. Initial Perception
```python
# Agent observes something
await memory_manager.ingest_observation(
    agent_id="agent_001",
    observation="Saw a stranger at the gate",
    location="town_gate",
    importance=0.8
)
```

### 2. Entity Extraction
Graphiti automatically extracts:
- Entities mentioned (stranger, gate)
- Relationships implied (AT_LOCATION)
- Temporal context

### 3. Embedding Generation
- Content is converted to embeddings
- Enables semantic search
- Finds similar memories

### 4. Graph Storage
Creates nodes and edges:
```
(Agent)-[:OBSERVED]->(Event)
(Event)-[:OCCURRED_AT]->(Location)
(Agent)-[:REMEMBERS]->(Memory)
```

## Memory Retrieval

### Hybrid Search Methods

#### 1. Semantic Search
Find memories by meaning:
```python
memories = await memory_manager.search_temporal(
    query="happy moments with friends",
    agent_id="agent_001",
    limit=5
)
```

#### 2. Graph Traversal
Follow relationships:
```python
# Find all memories involving a specific person
MATCH (a:Agent)-[:REMEMBERS]->(m:Memory)-[:INVOLVES]->(b:Agent {name: "John"})
RETURN m
```

#### 3. Temporal Queries
Time-based retrieval:
```python
memories = await memory_manager.retrieve_memories(
    agent_id="agent_001",
    time_range=(yesterday, today),
    min_importance=0.5
)
```

## Memory Dynamics

### Importance Calculation
Memory importance is calculated based on:
- Emotional intensity (0.0 - 1.0)
- Novelty (how different from existing memories)
- Relevance to goals
- Social significance

### Memory Decay
Memories decay over time unless reinforced:
```python
# Daily decay
new_importance = old_importance * 0.99

# Reinforcement
if memory_accessed:
    new_importance = min(1.0, old_importance + 0.1)
```

### Reflection & Compression
Periodically, agents reflect on memories:
1. Gather recent memories (last 100)
2. Identify patterns and themes
3. Create reflection memories
4. Decay old, unimportant memories

## Relationship Evolution

### Trust Building
```python
# Positive interaction increases trust
await memory_manager.update_relationship(
    agent_a="agent_001",
    agent_b="agent_002",
    changes={"trust": 0.1, "familiarity": 0.05}
)
```

### Conflict Resolution
```python
# After argument
changes = {
    "trust": -0.2,
    "friendship": -0.1,
    "respect": -0.05
}
```

## Knowledge Propagation

### Information Flow
1. **Direct Learning**: Agent experiences something
2. **Teaching**: Agent shares knowledge with TAUGHT edge
3. **Rumor Spreading**: Information travels with decreasing confidence
4. **Verification**: Rumors can become facts when confirmed

### Confidence Decay
```python
# Each hop reduces confidence
new_confidence = original_confidence * 0.8 ** hops
```

## Query Examples

### Find Recent Important Memories
```cypher
MATCH (a:Agent {id: $agent_id})-[:REMEMBERS]->(m:Memory)
WHERE m.timestamp > datetime() - duration('P7D')
  AND m.importance > 0.7
RETURN m
ORDER BY m.importance DESC
```

### Track Knowledge Source
```cypher
MATCH path = (a:Agent)-[:LEARNED_FROM*]->(source)
WHERE a.id = $agent_id
RETURN path, length(path) as hops
```

### Find Emotional Memories
```cypher
MATCH (a:Agent)-[:REMEMBERS]->(m:Memory)
WHERE m.emotion IN ['joy', 'love', 'excitement']
RETURN m.content, m.emotion, m.timestamp
```

## Performance Optimization

### Indexing Strategy
- Primary indexes on IDs
- Temporal indexes on timestamps
- Full-text index on content
- Bloom filters for existence checks

### Caching
- Recent memories cached in memory
- Frequently accessed relationships cached
- Embedding cache for repeated queries

### Batch Operations
- Group related memory stores
- Bulk relationship updates
- Periodic cleanup of old memories

## Integration with Agent Decision Making

Agents use memories to:

1. **Evaluate Options**
```python
# Check past experiences
memories = await retrieve_memories(query="market trading")
success_rate = calculate_success_from_memories(memories)
```

2. **Maintain Consistency**
```python
# Avoid contradictions
if new_action_contradicts(existing_memories):
    reconsider_action()
```

3. **Social Navigation**
```python
# Check relationships before interaction
relationship = await get_relationship_strength(other_agent)
if relationship["trust"] < 0.3:
    be_cautious()
```

## Memory Limits

### Per-Agent Limits
- Maximum active memories: 10,000
- Maximum relationships: 500
- Reflection threshold: 100 recent memories

### System Limits
- Total graph nodes: 1,000,000
- Total edges: 10,000,000
- Query timeout: 30 seconds

## Future Enhancements

### Planned Features
1. **Collective Memory**: Shared cultural knowledge
2. **Memory Dreams**: Subconscious processing during sleep
3. **Traumatic Memories**: Special handling for impactful events
4. **Memory Chains**: Linked episodic sequences
5. **False Memories**: Misremembered or altered memories
6. **Memory Palaces**: Spatial memory techniques

### Research Areas
- Emotion-colored memory retrieval
- Memory consolidation during rest
- Social memory synchronization
- Predictive memory pre-fetching