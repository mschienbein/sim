# Memory Node Types Documentation

This document describes all the node types available in the simulation's temporal knowledge graph. Nodes represent entities, concepts, and states in the simulation world.

## Node Categories (66 Total Types)

### 🧑 Core Entities (3 types)
Fundamental entities that exist in the world.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Agent` | Simulation agents with personalities | id, name, personality, stats, location |
| `Location` | Places in the world | name, type, capacity, coordinates |
| `Building` | Specific structures | name, owner, type, location |

### 📦 Items & Objects (7 types)
Physical objects that can be owned, traded, or used.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Item` | Physical objects | name, owner, value, weight |
| `Tool` | Items with specific uses | name, purpose, durability, effectiveness |
| `Food` | Consumable items | name, energy_value, freshness |
| `Book` | Written works | title, author, content, genre |
| `Scroll` | Written messages | author, content, recipient |
| `Artifact` | Special/magical items | name, power, rarity, effects |
| `Currency` | Money or tokens | amount, type, owner |

### 📚 Knowledge & Information (8 types)
Information, facts, and beliefs in the world.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Memory` | Event snippets | content, timestamp, importance, emotion |
| `Fact` | Verified knowledge | statement, source, confidence, verified |
| `Rumor` | Unverified claims | claim, source, confidence, hops |
| `Secret` | Hidden knowledge | content, known_by, importance |
| `Recipe` | Instructions | name, ingredients, steps, result |
| `Story` | Narratives | title, content, moral, author |
| `Prophecy` | Future predictions | prediction, prophet, timeframe |
| `Law` | Rules and regulations | title, content, enforcer, penalty |

### 👥 Social Structures (5 types)
Social organizations and groupings.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Community` | Social groups | name, members, purpose, values |
| `Guild` | Professional organizations | name, trade, members, rank_system |
| `Family` | Family units | surname, members, head, lineage |
| `Faction` | Political groups | name, ideology, leader, goals |
| `Relationship` | Explicit connections | type, participants, strength |

### 🎯 Skills & Abilities (4 types)
Capabilities and professional attributes.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Skill` | Learned abilities | name, level, category, teacher |
| `Talent` | Natural aptitudes | name, strength, discovered_date |
| `Profession` | Job roles | title, skills_required, income |
| `Title` | Earned honors | name, bestowed_by, date, privileges |

### 💰 Economic & Trade (6 types)
Commercial and economic entities.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Contract` | Agreements | parties, terms, expiry, penalties |
| `Service` | Providable services | name, provider, cost, duration |
| `Quest` | Tasks to complete | name, giver, reward, deadline |
| `Debt` | Money owed | creditor, debtor, amount, due_date |
| `Shop` | Commercial establishments | name, owner, inventory, location |
| `Market` | Trading venues | location, schedule, vendors |

### 🎭 Events & Activities (6 types)
Things that happen in the world.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Event` | Significant occurrences | name, timestamp, participants, impact |
| `Festival` | Celebrations | name, date, traditions, organizer |
| `Ritual` | Ceremonies | name, purpose, participants, requirements |
| `Meeting` | Planned gatherings | topic, attendees, location, agenda |
| `Conflict` | Disputes | parties, cause, intensity, resolution |
| `Achievement` | Accomplishments | name, achiever, date, significance |

### 🌍 Environmental (4 types)
Natural and environmental elements.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Weather` | Weather conditions | type, severity, duration, effects |
| `Season` | Time periods | name, start, end, characteristics |
| `Resource` | Natural resources | type, location, quantity, quality |
| `Landmark` | Geographic features | name, location, significance |

### 💭 Emotional & Mental States (6 types)
Internal states and feelings.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Emotion` | Emotional states | type, intensity, trigger, duration |
| `Mood` | Long-term conditions | type, intensity, cause |
| `Dream` | Dreams and visions | content, dreamer, meaning |
| `Fear` | Specific fears | target, intensity, origin |
| `Desire` | Wants and goals | object, intensity, motivation |
| `Belief` | Core values | statement, strength, origin |

### 🏥 Health & Status (5 types)
Physical and social condition.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Illness` | Diseases | name, symptoms, severity, contagious |
| `Injury` | Physical wounds | type, location, severity, cause |
| `Blessing` | Positive effects | type, source, duration, effects |
| `Curse` | Negative effects | type, source, duration, effects |
| `Status` | Social standing | level, domain, recognition |

### 💬 Communication (6 types)
Messages and creative works.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Message` | Agent communications | sender, recipient, content, timestamp |
| `Announcement` | Public declarations | announcer, content, audience |
| `Gossip` | Informal spreading | content, source, credibility |
| `Letter` | Written correspondence | sender, recipient, content, delivered |
| `Song` | Musical compositions | title, composer, lyrics, mood |
| `Poem` | Poetic works | title, author, content, style |

### 🤝 Relationships & Connections (8 types)
Explicit relationship records.

| Node Type | Purpose | Example Properties |
|-----------|---------|-------------------|
| `Trust` | Trust relationships | level, reason, last_updated |
| `Friendship` | Friend connections | strength, since, shared_experiences |
| `Romance` | Romantic relationships | status, since, commitment_level |
| `Rivalry` | Competitive relationships | domain, intensity, origin |
| `Mentorship` | Teaching relationships | mentor, student, subject, progress |
| `Alliance` | Cooperative bonds | parties, purpose, strength |
| `Trade` | Commercial relationships | partners, frequency, satisfaction |
| `Employment` | Work relationships | employer, employee, role, satisfaction |

## Node Creation Examples

### Creating an Agent Node
```python
agent_node = await memory_manager.create_entity_node(
    node_type="Agent",
    node_id="agent_001",
    name="Marcus the Guard",
    properties={
        "role": "guard",
        "personality": {"brave": 0.8, "loyal": 0.9},
        "location": "town_square",
        "stats": {"health": 100, "energy": 80}
    }
)
```

### Creating a Memory Node
```python
memory_node = await memory_manager.create_episodic_node(
    content="Witnessed a heated argument at the market",
    timestamp=datetime.now(),
    agent_id="agent_001",
    node_type="Memory",
    metadata={
        "emotion": "concerned",
        "importance": 0.7,
        "location": "market"
    }
)
```

### Creating a Skill Node
```python
skill_node = await memory_manager.create_entity_node(
    node_type="Skill",
    node_id="skill_blacksmithing",
    name="Blacksmithing",
    properties={
        "category": "crafting",
        "difficulty": 0.7,
        "tools_required": ["hammer", "anvil", "forge"]
    }
)
```

## Node Relationships

Nodes are connected by edges (see MEMORY_EDGES.md) to form a rich knowledge graph:
- **Agents** connect to **Locations** via AT_LOCATION
- **Agents** connect to **Skills** via KNOWS_SKILL
- **Agents** connect to **Items** via ownership relationships
- **Memories** connect to **Agents** via REMEMBERS
- **Facts** and **Rumors** spread between **Agents**
- **Emotions** influence **Agent** decisions
- **Events** create **Memories** for witnessing **Agents**

## Best Practices

1. **Use appropriate node types**: Choose the most specific type for your entity
2. **Include temporal data**: Add timestamps for time-sensitive nodes
3. **Link related nodes**: Create edges to connect related information
4. **Track ownership**: Use properties to track who owns or controls nodes
5. **Maintain consistency**: Use the same ID format across node types
6. **Add metadata**: Include relevant properties for filtering and querying

## Querying Nodes

### Find all agents at a location
```cypher
MATCH (a:Agent)-[:AT_LOCATION]->(l:Location {name: "market"})
RETURN a.name, a.id
```

### Get all memories for an agent
```cypher
MATCH (a:Agent {id: $agent_id})-[:REMEMBERS]->(m:Memory)
RETURN m.content, m.timestamp, m.importance
ORDER BY m.timestamp DESC
```

### Find all items owned by an agent
```cypher
MATCH (a:Agent {id: $agent_id})-[:OWNS]->(i:Item)
RETURN i.name, i.value, i.type
```