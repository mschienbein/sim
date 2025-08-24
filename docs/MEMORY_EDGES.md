# Memory Edge Types Documentation

This document describes all the relationship edge types available in the simulation's memory system. These edges connect nodes in the temporal knowledge graph and enable rich, nuanced agent interactions.

## Categories of Edge Types

### 🗣️ Social Interactions
Direct interactions between agents that form the basis of their social lives.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `SPOKE_WITH` | Records conversations between agents | timestamp, sentiment |
| `OBSERVED` | Agent noticed another's action | timestamp, importance |
| `GREETED` | Friendly acknowledgment | timestamp, warmth |
| `ARGUED_WITH` | Disagreement or conflict | timestamp, intensity, topic |
| `HELPED` | Assistance given | timestamp, task, gratitude |
| `TAUGHT` | Knowledge transfer | timestamp, subject, effectiveness |
| `COLLABORATED_WITH` | Working together | timestamp, project, harmony |

### ❤️ Emotional Relationships
Long-term emotional connections that affect decision-making.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `LIKES` | Positive feelings toward someone | intensity, since, reason |
| `DISLIKES` | Negative feelings | intensity, since, reason |
| `LOVES` | Deep affection (romantic/platonic/familial) | intensity, since, type |
| `FEARS` | Agent is afraid of someone | intensity, since, trigger |
| `ADMIRES` | Respect for specific qualities | aspect, intensity, since |
| `ENVIES` | Jealousy of another's attributes | reason, intensity, since |
| `RESPECTS` | Professional/moral regard | level, reason, since |
| `FEELS_SORRY_FOR` | Sympathy or pity | reason, intensity, timestamp |

### 🤝 Trust & Social Standing
Trust dynamics and social hierarchies.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `TRUSTS` | Confidence in reliability | weight, last_updated |
| `DISTRUSTS` | Lack of confidence | reason, intensity, since |
| `BETRAYED` | Trust violation | timestamp, severity, forgiven |
| `FORGAVE` | Reconciliation after offense | timestamp, offense |
| `ALLIED_WITH` | Strategic partnership | strength, purpose, since |
| `RIVALS_WITH` | Competition or opposition | domain, intensity, since |

### 🧠 Memory & Knowledge
How information is stored and connected.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `REMEMBERS` | Agent recalls something | timestamp, strength, emotion |
| `LEARNED_FROM` | Knowledge acquisition | timestamp, confidence |
| `FORGOT` | Memory decay | timestamp, importance |
| `REMINDS_OF` | Associative memory | similarity, emotional_weight |
| `CONTRADICTS` | Conflicting information | confidence, timestamp |

### 💰 Economic & Trade
Commercial relationships and transactions.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `OWES` | Debt relationship | amount, due_date, currency |
| `TRADED` | Exchange of goods | timestamp, items, satisfaction |
| `GIFTED` | Free transfer | timestamp, item, occasion |
| `STOLE_FROM` | Theft (if discovered) | timestamp, item, discovered |
| `EMPLOYED_BY` | Work relationship | since, role, satisfaction |
| `COMMISSIONED` | Contract work | timestamp, task, payment |

### 🚶 Activities & Actions
Movement and location-based activities.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `AT_LOCATION` | Current presence | since, purpose, mood |
| `TRAVELED_TO` | Movement between places | timestamp, from, reason |
| `AVOIDED` | Deliberate avoidance | timestamp, reason |
| `VISITED` | Temporary presence | timestamp, duration, purpose |
| `PERFORMED_FOR` | Entertainment/service | timestamp, type, reception |
| `COMPETED_WITH` | Contest participation | timestamp, contest, outcome |

### 💬 Information & Communication
How information flows through the village.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `WROTE` | Created written content | timestamp, content_type, length |
| `READ` | Consumed written content | timestamp, comprehension, enjoyment |
| `HEARD_FROM` | Received information | timestamp, credibility, topic |
| `GOSSIPED_ABOUT` | Spread rumors | timestamp, topic, malicious |
| `SHARED_SECRET` | Confidential information | timestamp, trust_level |
| `LIED_TO` | Deception | timestamp, topic, discovered |

### 💭 Beliefs & Opinions
What agents think and believe.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `BELIEVES` | Holds as true | confidence, source, since |
| `DOUBTS` | Questions validity | reason, strength, since |
| `AGREES_WITH` | Shared opinion | topic, strength, timestamp |
| `DISAGREES_WITH` | Opposing view | topic, strength, timestamp |
| `RUMOR_OF` | Unverified information | confidence, hops, timestamp |

### 🎓 Skills & Development
Personal growth and learning.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `KNOWS_SKILL` | Ability possession | level, learned_date, teacher |
| `MENTORED_BY` | Learning relationship | since, domain, progress |
| `INSPIRED_BY` | Motivational influence | timestamp, aspect, impact |
| `LEARNED_RECIPE` | Specific knowledge | timestamp, dish, mastery |

### 👥 Community & Groups
Social organization and belonging.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `PART_OF` | Group membership | role, joined, status |
| `LEADS` | Leadership position | since, approval, style |
| `FOLLOWS` | Subordinate relationship | since, loyalty, reason |
| `BANISHED_FROM` | Exclusion | timestamp, reason, duration |
| `WELCOMED_BY` | Acceptance | timestamp, warmth |

### ⚔️ Conflicts & Resolutions
Disputes and their resolutions.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `FOUGHT_WITH` | Physical/verbal conflict | timestamp, reason, outcome |
| `MADE_PEACE_WITH` | Conflict resolution | timestamp, mediator |
| `CHALLENGED` | Formal dispute | timestamp, type, accepted |
| `DEFENDED` | Protection given | timestamp, from, success |

### 🎭 Life Events
Significant moments and shared experiences.

| Edge Type | Purpose | Key Attributes |
|-----------|---------|----------------|
| `WITNESSED` | Observed important event | timestamp, event, impact |
| `CELEBRATED_WITH` | Shared joy | timestamp, occasion, joy |
| `MOURNED_WITH` | Shared grief | timestamp, loss, grief |
| `BLESSED_BY` | Positive supernatural | timestamp, type, power |
| `CURSED_BY` | Negative supernatural | timestamp, type, severity |

## Usage Examples

### Creating Emotional Bonds
```python
# After a positive interaction
await memory_manager.create_edge(
    from_node=agent_a,
    to_node=agent_b,
    edge_type="LIKES",
    attributes={
        "intensity": 0.7,
        "since": datetime.now(),
        "reason": "helped with harvest"
    }
)
```

### Recording Conflicts
```python
# During an argument
await memory_manager.create_edge(
    from_node=agent_a,
    to_node=agent_b,
    edge_type="ARGUED_WITH",
    attributes={
        "timestamp": datetime.now(),
        "intensity": 0.8,
        "topic": "market prices"
    }
)
```

### Economic Relationships
```python
# After a trade
await memory_manager.create_edge(
    from_node=buyer,
    to_node=seller,
    edge_type="TRADED",
    attributes={
        "timestamp": datetime.now(),
        "items": {"wheat": 5, "coins": 10},
        "satisfaction": 0.9
    }
)
```

## Edge Weight Evolution

Many edges evolve over time:
- **Trust edges** increase with positive interactions, decrease with betrayals
- **Like/Dislike** intensity changes based on accumulated experiences
- **Skills** improve with practice and mentorship
- **Rivalries** intensify with competition
- **Memories** decay over time unless reinforced

## Querying Relationships

### Find who an agent likes
```cypher
MATCH (a:Agent {id: $agent_id})-[r:LIKES]->(other:Agent)
RETURN other, r.intensity, r.reason
ORDER BY r.intensity DESC
```

### Track trust networks
```cypher
MATCH (a:Agent)-[r:TRUSTS]->(b:Agent)
WHERE r.weight > 0.7
RETURN a, b, r.weight
```

### Discover conflicts
```cypher
MATCH (a:Agent)-[r:ARGUED_WITH|FOUGHT_WITH|RIVALS_WITH]->(b:Agent)
WHERE r.timestamp > datetime() - duration('P7D')
RETURN a, b, type(r), r.intensity
```

## Best Practices

1. **Use appropriate granularity**: Not every interaction needs every edge type
2. **Update existing edges**: Modify weights rather than creating duplicates
3. **Consider bidirectionality**: Some relationships are mutual (FRIENDS), others directed (ADMIRES)
4. **Track temporal changes**: Use timestamps to understand relationship evolution
5. **Combine edges for complex relationships**: An agent might both LIKES and RIVALS_WITH another
6. **Use edge attributes for agent decision-making**: High trust leads to more cooperation
7. **Decay unused relationships**: Reduce intensity of edges that haven't been reinforced

## Integration with Agent Behavior

Agents should query their relationship edges when making decisions:
- Check `TRUSTS` before sharing secrets
- Consider `LIKES/DISLIKES` when choosing conversation partners
- Use `RIVALS_WITH` to add competitive dialogue
- Reference `OWES` for economic decisions
- Check `FEARS` to avoid certain agents or situations
- Use `MENTORED_BY` to seek advice from teachers