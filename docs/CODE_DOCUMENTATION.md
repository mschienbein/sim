# Code Documentation

Complete documentation for all source files in the simulation framework.

## Table of Contents

1. [Main Entry Point](#main-entry-point)
2. [Configuration](#configuration)
3. [Agent System](#agent-system)
4. [Memory System](#memory-system)
5. [Orchestration](#orchestration)
6. [World System](#world-system)
7. [Tools](#tools)

---

## Main Entry Point

### `src/main.py`

The main entry point for the simulation. Handles command-line arguments, initializes the simulation engine, and runs the simulation loop.

**Key Functions:**
- `main()`: Parses CLI arguments and starts simulation
- `run_simulation()`: Main simulation loop

**CLI Arguments:**
- `--agents`: Number of agents (default: 5)
- `--days`: Number of simulation days (default: 10)
- `--dashboard`: Enable live dashboard visualization
- `--checkpoint`: Resume from checkpoint

**Example Usage:**
```python
python -m src.main --agents 5 --days 3 --dashboard
```

---

## Configuration

### `src/config/settings.py`

Central configuration management using Pydantic settings. Loads environment variables and provides structured configuration objects.

**Configuration Classes:**

#### `LLMConfig`
- Provider selection (OpenAI/Bedrock)
- API keys and model IDs
- Default: GPT-5 for all agents
- Embedding model configuration

#### `Neo4jConfig`
- Database connection URI
- Authentication credentials
- Database name

#### `SimulationConfig`
- Max agents and days
- Ticks per day (24 = hourly)
- Tick duration in milliseconds
- World size (10x10 grid)

#### `RateLimitConfig`
- Daily token budget: 1,000,000 tokens
- Per-agent limit: 200,000 tokens
- Max conversation turns: 5

#### `PersonalityConfig`
- Big Five trait ranges (0.0-1.0)
- Emotion ranges and decay rates
- Energy and stress parameters

#### `WorldConfig`
- Location definitions with symbols
- Location capacities and resources
- Buffs and special properties
- Time of day effects

#### `Settings`
- Main aggregator class
- Creates all config instances
- Manages project paths
- Singleton instance: `settings`

---

## Agent System

### `src/agents/base_agent.py`

Base class for all simulation agents using Strands SDK.

**Key Components:**

#### `SimulationAgent` Class
- **Inherits**: Strands `Agent` class
- **Personality**: Big Five traits + emotions
- **Memory**: Neo4j graph integration
- **Position**: World grid location tracking
- **Tools**: Move, observe, speak, work, rest, trade, reflect

**Core Methods:**
- `__init__()`: Initialize agent with personality and starting position
- `perceive()`: Gather information about surroundings
- `decide()`: Choose next action based on personality and state
- `act()`: Execute chosen action
- `update_emotions()`: Modify emotional state based on events
- `form_memory()`: Create graph nodes for experiences
- `retrieve_memories()`: Query relevant past experiences
- `summarize_day()`: Daily reflection and memory consolidation

**Tool Methods** (decorated with `@tool`):
- `move()`: Navigate to new location
- `observe()`: Look around and describe environment
- `speak()`: Engage in conversation
- `work()`: Perform location-specific tasks
- `rest()`: Recover energy
- `trade()`: Exchange items
- `reflect()`: Deep introspection

### `src/agents/sage_agent.py`

Special knowledge-keeper agent with web search capability.

**Extends**: `SimulationAgent`

**Unique Features:**
- Web search ability (1x per day limit)
- Higher initial knowledge stats
- Special library location preference
- Knowledge artifact creation (scrolls)

**Additional Tools:**
- `web_search()`: Query the internet for information
- `create_scroll()`: Document knowledge as artifact
- `share_wisdom()`: Proactively share knowledge

**Web Search Management:**
- Tracks last search date
- Enforces daily limit
- Stores search results in memory graph

### `src/agents/personality.py`

Personality system based on Big Five model.

**Classes:**

#### `Personality`
- **Traits**: openness, conscientiousness, extraversion, agreeableness, neuroticism
- **Generation**: Random or predefined values
- **Influence**: Affects decision-making and behavior

#### `EmotionalState`
- **Emotions**: happiness, anger, fear, sadness, surprise
- **Physical**: energy (0-100), stress (0-1)
- **Decay**: Emotions naturally decrease over time
- **Updates**: Modified by events and interactions

**Utility Functions:**
- `generate_random_personality()`: Create random Big Five values
- `calculate_emotion_change()`: Determine emotional impact of events
- `apply_emotion_decay()`: Reduce emotion intensity over time

---

## Memory System

### `src/memory/manager.py`

Core memory management with Neo4j integration.

**Enums:**

#### `NodeType` (66 types)
Categories:
- Core Entities: Agent, Location, Item, Object
- Items & Objects: Tool, Weapon, Food, etc.
- Knowledge: Fact, Rumor, Secret, Belief
- Social: Group, Organization, Family
- Skills: Skill, Craft, Magic, Combat
- Economic: Currency, Trade, Contract
- Events: Event, Ceremony, Battle
- Environmental: Weather, Season, Resource
- Emotional: EmotionalState, Mood
- Health: Injury, Illness, Buff
- Communication: Message, Letter
- Relationships: Alliance, Rivalry

#### `EdgeType` (57 types)
- Social: LIKES, TRUSTS, FRIENDS_WITH
- Actions: SPOKE_WITH, TRADED, HELPED
- Knowledge: KNOWS, LEARNED_FROM
- Spatial: LOCATED_AT, TRAVELED_TO
- Temporal: HAPPENED_BEFORE/AFTER
- Emotional: FEELS_ABOUT, FEARS
- Economic: OWNS, SOLD_TO, BOUGHT_FROM

#### `MemoryType`
- EPISODIC: Personal experiences
- SEMANTIC: Facts and knowledge
- PROCEDURAL: Skills and how-to
- EMOTIONAL: Feelings and reactions
- SPATIAL: Location awareness
- SOCIAL: Relationship tracking

#### `MemoryManager` Class
- **Database**: Neo4j connection management
- **Node Creation**: Add entities to graph
- **Edge Creation**: Link entities with relationships
- **Retrieval**: Query memories by various criteria
- **Temporal**: Timestamp all memories
- **Decay**: Reduce memory strength over time
- **Consolidation**: Merge similar memories

**Key Methods:**
- `add_memory()`: Create new memory node
- `create_relationship()`: Link two memories
- `retrieve_recent()`: Get memories within time window
- `retrieve_by_type()`: Filter by memory type
- `retrieve_by_emotion()`: Find emotionally charged memories
- `search_semantic()`: Text-based memory search
- `get_agent_relationships()`: Social network analysis
- `propagate_knowledge()`: Spread information between agents

### `src/memory/graphiti_manager.py`

Advanced temporal knowledge graph using Graphiti framework.

**Features:**
- Bi-temporal tracking (valid time + system time)
- Entity disambiguation
- Fact extraction from conversations
- Relationship inference
- Memory importance scoring

**Classes:**

#### `GraphitiMemoryManager`
- **Client**: Graphiti client connection
- **Node Management**: Create/update temporal nodes
- **Edge Management**: Temporal relationships
- **Search**: Multi-faceted retrieval
- **Episodes**: Conversation tracking

**Core Methods:**
- `add_episode()`: Store complete conversation
- `add_node()`: Create temporal entity
- `add_edge()`: Create temporal relationship
- `search()`: Complex graph queries
- `get_neighbors()`: Find connected entities
- `update_node()`: Modify with temporal tracking
- `extract_facts()`: Parse facts from text
- `infer_relationships()`: Deduce connections

**Temporal Features:**
- Valid time: When fact was true
- System time: When fact was recorded
- Time travel queries: State at any point
- Version history: Track all changes

---

## Orchestration

### `src/orchestration/engine.py`

Main simulation engine coordinating all components.

**Classes:**

#### `SimulationEngine`
- **World**: Grid-based environment
- **Agents**: Agent pool management
- **Memory**: Graph database coordination
- **Events**: Event queue processing
- **Metrics**: Performance tracking

**Core Methods:**
- `initialize()`: Setup world and spawn agents
- `run()`: Main simulation loop
- `tick()`: Single time step execution
- `process_agent_turn()`: Individual agent actions
- `handle_conversations()`: Multi-agent interactions
- `update_world_state()`: Environmental changes
- `save_checkpoint()`: Persistence
- `load_checkpoint()`: Resume from save

**Event System:**
- Event queue for asynchronous processing
- Priority-based event handling
- Event types: conversation, trade, combat
- Event propagation to nearby agents

### `src/orchestration/conversation_manager.py`

Manages multi-agent conversations using Strands A2A.

**Classes:**

#### `ConversationManager`
- **Sessions**: Track active conversations
- **Turn Management**: Coordinate speaking order
- **Context**: Maintain conversation history
- **Moderation**: Filter inappropriate content

**Methods:**
- `initiate_conversation()`: Start new dialogue
- `add_participant()`: Include new agent
- `process_turn()`: Handle single utterance
- `end_conversation()`: Graceful termination
- `get_conversation_summary()`: Extract key points
- `apply_conversation_effects()`: Update relationships

**Conversation Flow:**
1. Initiator approaches target
2. Check if target is available
3. Generate greeting based on relationship
4. Exchange turns (max 5 per conversation)
5. Update memories and relationships
6. Apply emotional effects

### `src/orchestration/rate_limiter.py`

Token budget management and cost optimization.

**Classes:**

#### `TokenBudgetManager`
- **Daily Budget**: 1M tokens ($5/day)
- **Per-Agent Limit**: 200k tokens
- **Tracking**: Real-time usage monitoring
- **History**: Usage analytics

**Methods:**
- `can_call_llm()`: Check budget availability
- `track_usage()`: Record token consumption
- `reset_daily_budget()`: Midnight reset
- `get_remaining_budget()`: Current availability
- `save_daily_summary()`: Export usage report

#### `ConversationLimiter`
- **Max Turns**: 5 per conversation
- **Tracking**: Active conversation monitoring
- **Enforcement**: Automatic termination

#### `ActionThrottler`
- **Cooldowns**: Per-action rate limits
- **Web Search**: 24-hour cooldown
- **Reflect**: 1-hour cooldown
- **Trade**: 10-minute cooldown

#### `CostOptimizer`
- **Model Selection**: Dynamic based on budget
- **GPT-5**: Primary model ($0.005/1k tokens)
- **GPT-4o**: Fallback ($0.0025/1k tokens)
- **GPT-4o-mini**: Emergency ($0.00015/1k tokens)

**Optimization Strategy:**
- High priority (reflect, search): GPT-5 > 20% budget
- Medium priority (speak, trade): GPT-5 > 60% budget
- Low priority (move, observe): GPT-5 > 80% budget
- Automatic downgrade as budget depletes

---

## World System

### `src/world/grid.py`

Spatial environment with locations and movement.

**Classes:**

#### `LocationType` (Enum)
16 location types: Forest, Houses, Market, Temple, Library, Park, Square, Cafe, Inn, River, Blacksmith, Apothecary, Garden, Waterfront, Docks, Path

#### `Location`
- **Position**: (x, y) coordinates
- **Type**: LocationType enum
- **Capacity**: Max occupants
- **Occupants**: Current agents present
- **Resources**: Available items/activities
- **Buffs**: Stat modifiers for occupants
- **Properties**: Special flags (economic_hub, sage_home)

#### `WorldGrid`
- **Size**: 10x10 grid
- **Layout**: Predefined location placement
- **Navigation**: A* pathfinding
- **Spatial Queries**: Nearby agents/locations
- **NumPy Arrays**: Efficient spatial operations

**Key Methods:**
- `get_location()`: Retrieve by coordinates
- `get_nearby_locations()`: Radius search
- `move_agent()`: Update positions
- `get_path()`: A* pathfinding
- `find_nearest_location_type()`: Closest of type
- `get_world_state()`: Time, weather, stats
- `visualize_grid()`: ASCII representation

**NumPy Optimizations:**
- `location_grid`: Type indices
- `occupancy_grid`: Agent counts
- `walkability_grid`: Movement constraints
- `get_distance_matrix()`: All distances from point
- `get_reachable_area()`: Movement range
- `get_population_density()`: Heat map

**World Layout:**
```
FFFHHMMTTL  (F=Forest, H=Houses, M=Market, T=Temple, L=Library)
F..HHMMTTL  (.=Path)
..PPSSCC.L  (P=Park, S=Square, C=Cafe)
..PPSSCC..
RR..II..GG  (R=River, I=Inn, G=Garden)
RR..II..GG
..BB..AA..  (B=Blacksmith, A=Apothecary)
..BB..AA..
WWW.....DD  (W=Waterfront, D=Docks)
WWW.....DD
```

---

## Tools

### `src/tools/agent_tools.py`

Shared tool implementations for agents.

**Tool Categories:**

#### Movement Tools
- `move_toward()`: Navigate to target
- `wander()`: Random exploration
- `return_home()`: Go to house location
- `flee()`: Escape from threat

#### Social Tools
- `greet()`: Initialize conversation
- `gossip()`: Share rumors
- `compliment()`: Positive interaction
- `argue()`: Disagreement

#### Economic Tools
- `buy_item()`: Purchase goods
- `sell_item()`: Offer for sale
- `negotiate_price()`: Haggle
- `check_inventory()`: List possessions

#### Work Tools
- `gather_resources()`: Location-specific
- `craft_item()`: Create from materials
- `perform_service()`: Offer skills
- `train_skill()`: Improve ability

**Tool Decorators:**
- `@tool`: Strands tool registration
- `@requires_energy`: Check energy level
- `@location_specific`: Verify location type
- `@social_check`: Relationship requirements

---

## Module Initialization

### `src/__init__.py`

Package initialization file. Currently empty but reserves namespace for future exports.

**Potential Future Exports:**
```python
from .main import run_simulation
from .orchestration.engine import SimulationEngine
from .agents.base_agent import SimulationAgent
from .config.settings import settings

__version__ = "1.0.0"
__all__ = ["run_simulation", "SimulationEngine", "SimulationAgent", "settings"]
```