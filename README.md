# LLM Agent Simulation Framework

A multi-agent simulation where LLM-driven agents develop distinct personalities through interactions, memory formation, and knowledge propagation in a bounded world.

## Core Concepts

- **Autonomous Agents**: Each agent is powered by an LLM (OpenAI GPT-5) with unique personality traits and stats
- **Memory System**: Neo4j + Graphiti temporal knowledge graph stores experiences, relationships, and learned information
- **Limited Information**: One "sage" agent has web search (once daily) - knowledge must propagate through conversation
- **Emergent Personalities**: Agents evolve based on interactions, developing unique characteristics over time
- **Grid-based World**: Town and landmarks with location-aware interactions

## Architecture

- **Strands Agents SDK**: Model-driven agent framework with OpenAI integration
- **Strands A2A Protocol**: Agent-to-agent communication
- **Neo4j + Graphiti**: Temporal knowledge graph for real-time memory updates
- **Rate Limiting**: Token budget management for cost control

## Prerequisites

- Python 3.10+
- OpenAI API key
- Neo4j database (see setup below)
- 2GB+ RAM for Neo4j

## Complete Setup Guide

### Step 1: Clone and Install

```bash
# Clone repository
git clone <repo-url>
cd sim

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Neo4j Database Setup

Choose ONE of these options:

#### Option A: Docker (Recommended - Fastest)

```bash
# Start Neo4j with Docker
docker run --name neo4j-sim \
  --restart unless-stopped \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/simulation123 \
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  -v "$PWD/neo4j-data":/data \
  -d neo4j:5.26-community

# Wait for Neo4j to start (about 30 seconds)
sleep 30

# Verify it's running
docker logs neo4j-sim

# Neo4j Browser will be available at: http://localhost:7474
# Default credentials: neo4j / simulation123
```

#### Option B: Docker Compose (Full Stack)

```bash
# Start Neo4j and optional Grafana monitoring
docker-compose up -d

# Services:
# - Neo4j Browser: http://localhost:7474 (neo4j/simulation123)
# - Grafana: http://localhost:3000 (admin/admin)
```

#### Option C: Native Installation

**macOS (Homebrew):**
```bash
brew install neo4j
neo4j start
# Open http://localhost:7474
# Default: neo4j/neo4j (will prompt to change)
```

**Windows/Linux:**
Follow the [official installation guide](https://neo4j.com/docs/operations-manual/current/installation/)

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
nano .env  # or use your preferred editor
```

Required configuration in `.env`:

```bash
# LLM Provider
LLM_PROVIDER=openai

# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL_ID=gpt-5
OPENAI_SAGE_MODEL_ID=gpt-5

# Neo4j Configuration (REQUIRED)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=simulation123  # Or your chosen password

# Simulation Settings
SIMULATION_NAME=smallville_v1
MAX_AGENTS=5
MAX_DAYS=10

# Rate Limiting (Important for cost control)
DAILY_TOKEN_BUDGET=100000
PER_AGENT_TOKEN_LIMIT=20000
```

### Step 4: Verify Setup

```bash
# Test Neo4j connection
python -c "from neo4j import GraphDatabase; \
driver = GraphDatabase.driver('bolt://localhost:7687', \
auth=('neo4j', 'simulation123')); \
driver.verify_connectivity(); \
print('✓ Neo4j connected')"

# Test OpenAI
python -c "from openai import OpenAI; \
client = OpenAI(); \
print('✓ OpenAI configured')"
```

### Step 5: Run Simulation

```bash
# Basic run
python -m sim run --agents 5 --days 10

# With debug output
python -m sim run --agents 5 --days 3 --debug

# View help
python -m sim --help
```

## CLI Commands

```bash
# Run simulation
sim run --agents 5 --days 10

# View world map
sim world

# Check simulation stats
sim stats

# View latest checkpoint
sim view --latest

# Interview an agent
sim interview Aldric --latest

# Clean logs/checkpoints
sim clean
```

## Understanding the Simulation

### Agent Roles
- **Sage (Aldric)**: Keeper of knowledge with web search ability (1x/day)
- **Farmer (John)**: Works the fields, trades produce
- **Merchant (Mary)**: Trades goods, accumulates wealth
- **Artist (Luna)**: Creates art, expresses emotions
- **Guard (Marcus)**: Maintains order, patrols village

### World Locations
- **Library (L)**: Sage's home, knowledge hub
- **Market (M)**: Trading center
- **Temple (T)**: Spiritual center
- **Houses (H)**: Agent homes
- **Forest (F)**: Resource gathering
- **Square (S)**: Social gathering point

### Memory System
Agents use Graphiti's temporal knowledge graph to:
- Store conversations with timestamps
- Track relationships (trust, friendship)
- Remember facts and rumors
- Reflect on experiences
- Propagate knowledge through conversation

## Monitoring & Costs

### Token Usage
- Daily budget: 100,000 tokens (configurable)
- Per-agent limit: 20,000 tokens/day
- Estimated cost: ~$1-2 per simulated day with GPT-5

### Check Costs
```bash
# View token usage
sim stats

# Monitor in real-time
tail -f logs/token_usage_*.json
```

### Neo4j Monitoring
```bash
# Check database size
docker exec neo4j-sim cypher-shell -u neo4j -p simulation123 \
  "MATCH (n) RETURN count(n) as nodes"

# View recent memories
docker exec neo4j-sim cypher-shell -u neo4j -p simulation123 \
  "MATCH (m:Memory) RETURN m.content ORDER BY m.timestamp DESC LIMIT 10"
```

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check if Neo4j is running
docker ps | grep neo4j
# OR
neo4j status

# Restart Neo4j
docker restart neo4j-sim
# OR
neo4j restart

# Check logs
docker logs neo4j-sim
```

### OpenAI API Issues
- Verify API key is correct in `.env`
- Check API key has sufficient credits
- Ensure not hitting rate limits

### Memory Issues
- Neo4j requires ~2GB RAM minimum
- Reduce `MAX_AGENTS` if running out of memory
- Use `sim clean` to clear old data

## Development

### Project Structure
```
sim/
├── agents/          # Agent definitions and personalities
├── world/           # Grid system and locations
├── memory/          # Graphiti/Neo4j integration
├── orchestration/   # Simulation engine and A2A protocol
├── tools/           # Agent capabilities
└── config/          # Settings and configuration
```

### Adding New Agents
1. Define personality in `agents/personality.py`
2. Create agent config in `orchestration/engine.py`
3. Add role-specific goals and behaviors

### Extending Memory
The Graphiti integration in `memory/graphiti_manager.py` handles:
- Temporal knowledge graphs
- Bi-temporal event tracking
- Hybrid search (semantic + graph + full-text)
- Automatic reflection and compression

## Stop Services

```bash
# Stop simulation (Ctrl+C during run)

# Stop Neo4j Docker
docker stop neo4j-sim

# Stop all Docker Compose services
docker-compose down

# Clean up data (optional)
docker volume rm sim_neo4j_data
rm -rf neo4j-data/
```

## Project Structure

```
sim/
├── agents/          # Agent definitions and personas
├── world/           # Grid system and locations
├── memory/          # Neo4j integration and memory management
├── orchestration/   # Simulation engine and A2A protocol
├── tools/           # Agent capabilities (search, trade, etc.)
└── config/          # Configuration and environment settings
```
