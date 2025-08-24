# LLM Agent Simulation Framework

A sophisticated multi-agent simulation where AI agents develop distinct personalities through interactions, form memories using a temporal knowledge graph, and evolve their behaviors over time.

## 🌟 Features

- **Personality Evolution**: Agents develop unique personalities based on Big Five traits
- **Temporal Memory**: Neo4j + Graphiti for bi-temporal knowledge graphs
- **Smart Conversations**: Strands A2A protocol for autonomous agent interactions  
- **Special Sage Agent**: One agent with limited web search capabilities (1x/day)
- **Token Management**: Built-in rate limiting to control LLM costs
- **Rich Monitoring**: Grafana dashboards for real-time simulation metrics
- **Live Dashboard**: Real-time CLI visualization of agent activities

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

### Key Components
- **Agents**: Autonomous entities with personalities, memories, and tools
- **World Grid**: 10x10 spatial environment with locations and resources
- **Memory System**: Neo4j graph database with temporal awareness
- **Orchestration**: Event-driven simulation with conversation management
- **Monitoring**: Prometheus + Grafana for metrics and visualization

## 🚀 Quick Start with Docker Compose

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- Make (for convenience commands)

### Setup

1. **Clone the repository**
```bash
git clone <repository>
cd sim
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. **Start services with Docker Compose**
```bash
# Initial setup (creates directories, copies env)
make setup

# Start all services
make up
```

This starts:
- Neo4j (http://localhost:7474) - Graph database
- Grafana (http://localhost:3000) - Metrics dashboard (admin/admin)
- Prometheus (http://localhost:9090) - Metrics collection
- Redis - Caching layer
- Jupyter (http://localhost:8888) - Analysis notebooks (token: simulation123)

4. **Run the simulation**
```bash
# In a separate terminal, with services running
python -m src.main --agents 5 --days 3
```

## 🛠️ Alternative Setup (Without Docker)

If you prefer to run services locally:

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Neo4j Setup

#### Option A: Docker (Just Neo4j)

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

#### Option B: Native Installation

**macOS (Homebrew):**
```bash
brew install neo4j
neo4j start
# Open http://localhost:7474
# Default: neo4j/neo4j (will prompt to change)
```

**Windows/Linux:**
Follow the [official installation guide](https://neo4j.com/docs/operations-manual/current/installation/)


### Step 3: Run Simulation

```bash
# Basic run
python -m src.main --agents 5 --days 10

# With monitoring dashboard
python -m src.main --agents 5 --days 3 --dashboard
```

## 📝 Configuration

Edit `.env` file for customization:

```env
# LLM Configuration
OPENAI_API_KEY=your-api-key
OPENAI_MODEL_ID=gpt-4
OPENAI_SAGE_MODEL_ID=gpt-4

# Simulation Parameters
MAX_AGENTS=5
MAX_DAYS=10
TICKS_PER_DAY=24

# Rate Limiting
DAILY_TOKEN_BUDGET=100000
PER_AGENT_TOKEN_LIMIT=20000

# Neo4j (automatically configured by docker-compose)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=simulation123
```

## 🛠️ Useful Commands

```bash
# Service management
make up          # Start all services
make down        # Stop all services
make restart     # Restart services
make status      # Check service status
make logs        # View all logs
make neo4j       # View Neo4j logs

# Database
make neo4j-shell # Access Neo4j cypher shell
make neo4j-backup # Backup Neo4j data

# Monitoring
make monitoring  # Open Grafana dashboard

# Cleanup
make clean       # Remove all data (careful!)
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
- Estimated cost: ~$1-2 per simulated day with GPT-4

### Grafana Dashboard
Access at http://localhost:3000 (admin/admin)
- Agent activity metrics
- Memory formation rates
- Token usage tracking
- Conversation statistics

### Neo4j Browser
Access at http://localhost:7474
- Visualize agent memory graphs
- Query temporal relationships
- Explore knowledge networks

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check if Neo4j is running
make status

# Restart Neo4j
make restart

# Check logs
make neo4j
```

### OpenAI API Issues
- Verify API key is correct in `.env`
- Check API key has sufficient credits
- Ensure not hitting rate limits
- Use GPT-4 instead of GPT-5 if model not available

### Memory Issues
- Neo4j requires ~2GB RAM minimum
- Reduce `MAX_AGENTS` if running out of memory
- Use `make clean` to clear old data (careful!)

## 🧑‍🔬 Agent Types

### Regular Agents
- Develop personalities through interactions
- Form memories and relationships
- Make decisions based on personality traits
- Engage in autonomous conversations

### Sage Agent
- Special knowledge-keeper role
- Limited web search ability (1x/day)
- Higher initial knowledge level
- Shares wisdom with other agents
- Creates knowledge artifacts (scrolls)

## 🔧 Extending the Framework

### Adding New Agent Types
Create a new class extending `SimulationAgent` in `src/agents/`:
```python
from src.agents.base_agent import SimulationAgent

class MerchantAgent(SimulationAgent):
    def __init__(self, ...):
        super().__init__(...)
        # Add custom initialization
```

### Adding New Tools
Use the Strands `@tool` decorator:
```python
@tool
async def custom_action(self, parameter: str) -> str:
    """Tool description"""
    # Implementation
    return result
```

### Adding New Locations
Edit `src/world/grid.py` to add location types and their properties.

## 📁 Project Structure

```
sim/
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   ├── memory/            # Memory management
│   ├── world/             # World grid system
│   ├── orchestration/     # Simulation engine
│   └── config/            # Configuration
├── monitoring/            # Grafana dashboards
├── docker-compose.yml     # Service definitions
├── Makefile              # Convenience commands
└── .env                  # Configuration (create from .env.example)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `make status` and run simulation
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with [Strands Agents SDK](https://github.com/strands-ai/strands)
- Memory system inspired by Graphiti
- Simulation concepts from Stanford's Smallville paper
