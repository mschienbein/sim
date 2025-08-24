# LLM Agent Simulation Framework

A multi-agent simulation where LLM-driven agents develop distinct personalities through interactions, memory formation, and knowledge propagation in a bounded world.

## Core Concepts

- **Autonomous Agents**: Each agent is powered by an LLM with unique personality traits and stats
- **Memory System**: Neo4j knowledge graph stores experiences, relationships, and learned information
- **Limited Information**: One "sage" agent has web search (once daily) - knowledge must propagate through conversation
- **Emergent Personalities**: Agents evolve based on interactions, developing unique characteristics over time
- **Grid-based World**: Town and landmarks with location-aware interactions

## Architecture

- **AWS Bedrock AgentCore**: Multi-agent orchestration and LLM integration
- **Strands SDK**: Agent-to-agent communication protocol
- **Neo4j**: Temporal knowledge graph for memory storage
- **Rate Limiting**: Token budget management for cost control

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials and Neo4j connection

# Start simulation
python -m sim.main --agents 5 --days 10
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
