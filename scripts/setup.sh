#!/bin/bash

# Setup script for LLM Agent Simulation

echo "🚀 Setting up LLM Agent Simulation Framework..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.10+ is required. Current version: $python_version"
    exit 1
fi

echo "✓ Python version check passed"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for Neo4j
echo "Checking Neo4j..."
if ! command -v neo4j &> /dev/null; then
    echo "⚠️  Neo4j not found. Please install Neo4j:"
    echo "   - macOS: brew install neo4j"
    echo "   - Linux: Follow https://neo4j.com/docs/operations-manual/current/installation/"
    echo "   - Docker: docker run -p 7474:7474 -p 7687:7687 neo4j:latest"
else
    echo "✓ Neo4j found"
fi

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your credentials"
fi

# Create necessary directories
mkdir -p logs checkpoints data

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your AWS and Neo4j credentials"
echo "2. Start Neo4j: neo4j start (or use Docker)"
echo "3. Run simulation: python -m sim run --agents 5 --days 10"
echo ""
echo "For help: python -m sim --help"