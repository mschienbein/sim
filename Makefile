.PHONY: help up down restart logs clean setup monitoring neo4j redis

# Default target
help:
	@echo "Simulation Docker Management"
	@echo "============================"
	@echo "make setup       - Initial setup (create directories, copy env)"
	@echo "make up          - Start all services"
	@echo "make down        - Stop all services"
	@echo "make restart     - Restart all services"
	@echo "make logs        - View logs (all services)"
	@echo "make neo4j       - View Neo4j logs"
	@echo "make monitoring  - Open monitoring dashboards"
	@echo "make clean       - Clean up volumes and data"
	@echo "make status      - Check service status"

# Setup environment
setup:
	@echo "Setting up environment..."
	@mkdir -p monitoring/prometheus monitoring/grafana/provisioning/datasources 
	@mkdir -p monitoring/grafana/provisioning/dashboards monitoring/grafana/dashboards
	@mkdir -p notebooks data logs neo4j/conf
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from template"; fi
	@echo "Setup complete!"

# Start services
up:
	@echo "Starting services..."
	docker compose up -d
	@echo "Services started!"
	@echo "Neo4j Browser: http://localhost:7474"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jupyter: http://localhost:8888 (token: simulation123)"

# Stop services
down:
	@echo "Stopping services..."
	docker compose down
	@echo "Services stopped!"

# Restart services
restart:
	@echo "Restarting services..."
	docker compose restart
	@echo "Services restarted!"

# View logs
logs:
	docker compose logs -f --tail=100

# Neo4j specific commands
neo4j:
	docker compose logs -f neo4j --tail=100

neo4j-shell:
	docker exec -it sim-neo4j cypher-shell -u neo4j -p simulation123

neo4j-backup:
	@echo "Backing up Neo4j database..."
	@mkdir -p backups
	docker exec sim-neo4j neo4j-admin database dump --database=simulation --to-path=/data/
	docker cp sim-neo4j:/data/simulation.dump ./backups/simulation-$$(date +%Y%m%d-%H%M%S).dump
	@echo "Backup complete!"

# Redis commands
redis:
	docker compose logs -f redis --tail=100

redis-cli:
	docker exec -it sim-redis redis-cli

# Monitoring
monitoring:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Neo4j Browser: http://localhost:7474"
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000"

# Clean up
clean:
	@echo "Warning: This will delete all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		rm -rf neo4j_data grafana_data prometheus_data redis_data; \
		echo "Cleanup complete!"; \
	else \
		echo "Cleanup cancelled."; \
	fi

# Check status
status:
	@echo "Service Status:"
	@echo "==============="
	@docker compose ps
	@echo ""
	@echo "Port Availability:"
	@echo "=================="
	@echo "Neo4j Browser (7474): $$(nc -zv localhost 7474 2>&1 | grep -q succeeded && echo '✓ Available' || echo '✗ Not available')"
	@echo "Neo4j Bolt (7687): $$(nc -zv localhost 7687 2>&1 | grep -q succeeded && echo '✓ Available' || echo '✗ Not available')"
	@echo "Grafana (3000): $$(nc -zv localhost 3000 2>&1 | grep -q succeeded && echo '✓ Available' || echo '✗ Not available')"
	@echo "Prometheus (9090): $$(nc -zv localhost 9090 2>&1 | grep -q succeeded && echo '✓ Available' || echo '✗ Not available')"
	@echo "Redis (6379): $$(nc -zv localhost 6379 2>&1 | grep -q succeeded && echo '✓ Available' || echo '✗ Not available')"
	@echo "Jupyter (8888): $$(nc -zv localhost 8888 2>&1 | grep -q succeeded && echo '✓ Available' || echo '✗ Not available')"

# Development helpers
dev-setup: setup up
	@echo "Development environment ready!"
	@echo "Run 'make monitoring' to open dashboards"

dev-reset: down clean setup up
	@echo "Development environment reset complete!"