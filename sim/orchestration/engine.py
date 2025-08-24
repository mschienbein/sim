"""
Main simulation engine that orchestrates agent interactions and world state.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import random
from collections import defaultdict
from itertools import combinations

from ..agents.base_agent import SimulationAgent
from ..agents.sage_agent import SageAgent
from ..agents.personality import PersonalityGenerator
from ..memory.graphiti_manager import GraphitiMemoryManager
from ..world.grid import WorldGrid, LocationType
from ..config.settings import settings
from .rate_limiter import TokenBudgetManager
from .conversation_manager import ConversationManager

class SimulationEngine:
    """
    Core simulation engine that manages the world, agents, and their interactions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the simulation engine"""
        self.config = config or {}
        
        # Core components
        self.world = WorldGrid(
            width=settings.simulation.world_size[0],
            height=settings.simulation.world_size[1]
        )
        self.memory_manager = GraphitiMemoryManager()
        self.token_manager = TokenBudgetManager()
        self.conversation_manager = ConversationManager()
        
        # Agent management
        self.agents: Dict[str, SimulationAgent] = {}
        self.agent_order: List[str] = []
        
        # Simulation state
        self.current_tick = 0
        self.current_day = 0
        self.running = False
        self.paused = False
        
        # Metrics and logging
        self.metrics = {
            "total_conversations": 0,
            "total_trades": 0,
            "total_movements": 0,
            "total_reflections": 0,
            "knowledge_propagated": 0,
            "tokens_used": 0,
            "emergent_events": []
        }
        
        self.conversation_logs = []
        self.event_log = []
    
    async def initialize(self):
        """Initialize all components and connections"""
        print("🌍 Initializing simulation world...")
        
        # Initialize memory manager with Graphiti
        await self.memory_manager.initialize()
        print("✓ Memory system (Graphiti + Neo4j) initialized")
        
        # Create agents
        await self._create_agents()
        print(f"✓ Created {len(self.agents)} agents")
        
        # Place agents in world
        self._place_agents()
        print("✓ Agents placed in world")
        
        # Initialize agent memories with backstories
        await self._initialize_agent_memories()
        print("✓ Agent memories initialized")
        
        print("🚀 Simulation ready to start!")
    
    async def _create_agents(self):
        """Create the initial set of agents"""
        agent_configs = [
            {
                "id": "sage_librarian",
                "name": "Aldric",
                "role": "sage",
                "archetype": "sage",
                "is_sage": True
            },
            {
                "id": "farmer_john",
                "name": "John",
                "role": "farmer",
                "archetype": "farmer"
            },
            {
                "id": "merchant_mary",
                "name": "Mary",
                "role": "merchant",
                "archetype": "merchant"
            },
            {
                "id": "artist_luna",
                "name": "Luna",
                "role": "artist",
                "archetype": "artist"
            },
            {
                "id": "guard_marcus",
                "name": "Marcus",
                "role": "guard",
                "archetype": "guard"
            }
        ]
        
        for config in agent_configs[:settings.simulation.max_agents]:
            if config.get("is_sage"):
                # Create the special sage agent
                agent = SageAgent(
                    agent_id=config["id"],
                    name=config["name"],
                    initial_location=(0, 0),  # Will be placed properly
                    memory_manager=self.memory_manager,
                    search_limit_per_day=settings.SAGE_SEARCH_LIMIT_PER_DAY
                )
            else:
                # Create regular agent
                personality = PersonalityGenerator.generate_archetype(config["archetype"])
                agent = SimulationAgent(
                    agent_id=config["id"],
                    name=config["name"],
                    role=config["role"],
                    personality=personality,
                    initial_location=(0, 0),  # Will be placed properly
                    memory_manager=self.memory_manager,
                    archetype=config["archetype"]
                )
            
            self.agents[config["id"]] = agent
            self.agent_order.append(config["id"])
    
    def _place_agents(self):
        """Place agents in appropriate starting locations"""
        # Place sage in library
        library_locs = self.world.get_locations_by_type(LocationType.LIBRARY)
        if library_locs:
            sage = self.agents.get("sage_librarian")
            if sage:
                sage.location = library_locs[0].position
                self.world.move_agent("sage_librarian", (0, 0), library_locs[0].position)
        
        # Place other agents in houses or appropriate locations
        placement_map = {
            "farmer": LocationType.HOUSES,
            "merchant": LocationType.MARKET,
            "artist": LocationType.GARDEN,
            "guard": LocationType.SQUARE
        }
        
        for agent_id, agent in self.agents.items():
            if agent_id == "sage_librarian":
                continue
            
            # Find appropriate location
            loc_type = placement_map.get(agent.role, LocationType.HOUSES)
            locations = self.world.get_locations_by_type(loc_type)
            
            if locations:
                # Pick random location of that type
                location = random.choice(locations)
                agent.location = location.position
                self.world.move_agent(agent_id, (0, 0), location.position)
    
    async def _initialize_agent_memories(self):
        """Give agents initial memories and relationships"""
        for agent_id, agent in self.agents.items():
            # Store backstory as initial memory
            await self.memory_manager.ingest_observation(
                agent_id=agent_id,
                observation=f"I am {agent.name}, a {agent.role}. {agent.backstory}",
                location=str(agent.location),
                importance=0.9
            )
            
            # Create initial relationships (everyone knows of each other)
            for other_id, other_agent in self.agents.items():
                if other_id != agent_id:
                    await self.memory_manager.ingest_knowledge(
                        agent_id=agent_id,
                        fact=f"{other_agent.name} is a {other_agent.role} in the village",
                        source="common_knowledge",
                        confidence=1.0,
                        verified=True
                    )
    
    async def run_simulation(self, days: int = 10):
        """Run the main simulation loop"""
        self.running = True
        print(f"\n🎭 Starting simulation for {days} days...")
        
        for day in range(days):
            if not self.running:
                break
            
            self.current_day = day
            print(f"\n📅 Day {day + 1}")
            
            for tick in range(settings.simulation.ticks_per_day):
                if not self.running or self.paused:
                    await self._handle_pause()
                    if not self.running:
                        break
                
                self.current_tick = day * settings.simulation.ticks_per_day + tick
                
                # Get world state
                world_state = self.world.get_world_state(self.current_tick)
                
                # Phase 1: Perception - all agents perceive environment
                perceptions = await self._gather_perceptions(world_state)
                
                # Phase 2: Decision - agents decide what to do
                decisions = await self._make_decisions(perceptions)
                
                # Phase 3: Action resolution
                await self._resolve_actions(decisions)
                
                # Phase 4: Interactions - handle conversations/trades
                await self._handle_interactions()
                
                # Phase 5: State updates
                self._update_world_state()
                self._update_agent_states(tick)
                
                # Phase 6: Periodic tasks
                if tick % 6 == 0:  # Every 6 ticks
                    await self._trigger_reflections()
                
                # Record metrics
                self._record_metrics()
                
                # Small delay for rate limiting
                await asyncio.sleep(settings.simulation.tick_duration_ms / 1000)
            
            # End of day processing
            await self._end_of_day_processing()
            print(f"Day {day + 1} complete. Tokens used today: {self.token_manager.used_today}")
        
        print("\n🏁 Simulation complete!")
        await self._print_final_summary()
    
    async def _gather_perceptions(self, world_state: Dict[str, Any]) -> Dict[str, Dict]:
        """Gather perceptions for all agents"""
        perceptions = {}
        
        for agent_id, agent in self.agents.items():
            # Get location info
            location_info = self.world.get_location_info(agent.location, agent_id)
            
            # Get agent context from Graphiti
            context = await self.memory_manager.get_agent_context(
                agent_id=agent_id,
                location=location_info["name"],
                nearby_agents=location_info.get("nearby_agents", [])
            )
            
            perceptions[agent_id] = {
                "world_state": world_state,
                "location": location_info,
                "context": context,
                "internal_state": {
                    "energy": agent.stats.energy,
                    "health": agent.stats.health,
                    "emotion": agent.emotions.get_dominant_emotion(),
                    "goals": agent.stats.short_term_goals[:2]
                }
            }
        
        return perceptions
    
    async def _make_decisions(self, perceptions: Dict[str, Dict]) -> Dict[str, Dict]:
        """Agents make decisions based on perceptions"""
        decisions = {}
        
        # Randomize order to prevent bias
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)
        
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            perception = perceptions[agent_id]
            
            # Check token budget
            if not self.token_manager.can_call_llm(agent_id, estimated_tokens=500):
                # Fallback to simple decision
                decisions[agent_id] = self._make_simple_decision(agent, perception)
                continue
            
            # Make LLM-based decision (would call agent.decide() in production)
            # For now, use heuristics
            decision = await self._make_agent_decision(agent, perception)
            decisions[agent_id] = decision
            
            # Track token usage
            self.token_manager.track_usage(agent_id, 500)  # Estimated
        
        return decisions
    
    async def _make_agent_decision(
        self,
        agent: SimulationAgent,
        perception: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make decision for an agent based on perception"""
        nearby_agents = perception["location"].get("nearby_agents", [])
        location_type = perception["location"]["type"]
        energy = perception["internal_state"]["energy"]
        
        # Priority-based decision making
        if energy < 20:
            return {"action": "rest", "reason": "Low energy"}
        
        if nearby_agents and random.random() < 0.6:
            # Social interaction
            target = random.choice(nearby_agents)
            if random.random() < 0.3:
                return {
                    "action": "trade",
                    "target": target,
                    "reason": "Trading opportunity"
                }
            else:
                return {
                    "action": "speak",
                    "target": target,
                    "message": "Hello! How are you today?",
                    "reason": "Social interaction"
                }
        
        if location_type in ["market", "square"] and energy > 50:
            return {"action": "work", "reason": "Good place to work"}
        
        if random.random() < 0.3:
            # Move to new location
            directions = ["north", "south", "east", "west"]
            return {
                "action": "move",
                "direction": random.choice(directions),
                "reason": "Exploring"
            }
        
        return {"action": "observe", "reason": "Taking in surroundings"}
    
    def _make_simple_decision(
        self,
        agent: SimulationAgent,
        perception: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback simple decision without LLM"""
        if agent.stats.energy < 30:
            return {"action": "rest", "reason": "Conserving tokens and energy"}
        return {"action": "observe", "reason": "Conserving tokens"}
    
    async def _resolve_actions(self, decisions: Dict[str, Dict]):
        """Execute agent actions"""
        for agent_id, decision in decisions.items():
            agent = self.agents[agent_id]
            action = decision.get("action", "observe")
            
            try:
                if action == "move":
                    direction = decision.get("direction", "north")
                    result = await agent.move(direction)
                    
                elif action == "speak":
                    target = decision.get("target")
                    message = decision.get("message", "Hello")
                    if target:
                        result = await agent.speak(target, message)
                        self.metrics["total_conversations"] += 1
                        
                elif action == "work":
                    result = await agent.work()
                    
                elif action == "rest":
                    result = await agent.rest()
                    
                elif action == "trade":
                    # Handle trade (simplified)
                    self.metrics["total_trades"] += 1
                    
                elif action == "observe":
                    result = await agent.observe()
                    
                # Log action
                self.event_log.append({
                    "tick": self.current_tick,
                    "agent": agent_id,
                    "action": action,
                    "decision": decision
                })
                
            except Exception as e:
                print(f"Error executing action for {agent_id}: {e}")
    
    async def _handle_interactions(self):
        """Handle agent interactions when they're in the same location"""
        # Group agents by location
        location_groups = defaultdict(list)
        for agent_id, agent in self.agents.items():
            location_groups[agent.location].append(agent_id)
        
        # Handle interactions in each location
        for location, agent_ids in location_groups.items():
            if len(agent_ids) < 2:
                continue
            
            # Probabilistic interactions
            for agent_a_id, agent_b_id in combinations(agent_ids, 2):
                if random.random() < 0.3:  # 30% chance of interaction
                    await self._run_conversation(agent_a_id, agent_b_id, location)
    
    async def _run_conversation(
        self,
        agent_a_id: str,
        agent_b_id: str,
        location: Tuple[int, int]
    ):
        """Run a conversation between two agents"""
        agent_a = self.agents[agent_a_id]
        agent_b = self.agents[agent_b_id]
        
        # Check energy
        if agent_a.stats.energy < 5 or agent_b.stats.energy < 5:
            return
        
        # Check token budget
        if not self.token_manager.can_call_llm(agent_a_id, 200):
            return
        if not self.token_manager.can_call_llm(agent_b_id, 200):
            return
        
        # Run conversation (simplified)
        dialogue = [
            {"speaker": agent_a.name, "message": "Hello! Nice to see you here."},
            {"speaker": agent_b.name, "message": "Hello! How has your day been?"},
            {"speaker": agent_a.name, "message": "Quite well, thank you for asking!"}
        ]
        
        # Store conversation in Graphiti
        await self.memory_manager.ingest_conversation(
            agent_a_id=agent_a_id,
            agent_b_id=agent_b_id,
            dialogue=dialogue,
            location=str(location)
        )
        
        # Update relationships
        agent_a.social.update_relationship(agent_b_id, {
            "trust": 0.05,
            "friendship": 0.03,
            "familiarity": 0.1
        })
        agent_b.social.update_relationship(agent_a_id, {
            "trust": 0.05,
            "friendship": 0.03,
            "familiarity": 0.1
        })
        
        # Update energy
        agent_a.stats.update_energy(-5)
        agent_b.stats.update_energy(-5)
        
        # Track tokens
        self.token_manager.track_usage(agent_a_id, 200)
        self.token_manager.track_usage(agent_b_id, 200)
        
        # Log conversation
        self.conversation_logs.append({
            "tick": self.current_tick,
            "participants": [agent_a_id, agent_b_id],
            "location": location,
            "dialogue": dialogue
        })
    
    def _update_world_state(self):
        """Update world state each tick"""
        # Weather changes, time progression, etc.
        pass
    
    def _update_agent_states(self, tick: int):
        """Update all agent states"""
        for agent in self.agents.values():
            agent.update_tick(tick, {})
    
    async def _trigger_reflections(self):
        """Trigger agent reflections periodically"""
        for agent_id, agent in self.agents.items():
            if agent.stats.energy > 30:
                await self.memory_manager.reflect_and_compress(agent_id)
                self.metrics["total_reflections"] += 1
    
    async def _end_of_day_processing(self):
        """Process end of day tasks"""
        print(f"  🌙 End of day {self.current_day + 1}")
        
        # Reset daily limits
        self.token_manager.reset_daily_budget()
        
        # Sage search reset
        sage = self.agents.get("sage_librarian")
        if sage and isinstance(sage, SageAgent):
            sage.searches_today = 0
        
        # Major reflection for all agents
        for agent_id in self.agents:
            await self.memory_manager.reflect_and_compress(agent_id)
        
        # Save checkpoint
        await self._save_checkpoint()
    
    async def _save_checkpoint(self):
        """Save simulation state"""
        checkpoint = {
            "day": self.current_day,
            "tick": self.current_tick,
            "agents": {
                agent_id: agent.to_save_state()
                for agent_id, agent in self.agents.items()
            },
            "metrics": self.metrics,
            "token_usage": self.token_manager.get_usage_report()
        }
        
        # Save to file
        filename = f"checkpoint_day_{self.current_day}.json"
        filepath = settings.checkpoints_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def _record_metrics(self):
        """Record simulation metrics"""
        self.metrics["tokens_used"] = self.token_manager.total_used
    
    async def _handle_pause(self):
        """Handle pause state"""
        while self.paused and self.running:
            await asyncio.sleep(0.1)
    
    async def _print_final_summary(self):
        """Print final simulation summary"""
        print("\n" + "="*50)
        print("📊 SIMULATION SUMMARY")
        print("="*50)
        
        print(f"\n🌍 World State:")
        print(f"  Days simulated: {self.current_day}")
        print(f"  Total ticks: {self.current_tick}")
        
        print(f"\n👥 Agent Activity:")
        print(f"  Total conversations: {self.metrics['total_conversations']}")
        print(f"  Total trades: {self.metrics['total_trades']}")
        print(f"  Total reflections: {self.metrics['total_reflections']}")
        
        print(f"\n🧠 Knowledge & Memory:")
        for agent_id, agent in self.agents.items():
            summary = await self.memory_manager.get_agent_summary(agent_id)
            print(f"  {agent.name} ({agent.role}):")
            print(f"    Memories: {summary['memory_count']}")
            print(f"    Knowledge: {summary['knowledge_count']}")
            print(f"    Relationships: {summary['relationship_count']}")
        
        print(f"\n💰 Token Usage:")
        print(f"  Total tokens: {self.metrics['tokens_used']}")
        print(f"  Estimated cost: ${self.metrics['tokens_used'] * 0.00001:.2f}")
        
        print("\n" + "="*50)
    
    async def stop(self):
        """Stop the simulation gracefully"""
        self.running = False
        await self._save_checkpoint()
        await self.memory_manager.close()
    
    def pause(self):
        """Pause the simulation"""
        self.paused = True
    
    def resume(self):
        """Resume the simulation"""
        self.paused = False