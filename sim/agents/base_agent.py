"""
Base agent implementation using Strands SDK with personality and memory.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
import json
from datetime import datetime

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from .personality import (
    Personality, EmotionalProfile, SocialProfile,
    PersonalityGenerator, Relationship
)
from ..memory.manager import MemoryManager
from ..config.settings import settings

@dataclass
class AgentStats:
    """Physical and economic stats for an agent"""
    
    health: float = 100.0
    energy: float = 100.0
    hunger: float = 0.0
    thirst: float = 0.0
    
    # Economy
    gold: int = 50
    silver: int = 100
    reputation: float = 0.5
    
    # Knowledge
    knowledge_level: float = 0.1
    skills: Dict[str, float] = field(default_factory=dict)
    known_facts: List[str] = field(default_factory=list)
    beliefs: List[str] = field(default_factory=list)
    rumors: List[str] = field(default_factory=list)
    
    # Goals
    short_term_goals: List[str] = field(default_factory=list)
    long_term_goals: List[str] = field(default_factory=list)
    
    def update_energy(self, amount: float):
        """Update energy with bounds checking"""
        self.energy = max(0.0, min(100.0, self.energy + amount))
    
    def update_health(self, amount: float):
        """Update health with bounds checking"""
        self.health = max(0.0, min(100.0, self.health + amount))
    
    def can_afford(self, gold: int = 0, silver: int = 0) -> bool:
        """Check if agent can afford a cost"""
        total_silver = self.silver + (self.gold * 100)
        cost_silver = silver + (gold * 100)
        return total_silver >= cost_silver
    
    def spend(self, gold: int = 0, silver: int = 0) -> bool:
        """Spend money if affordable"""
        if not self.can_afford(gold, silver):
            return False
        
        total_silver = self.silver + (self.gold * 100)
        cost_silver = silver + (gold * 100)
        remaining = total_silver - cost_silver
        
        self.gold = remaining // 100
        self.silver = remaining % 100
        return True
    
    def earn(self, gold: int = 0, silver: int = 0):
        """Add money to wallet"""
        total_silver = self.silver + (self.gold * 100) + silver + (gold * 100)
        self.gold = total_silver // 100
        self.silver = total_silver % 100

@dataclass
class AgentInventory:
    """Agent inventory management"""
    
    capacity: int = 20
    items: Dict[str, int] = field(default_factory=dict)
    
    def add_item(self, item_id: str, quantity: int = 1) -> bool:
        """Add items to inventory"""
        current_weight = sum(self.items.values())
        if current_weight + quantity > self.capacity:
            return False
        
        self.items[item_id] = self.items.get(item_id, 0) + quantity
        return True
    
    def remove_item(self, item_id: str, quantity: int = 1) -> bool:
        """Remove items from inventory"""
        if item_id not in self.items:
            return False
        if self.items[item_id] < quantity:
            return False
        
        self.items[item_id] -= quantity
        if self.items[item_id] == 0:
            del self.items[item_id]
        return True
    
    def has_item(self, item_id: str, quantity: int = 1) -> bool:
        """Check if inventory has items"""
        return self.items.get(item_id, 0) >= quantity
    
    def get_weight(self) -> int:
        """Get total inventory weight"""
        return sum(self.items.values())

class SimulationAgent(Agent):
    """Base agent class for the simulation"""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        personality: Personality,
        initial_location: Tuple[int, int],
        memory_manager: MemoryManager,
        backstory: str = "",
        archetype: Optional[str] = None,
        model: Optional[OpenAIModel] = None
    ):
        # Build system prompt from personality and role
        system_prompt = self._build_system_prompt(name, role, personality, backstory)
        
        # Create OpenAI model if not provided
        if model is None:
            model = OpenAIModel(
                client_args={
                    "api_key": settings.llm.openai_api_key,
                },
                model_id=settings.llm.openai_model_id,
                params={
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
            )
        
        # Initialize Strands Agent
        super().__init__(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=self._get_tools()
        )
        
        # Core attributes
        self.agent_id = agent_id
        self.role = role
        self.backstory = backstory
        self.archetype = archetype
        
        # State
        self.personality = personality
        self.emotions = EmotionalProfile(baseline_happiness=0.5 + personality.agreeableness * 0.2)
        self.social = SocialProfile()
        self.stats = AgentStats()
        self.inventory = AgentInventory()
        
        # Location
        self.location = initial_location
        self.destination = None
        self.path = []
        
        # Memory
        self.memory_manager = memory_manager
        self.short_term_memory = []  # Recent events
        self.conversation_history = []
        
        # Action management
        self.current_action = None
        self.action_queue = []
        self.last_action_tick = 0
        
        # Initialize goals based on role
        self._initialize_goals()
    
    def _build_system_prompt(
        self,
        name: str,
        role: str,
        personality: Personality,
        backstory: str
    ) -> str:
        """Build the system prompt for the LLM"""
        
        personality_desc = personality.to_prompt_description()
        
        prompt = f"""You are {name}, a {role} in a small village.

PERSONALITY: You are {personality_desc}.

BACKSTORY: {backstory if backstory else f"You have lived in this village for many years, working as a {role}."}

BEHAVIOR GUIDELINES:
1. Respond naturally based on your personality and current emotional state
2. Consider your relationships with other agents when interacting
3. Make decisions that align with your goals and values
4. Remember past interactions and learn from them
5. Express emotions appropriately based on events
6. Manage your energy and needs (hunger, rest, etc.)

When making decisions or responding:
- Consider your current location and surroundings
- Think about your relationships with others
- Act according to your personality traits
- Pursue your goals while maintaining character consistency

You can perform various actions like moving, speaking, trading, working, and more.
Always stay in character and respond as {name} would."""
        
        return prompt
    
    def _get_tools(self) -> List:
        """Get the list of tools available to this agent"""
        tools = [
            self.speak,
            self.move,
            self.observe,
            self.remember,
            self.trade,
            self.work,
            self.rest,
            self.reflect
        ]
        return tools
    
    def _initialize_goals(self):
        """Initialize agent goals based on role and personality"""
        # Role-based goals
        role_goals = {
            "farmer": [
                "Grow and harvest crops",
                "Trade produce at the market",
                "Maintain good relationships with neighbors"
            ],
            "merchant": [
                "Buy low and sell high",
                "Build trade relationships",
                "Accumulate wealth"
            ],
            "sage": [
                "Seek and share knowledge",
                "Help others with wisdom",
                "Discover new information"
            ],
            "artist": [
                "Create beautiful works",
                "Express emotions through art",
                "Find inspiration in daily life"
            ],
            "guard": [
                "Maintain order and safety",
                "Patrol the village",
                "Protect citizens"
            ]
        }
        
        # Set role-specific goals
        if self.role in role_goals:
            self.stats.long_term_goals = role_goals[self.role]
        
        # Personality-based goals
        if self.personality.openness > 0.7:
            self.stats.short_term_goals.append("Explore new areas")
        if self.personality.extraversion > 0.7:
            self.stats.short_term_goals.append("Meet and talk with others")
        if self.personality.conscientiousness > 0.7:
            self.stats.short_term_goals.append("Complete daily tasks efficiently")
    
    @tool
    async def speak(self, target: str, message: str) -> str:
        """Speak to another agent"""
        if self.stats.energy < 2:
            return "Too tired to speak"
        
        self.stats.update_energy(-2)
        
        # Store conversation in memory
        memory_content = f"Said to {target}: {message}"
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="conversation",
            content=memory_content,
            metadata={
                "target": target,
                "location": str(self.location),
                "emotion": self.emotions.get_dominant_emotion()
            }
        )
        
        # Update conversation history
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "target": target,
            "message": message,
            "type": "outgoing"
        })
        
        return f"Spoke to {target}: {message}"
    
    @tool
    async def move(self, direction: str) -> str:
        """Move in a direction (north, south, east, west)"""
        if self.stats.energy < 5:
            return "Too tired to move"
        
        direction_map = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0)
        }
        
        if direction not in direction_map:
            return f"Invalid direction: {direction}"
        
        dx, dy = direction_map[direction]
        new_x = self.location[0] + dx
        new_y = self.location[1] + dy
        
        # Check bounds (assuming 10x10 grid)
        if not (0 <= new_x < 10 and 0 <= new_y < 10):
            return "Cannot move outside the village boundaries"
        
        self.location = (new_x, new_y)
        self.stats.update_energy(-5)
        
        # Store movement in memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="observation",
            content=f"Moved {direction} to location {self.location}",
            metadata={"location": str(self.location)}
        )
        
        return f"Moved {direction} to {self.location}"
    
    @tool
    async def observe(self) -> str:
        """Observe the current surroundings"""
        # This would interface with the world state
        observation = f"You are at location {self.location}. "
        observation += f"Energy: {self.stats.energy}/100, Health: {self.stats.health}/100"
        
        # Store observation
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="observation",
            content=observation,
            metadata={"location": str(self.location)}
        )
        
        return observation
    
    @tool
    async def remember(self, topic: str) -> str:
        """Recall memories about a topic"""
        memories = await self.memory_manager.retrieve_memories(
            agent_id=self.agent_id,
            query=topic,
            k=5
        )
        
        if not memories:
            return f"No memories about {topic}"
        
        memory_texts = [m.content for m in memories[:3]]
        return f"Memories about {topic}: " + "; ".join(memory_texts)
    
    @tool
    async def trade(
        self,
        target: str,
        offer_items: Dict[str, int],
        request_items: Dict[str, int]
    ) -> str:
        """Propose a trade with another agent"""
        if self.stats.energy < 5:
            return "Too tired to trade"
        
        # Check if we have the items to offer
        for item, quantity in offer_items.items():
            if not self.inventory.has_item(item, quantity):
                return f"Don't have enough {item} to trade"
        
        self.stats.update_energy(-5)
        
        # This would interface with the other agent
        trade_proposal = {
            "from": self.agent_id,
            "to": target,
            "offer": offer_items,
            "request": request_items
        }
        
        # Store trade attempt in memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="action",
            content=f"Proposed trade with {target}",
            metadata=trade_proposal
        )
        
        return f"Proposed trade to {target}"
    
    @tool
    async def work(self) -> str:
        """Perform work based on role"""
        if self.stats.energy < 20:
            return "Too tired to work"
        
        self.stats.update_energy(-20)
        
        # Role-specific work rewards
        work_rewards = {
            "farmer": {"gold": 5, "items": {"wheat": 3}},
            "merchant": {"gold": 10, "silver": 50},
            "artist": {"gold": 3, "items": {"painting": 1}},
            "guard": {"gold": 7, "reputation": 0.01},
            "sage": {"knowledge": 0.05, "items": {"scroll": 1}}
        }
        
        reward = work_rewards.get(self.role, {"gold": 5})
        
        # Apply rewards
        if "gold" in reward:
            self.stats.earn(gold=reward["gold"])
        if "silver" in reward:
            self.stats.earn(silver=reward.get("silver", 0))
        if "reputation" in reward:
            self.stats.reputation = min(1.0, self.stats.reputation + reward["reputation"])
        if "knowledge" in reward:
            self.stats.knowledge_level = min(1.0, self.stats.knowledge_level + reward["knowledge"])
        if "items" in reward:
            for item, qty in reward["items"].items():
                self.inventory.add_item(item, qty)
        
        work_description = f"Worked as a {self.role}"
        
        # Store work in memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="action",
            content=work_description,
            metadata={"reward": reward}
        )
        
        return work_description
    
    @tool
    async def rest(self) -> str:
        """Rest to recover energy"""
        energy_recovered = min(30, 100 - self.stats.energy)
        self.stats.update_energy(energy_recovered)
        
        # Reduce stress while resting
        self.emotions.stress = max(0, self.emotions.stress - 0.1)
        
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="action",
            content=f"Rested and recovered {energy_recovered} energy",
            metadata={"energy_recovered": energy_recovered}
        )
        
        return f"Rested and recovered {energy_recovered} energy"
    
    @tool
    async def reflect(self) -> str:
        """Reflect on recent experiences"""
        # Retrieve recent memories
        recent_memories = await self.memory_manager.retrieve_memories(
            agent_id=self.agent_id,
            query="recent experiences",
            k=10
        )
        
        if not recent_memories:
            return "Nothing significant to reflect on"
        
        # Generate reflection (would use LLM in full implementation)
        reflection = f"Reflected on {len(recent_memories)} recent experiences"
        
        # Store reflection as a high-level memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="reflection",
            content=reflection,
            metadata={"memory_count": len(recent_memories)}
        )
        
        # Update emotional baseline based on reflection
        positive_memories = sum(1 for m in recent_memories if "happy" in m.content.lower())
        if positive_memories > len(recent_memories) / 2:
            self.emotions.baseline_happiness = min(0.8, self.emotions.baseline_happiness + 0.05)
        
        return reflection
    
    async def perceive_environment(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive the current environment and nearby agents"""
        perception = {
            "location": self.location,
            "location_type": world_state.get("location_type", "path"),
            "nearby_agents": world_state.get("nearby_agents", []),
            "time_of_day": world_state.get("time_of_day", "day"),
            "weather": world_state.get("weather", "clear"),
            "available_actions": self._get_available_actions(world_state)
        }
        
        # Add internal state
        perception["internal_state"] = {
            "energy": self.stats.energy,
            "health": self.stats.health,
            "emotion": self.emotions.get_dominant_emotion(),
            "hunger": self.stats.hunger,
            "gold": self.stats.gold
        }
        
        return perception
    
    def _get_available_actions(self, world_state: Dict[str, Any]) -> List[str]:
        """Determine available actions based on current state"""
        actions = ["observe", "remember", "reflect"]
        
        if self.stats.energy >= 5:
            actions.append("move")
        
        if self.stats.energy >= 2 and world_state.get("nearby_agents"):
            actions.append("speak")
        
        if self.stats.energy >= 20:
            actions.append("work")
        
        if self.stats.energy < 50:
            actions.append("rest")
        
        if self.inventory.items and world_state.get("nearby_agents"):
            actions.append("trade")
        
        return actions
    
    def update_tick(self, tick: int, world_state: Dict[str, Any]):
        """Update agent state each tick"""
        # Natural energy regeneration
        if tick - self.last_action_tick > 5:
            self.stats.update_energy(1)
        
        # Hunger increases over time
        self.stats.hunger = min(1.0, self.stats.hunger + 0.01)
        
        # Apply emotional decay
        self.emotions.decay(settings.personality.emotion_decay_rates)
        
        # Update last action tick
        self.last_action_tick = tick
    
    def to_save_state(self) -> Dict[str, Any]:
        """Serialize agent state for saving"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "backstory": self.backstory,
            "archetype": self.archetype,
            "location": self.location,
            "personality": self.personality.__dict__,
            "emotions": self.emotions.__dict__,
            "stats": {
                "health": self.stats.health,
                "energy": self.stats.energy,
                "hunger": self.stats.hunger,
                "gold": self.stats.gold,
                "silver": self.stats.silver,
                "reputation": self.stats.reputation,
                "knowledge_level": self.stats.knowledge_level,
                "skills": self.stats.skills,
                "known_facts": self.stats.known_facts,
                "short_term_goals": self.stats.short_term_goals,
                "long_term_goals": self.stats.long_term_goals
            },
            "inventory": self.inventory.items,
            "relationships": {
                aid: {
                    "trust": rel.trust,
                    "friendship": rel.friendship,
                    "respect": rel.respect,
                    "interaction_count": rel.interaction_count
                }
                for aid, rel in self.social.relationships.items()
            }
        }
    
    @classmethod
    def from_save_state(
        cls,
        state: Dict[str, Any],
        memory_manager: MemoryManager
    ) -> 'SimulationAgent':
        """Restore agent from saved state"""
        personality = Personality(**state["personality"])
        
        agent = cls(
            agent_id=state["agent_id"],
            name=state["name"],
            role=state["role"],
            personality=personality,
            initial_location=tuple(state["location"]),
            memory_manager=memory_manager,
            backstory=state.get("backstory", ""),
            archetype=state.get("archetype")
        )
        
        # Restore state
        agent.emotions = EmotionalProfile(**state["emotions"])
        for key, value in state["stats"].items():
            if hasattr(agent.stats, key):
                setattr(agent.stats, key, value)
        
        agent.inventory.items = state["inventory"]
        
        # Restore relationships
        for agent_id, rel_data in state.get("relationships", {}).items():
            rel = Relationship(agent_id=agent_id)
            for key, value in rel_data.items():
                if hasattr(rel, key):
                    setattr(rel, key, value)
            agent.social.relationships[agent_id] = rel
        
        return agent