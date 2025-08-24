"""
Tools available to agents for interacting with the world.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class Tool:
    """Base tool definition"""
    name: str
    description: str
    energy_cost: int = 0
    cooldown: int = 0  # seconds
    requires_target: bool = False
    requires_location: Optional[str] = None

class ToolType(Enum):
    """Types of tools available"""
    MOVEMENT = "movement"
    SOCIAL = "social"
    ECONOMIC = "economic"
    KNOWLEDGE = "knowledge"
    CRAFT = "craft"
    SPECIAL = "special"

# Define all available tools
AGENT_TOOLS = {
    # Movement tools
    "move": Tool(
        name="move",
        description="Move in a cardinal direction",
        energy_cost=5,
        cooldown=0
    ),
    "plan_route": Tool(
        name="plan_route",
        description="Plan optimal route to destination",
        energy_cost=1,
        cooldown=60
    ),
    "explore": Tool(
        name="explore",
        description="Explore new areas randomly",
        energy_cost=10,
        cooldown=300
    ),
    
    # Social tools
    "speak": Tool(
        name="speak",
        description="Initiate conversation with another agent",
        energy_cost=2,
        cooldown=60,
        requires_target=True
    ),
    "gossip": Tool(
        name="gossip",
        description="Share rumors and news",
        energy_cost=3,
        cooldown=120,
        requires_target=True
    ),
    "teach": Tool(
        name="teach",
        description="Teach knowledge or skill to another",
        energy_cost=10,
        cooldown=600,
        requires_target=True
    ),
    "learn": Tool(
        name="learn",
        description="Learn from another agent",
        energy_cost=8,
        cooldown=600,
        requires_target=True
    ),
    "befriend": Tool(
        name="befriend",
        description="Attempt to build friendship",
        energy_cost=5,
        cooldown=300,
        requires_target=True
    ),
    
    # Economic tools
    "trade": Tool(
        name="trade",
        description="Propose trade with another agent",
        energy_cost=5,
        cooldown=300,
        requires_target=True
    ),
    "work": Tool(
        name="work",
        description="Perform role-specific work for income",
        energy_cost=20,
        cooldown=1800
    ),
    "craft": Tool(
        name="craft",
        description="Create items from materials",
        energy_cost=15,
        cooldown=900,
        requires_location="blacksmith"
    ),
    "harvest": Tool(
        name="harvest",
        description="Gather resources from environment",
        energy_cost=15,
        cooldown=600,
        requires_location="forest"
    ),
    "shop": Tool(
        name="shop",
        description="Buy items from market",
        energy_cost=3,
        cooldown=300,
        requires_location="market"
    ),
    
    # Knowledge tools
    "read": Tool(
        name="read",
        description="Read books or scrolls",
        energy_cost=10,
        cooldown=600,
        requires_location="library"
    ),
    "write_note": Tool(
        name="write_note",
        description="Write a note or letter",
        energy_cost=15,
        cooldown=900
    ),
    "post_notice": Tool(
        name="post_notice",
        description="Post public notice on board",
        energy_cost=5,
        cooldown=1800,
        requires_location="square"
    ),
    "research": Tool(
        name="research",
        description="Research a specific topic",
        energy_cost=20,
        cooldown=3600,
        requires_location="library"
    ),
    
    # Personal tools
    "rest": Tool(
        name="rest",
        description="Rest to recover energy",
        energy_cost=0,
        cooldown=300
    ),
    "eat": Tool(
        name="eat",
        description="Eat food to reduce hunger",
        energy_cost=0,
        cooldown=600
    ),
    "reflect": Tool(
        name="reflect",
        description="Reflect on experiences and memories",
        energy_cost=5,
        cooldown=3600
    ),
    "meditate": Tool(
        name="meditate",
        description="Meditate to reduce stress",
        energy_cost=0,
        cooldown=1800,
        requires_location="temple"
    ),
    "observe": Tool(
        name="observe",
        description="Observe surroundings carefully",
        energy_cost=1,
        cooldown=60
    ),
    "remember": Tool(
        name="remember",
        description="Recall specific memories",
        energy_cost=2,
        cooldown=120
    )
}

# Sage-exclusive tools
SAGE_TOOLS = {
    "web_search": Tool(
        name="web_search",
        description="Divine knowledge from the cosmic library",
        energy_cost=20,
        cooldown=86400  # Once per day
    ),
    "verify_knowledge": Tool(
        name="verify_knowledge",
        description="Verify if information is true",
        energy_cost=10,
        cooldown=3600
    ),
    "write_book": Tool(
        name="write_book",
        description="Author a book on accumulated knowledge",
        energy_cost=50,
        cooldown=86400,
        requires_location="library"
    ),
    "prophecy": Tool(
        name="prophecy",
        description="Make predictions about future events",
        energy_cost=30,
        cooldown=43200  # Twice per day
    )
}

class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools = AGENT_TOOLS.copy()
        self.sage_tools = SAGE_TOOLS.copy()
        self.custom_tools = {}
    
    def get_tool(self, name: str, is_sage: bool = False) -> Optional[Tool]:
        """Get a tool by name"""
        if name in self.tools:
            return self.tools[name]
        if is_sage and name in self.sage_tools:
            return self.sage_tools[name]
        if name in self.custom_tools:
            return self.custom_tools[name]
        return None
    
    def get_available_tools(
        self,
        agent_role: str,
        location: Optional[str] = None,
        energy: float = 100,
        is_sage: bool = False
    ) -> List[Tool]:
        """Get tools available to an agent based on context"""
        available = []
        
        # Check regular tools
        for tool in self.tools.values():
            # Check energy requirement
            if tool.energy_cost > energy:
                continue
            
            # Check location requirement
            if tool.requires_location and tool.requires_location != location:
                continue
            
            available.append(tool)
        
        # Add sage tools if applicable
        if is_sage:
            for tool in self.sage_tools.values():
                if tool.energy_cost <= energy:
                    if not tool.requires_location or tool.requires_location == location:
                        available.append(tool)
        
        return available
    
    def register_custom_tool(self, tool: Tool):
        """Register a custom tool"""
        self.custom_tools[tool.name] = tool
    
    def get_tools_by_type(self, tool_type: ToolType) -> List[Tool]:
        """Get all tools of a specific type"""
        tools = []
        
        # Categorize tools by type
        type_mapping = {
            ToolType.MOVEMENT: ["move", "plan_route", "explore"],
            ToolType.SOCIAL: ["speak", "gossip", "teach", "learn", "befriend"],
            ToolType.ECONOMIC: ["trade", "work", "craft", "harvest", "shop"],
            ToolType.KNOWLEDGE: ["read", "write_note", "post_notice", "research"],
            ToolType.CRAFT: ["craft", "harvest"],
            ToolType.SPECIAL: ["web_search", "verify_knowledge", "write_book", "prophecy"]
        }
        
        tool_names = type_mapping.get(tool_type, [])
        for name in tool_names:
            tool = self.get_tool(name, is_sage=True)
            if tool:
                tools.append(tool)
        
        return tools

# Global tool registry
tool_registry = ToolRegistry()