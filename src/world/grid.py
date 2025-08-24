"""
Grid-based world system with locations and spatial awareness.
"""

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np

from src.config.settings import settings

class LocationType(Enum):
    """Types of locations in the world"""
    FOREST = "forest"
    HOUSES = "houses"
    MARKET = "market"
    TEMPLE = "temple"
    LIBRARY = "library"
    PARK = "park"
    SQUARE = "square"
    CAFE = "cafe"
    INN = "inn"
    RIVER = "river"
    BLACKSMITH = "blacksmith"
    APOTHECARY = "apothecary"
    GARDEN = "garden"
    WATERFRONT = "waterfront"
    DOCKS = "docks"
    PATH = "path"

@dataclass
class Location:
    """Individual location in the world"""
    x: int
    y: int
    location_type: LocationType
    name: str
    capacity: int = 100
    current_occupants: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    buffs: Dict[str, float] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    @property
    def is_full(self) -> bool:
        return len(self.current_occupants) >= self.capacity
    
    def add_occupant(self, agent_id: str) -> bool:
        """Add an agent to this location"""
        if self.is_full:
            return False
        if agent_id not in self.current_occupants:
            self.current_occupants.append(agent_id)
        return True
    
    def remove_occupant(self, agent_id: str):
        """Remove an agent from this location"""
        if agent_id in self.current_occupants:
            self.current_occupants.remove(agent_id)
    
    def get_description(self) -> str:
        """Get a text description of the location"""
        desc = f"{self.name} ({self.location_type.value})"
        if self.current_occupants:
            desc += f" with {len(self.current_occupants)} people"
        if self.resources:
            desc += f", offering {', '.join(self.resources)}"
        return desc

class WorldGrid:
    """The world grid containing all locations"""
    
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.grid: Dict[Tuple[int, int], Location] = {}
        self.location_types: Dict[LocationType, List[Location]] = {}
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        
        # Create numpy arrays for efficient spatial operations
        self.location_grid = np.zeros((height, width), dtype=int)  # Location type indices
        self.occupancy_grid = np.zeros((height, width), dtype=int)  # Number of occupants
        self.walkability_grid = np.ones((height, width), dtype=bool)  # Can walk here
        
        # Initialize the world
        self._generate_world()
    
    def _generate_world(self):
        """Generate the world layout"""
        # Define the world layout (10x10 grid)
        layout = [
            "FFFHHMMTTL",
            "F..HHMMTTL",
            "..PPSSCC.L",
            "..PPSSCC..",
            "RR..II..GG",
            "RR..II..GG",
            "..BB..AA..",
            "..BB..AA..",
            "WWW.....DD",
            "WWW.....DD"
        ]
        
        # Mapping of symbols to location types
        symbol_map = {
            'F': LocationType.FOREST,
            'H': LocationType.HOUSES,
            'M': LocationType.MARKET,
            'T': LocationType.TEMPLE,
            'L': LocationType.LIBRARY,
            'P': LocationType.PARK,
            'S': LocationType.SQUARE,
            'C': LocationType.CAFE,
            'I': LocationType.INN,
            'R': LocationType.RIVER,
            'B': LocationType.BLACKSMITH,
            'A': LocationType.APOTHECARY,
            'G': LocationType.GARDEN,
            'W': LocationType.WATERFRONT,
            'D': LocationType.DOCKS,
            '.': LocationType.PATH
        }
        
        # Create locations based on layout
        for y, row in enumerate(layout):
            for x, symbol in enumerate(row):
                loc_type = symbol_map[symbol]
                loc_config = settings.world.locations.get(loc_type.value, {})
                
                # Generate location name
                if loc_type == LocationType.PATH:
                    name = f"Path ({x},{y})"
                else:
                    name = f"The {loc_type.value.title()}"
                
                location = Location(
                    x=x,
                    y=y,
                    location_type=loc_type,
                    name=name,
                    capacity=loc_config.get("capacity", 100),
                    resources=loc_config.get("resources", []),
                    buffs=loc_config.get("buff", {}),
                    properties={
                        "economic_hub": loc_config.get("economic_hub", False),
                        "economic_activity": loc_config.get("economic_activity", False),
                        "meeting_point": loc_config.get("meeting_point", False),
                        "sage_home": loc_config.get("sage_home", False)
                    }
                )
                
                self.grid[(x, y)] = location
                
                # Update numpy grids
                self.location_grid[y, x] = list(LocationType).index(loc_type)
                
                # Mark unwalkable locations if needed
                if loc_type in [LocationType.RIVER]:  # Rivers are harder to cross
                    self.walkability_grid[y, x] = False
                
                # Track by type
                if loc_type not in self.location_types:
                    self.location_types[loc_type] = []
                self.location_types[loc_type].append(location)
    
    def get_location(self, x: int, y: int) -> Optional[Location]:
        """Get location at specific coordinates"""
        return self.grid.get((x, y))
    
    def get_location_by_position(self, position: Tuple[int, int]) -> Optional[Location]:
        """Get location by position tuple"""
        return self.grid.get(position)
    
    def get_locations_by_type(self, location_type: LocationType) -> List[Location]:
        """Get all locations of a specific type"""
        return self.location_types.get(location_type, [])
    
    def get_nearby_locations(
        self,
        position: Tuple[int, int],
        radius: int = 1
    ) -> List[Location]:
        """Get locations within a radius"""
        x, y = position
        nearby = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    loc = self.get_location(nx, ny)
                    if loc:
                        nearby.append(loc)
        
        return nearby
    
    def get_nearby_agents(
        self,
        position: Tuple[int, int],
        radius: int = 1,
        exclude_self: str = None
    ) -> List[str]:
        """Get agents within a radius"""
        nearby_agents = []
        nearby_locations = self.get_nearby_locations(position, radius)
        
        # Also check current location
        current = self.get_location_by_position(position)
        if current:
            nearby_locations.append(current)
        
        for loc in nearby_locations:
            for agent_id in loc.current_occupants:
                if agent_id != exclude_self:
                    nearby_agents.append(agent_id)
        
        return nearby_agents
    
    def move_agent(
        self,
        agent_id: str,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int]
    ) -> bool:
        """Move an agent from one position to another"""
        # Validate positions
        if not self.is_valid_position(to_pos):
            return False
        
        # Check walkability using numpy grid
        if not self.walkability_grid[to_pos[1], to_pos[0]]:
            return False
        
        # Get locations
        from_loc = self.get_location_by_position(from_pos)
        to_loc = self.get_location_by_position(to_pos)
        
        if not to_loc:
            return False
        
        # Check capacity
        if to_loc.is_full:
            return False
        
        # Move agent
        if from_loc:
            from_loc.remove_occupant(agent_id)
            # Update numpy occupancy grid
            self.occupancy_grid[from_pos[1], from_pos[0]] -= 1
            
        to_loc.add_occupant(agent_id)
        # Update numpy occupancy grid
        self.occupancy_grid[to_pos[1], to_pos[0]] += 1
        
        # Update tracking
        self.agent_positions[agent_id] = to_pos
        
        return True
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if a position is valid"""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Find path between two positions using A* algorithm with numpy optimization"""
        if not self.is_valid_position(start) or not self.is_valid_position(end):
            return []
        
        # Use numpy for efficient distance calculation
        from heapq import heappush, heappop
        
        def heuristic(a, b):
            # Use numpy for vectorized distance calculation
            return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_position(neighbor):
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def get_direction_to(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int]
    ) -> Optional[str]:
        """Get the direction to move from one position toward another"""
        path = self.get_path(from_pos, to_pos)
        if not path:
            return None
        
        next_pos = path[0] if path else to_pos
        dx = next_pos[0] - from_pos[0]
        dy = next_pos[1] - from_pos[1]
        
        if dx > 0:
            return "east"
        elif dx < 0:
            return "west"
        elif dy > 0:
            return "south"
        elif dy < 0:
            return "north"
        
        return None
    
    def find_nearest_location_type(
        self,
        position: Tuple[int, int],
        location_type: LocationType
    ) -> Optional[Location]:
        """Find the nearest location of a specific type using numpy"""
        locations = self.get_locations_by_type(location_type)
        if not locations:
            return None
        
        # Use numpy for vectorized distance calculation
        positions = np.array([(loc.x, loc.y) for loc in locations])
        pos_array = np.array(position)
        
        # Calculate Manhattan distances using numpy
        distances = np.sum(np.abs(positions - pos_array), axis=1)
        
        # Find minimum distance index
        min_idx = np.argmin(distances)
        
        return locations[min_idx]
    
    def get_world_state(self, current_tick: int) -> Dict[str, Any]:
        """Get the current state of the world"""
        # Determine time of day
        hour = current_tick % 24
        if 5 <= hour < 8:
            time_of_day = "dawn"
        elif 8 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 14:
            time_of_day = "noon"
        elif 14 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 21:
            time_of_day = "dusk"
        else:
            time_of_day = "night"
        
        # Simple weather simulation
        weather_options = ["clear", "cloudy", "rainy", "foggy"]
        weather = random.choice(weather_options) if random.random() < 0.1 else "clear"
        
        return {
            "tick": current_tick,
            "time_of_day": time_of_day,
            "hour": hour,
            "weather": weather,
            "total_agents": len(self.agent_positions)
        }
    
    def get_location_info(
        self,
        position: Tuple[int, int],
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a location"""
        loc = self.get_location_by_position(position)
        if not loc:
            return {}
        
        info = {
            "position": position,
            "name": loc.name,
            "type": loc.location_type.value,
            "capacity": loc.capacity,
            "occupancy": len(loc.current_occupants),
            "resources": loc.resources,
            "buffs": loc.buffs,
            "properties": loc.properties
        }
        
        # Add nearby agents if requested
        if agent_id:
            info["nearby_agents"] = self.get_nearby_agents(position, radius=1, exclude_self=agent_id)
            info["agents_here"] = [a for a in loc.current_occupants if a != agent_id]
        
        return info
    
    def get_distance_matrix(self, position: Tuple[int, int]) -> np.ndarray:
        """Get a numpy array of distances from a position to all grid cells"""
        y, x = position
        
        # Create coordinate grids
        Y, X = np.mgrid[0:self.height, 0:self.width]
        
        # Calculate Manhattan distances
        distances = np.abs(X - x) + np.abs(Y - y)
        
        return distances
    
    def get_reachable_area(self, position: Tuple[int, int], max_distance: int) -> List[Tuple[int, int]]:
        """Get all positions reachable within max_distance steps using numpy"""
        distance_matrix = self.get_distance_matrix(position)
        
        # Find positions within max_distance
        reachable = np.where(distance_matrix <= max_distance)
        
        # Convert to list of tuples
        positions = []
        for y, x in zip(reachable[0], reachable[1]):
            if self.walkability_grid[y, x]:  # Only include walkable positions
                positions.append((x, y))
        
        return positions
    
    def get_population_density(self) -> np.ndarray:
        """Get population density heatmap as numpy array"""
        density = np.zeros((self.height, self.width))
        
        for (x, y), location in self.grid.items():
            density[y, x] = len(location.current_occupants)
        
        return density
    
    def find_least_crowded_location(self, location_type: Optional[LocationType] = None) -> Optional[Location]:
        """Find the least crowded location using numpy"""
        if location_type:
            locations = self.get_locations_by_type(location_type)
        else:
            locations = list(self.grid.values())
        
        if not locations:
            return None
        
        # Use numpy to find minimum occupancy
        occupancies = np.array([len(loc.current_occupants) for loc in locations])
        min_idx = np.argmin(occupancies)
        
        return locations[min_idx]
    
    def visualize_grid(self, show_agents: bool = True) -> str:
        """Create ASCII visualization of the world"""
        lines = []
        lines.append("  " + " ".join(str(x) for x in range(self.width)))
        
        for y in range(self.height):
            row = f"{y} "
            for x in range(self.width):
                loc = self.get_location(x, y)
                if loc:
                    # Show location symbol
                    symbol = settings.world.locations[loc.location_type.value]["symbol"]
                    
                    # Show agent count if any
                    if show_agents and loc.current_occupants:
                        count = len(loc.current_occupants)
                        if count > 9:
                            symbol = "*"
                        else:
                            symbol = str(count)
                    
                    row += symbol + " "
                else:
                    row += ". "
            
            lines.append(row)
        
        # Add legend
        lines.append("\nLegend:")
        for loc_type, config in settings.world.locations.items():
            lines.append(f"  {config['symbol']} = {loc_type}")
        
        return "\n".join(lines)