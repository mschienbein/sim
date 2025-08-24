"""
Core configuration and settings management for the simulation.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig(BaseSettings):
    """LLM Provider configuration"""
    provider: str = Field(default="openai", env="LLM_PROVIDER")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model_id: str = Field(default="gpt-5", env="OPENAI_MODEL_ID")
    openai_sage_model_id: str = Field(default="gpt-5", env="OPENAI_SAGE_MODEL_ID")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # AWS/Bedrock settings (backup)
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    bedrock_region: str = Field(default="us-east-1", env="BEDROCK_REGION")
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-haiku-20240307",
        env="BEDROCK_MODEL_ID"
    )
    bedrock_sage_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229", 
        env="BEDROCK_SAGE_MODEL_ID"
    )
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class Neo4jConfig(BaseSettings):
    """Neo4j database configuration"""
    uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    password: str = Field(default="password", env="NEO4J_PASSWORD")
    database: str = Field(default="simulation", env="NEO4J_DATABASE")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class SimulationConfig(BaseSettings):
    """Core simulation parameters"""
    name: str = Field(default="smallville_v1", env="SIMULATION_NAME")
    max_agents: int = Field(default=5, env="MAX_AGENTS")
    max_days: int = Field(default=10, env="MAX_DAYS")
    ticks_per_day: int = Field(default=24, env="TICKS_PER_DAY")
    tick_duration_ms: int = Field(default=5000, env="TICK_DURATION_MS")
    
    # World configuration
    world_size: tuple = (10, 10)
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class RateLimitConfig(BaseSettings):
    """Token budget and rate limiting"""
    daily_token_budget: int = Field(default=1000000, env="DAILY_TOKEN_BUDGET")
    per_agent_token_limit: int = Field(default=200000, env="PER_AGENT_TOKEN_LIMIT")
    max_conversation_turns: int = Field(default=5, env="MAX_CONVERSATION_TURNS")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class PersonalityConfig(BaseSettings):
    """Agent personality trait ranges"""
    trait_ranges: dict = Field(default={
        "openness": (0.0, 1.0),
        "conscientiousness": (0.0, 1.0),
        "extraversion": (0.0, 1.0),
        "agreeableness": (0.0, 1.0),
        "neuroticism": (0.0, 1.0)
    })
    
    emotion_ranges: dict = Field(default={
        "happiness": (0.0, 1.0),
        "anger": (0.0, 1.0),
        "fear": (0.0, 1.0),
        "sadness": (0.0, 1.0),
        "surprise": (0.0, 1.0),
        "energy": (0.0, 100.0),
        "stress": (0.0, 1.0)
    })
    
    # Decay rates per tick
    emotion_decay_rates: dict = Field(default={
        "happiness": 0.01,
        "anger": 0.02,
        "fear": 0.015,
        "sadness": 0.008,
        "surprise": 0.05,
        "stress": 0.012
    })
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class WorldConfig(BaseSettings):
    """World and location configuration"""
    
    locations: dict = Field(default={
        "forest": {
            "symbol": "F",
            "type": "natural",
            "capacity": 10,
            "resources": ["wood", "herbs", "mushrooms"],
            "buff": {"energy_regen": 1.2}
        },
        "houses": {
            "symbol": "H",
            "type": "residential",
            "capacity": 4,
            "resources": ["rest", "privacy"],
            "buff": {"stress_reduction": 1.5}
        },
        "market": {
            "symbol": "M",
            "type": "commercial",
            "capacity": 20,
            "resources": ["goods", "trade"],
            "economic_hub": True
        },
        "temple": {
            "symbol": "T",
            "type": "spiritual",
            "capacity": 15,
            "resources": ["wisdom", "peace"],
            "buff": {"stress_reduction": 2.0, "happiness": 1.3}
        },
        "library": {
            "symbol": "L",
            "type": "knowledge",
            "capacity": 8,
            "resources": ["books", "scrolls", "maps"],
            "buff": {"learning_rate": 2.0},
            "sage_home": True
        },
        "park": {
            "symbol": "P",
            "type": "recreation",
            "capacity": 30,
            "resources": ["relaxation", "social"],
            "buff": {"happiness": 1.2}
        },
        "square": {
            "symbol": "S",
            "type": "social",
            "capacity": 50,
            "resources": ["news", "gossip"],
            "meeting_point": True
        },
        "cafe": {
            "symbol": "C",
            "type": "social",
            "capacity": 12,
            "resources": ["food", "drink", "conversation"],
            "buff": {"social_bonus": 1.5}
        },
        "inn": {
            "symbol": "I",
            "type": "hospitality",
            "capacity": 10,
            "resources": ["rest", "food", "gossip"],
            "buff": {"energy_regen": 1.5}
        },
        "river": {
            "symbol": "R",
            "type": "natural",
            "capacity": 20,
            "resources": ["water", "fish"],
            "buff": {"stress_reduction": 1.3}
        },
        "blacksmith": {
            "symbol": "B",
            "type": "craft",
            "capacity": 6,
            "resources": ["tools", "weapons"],
            "economic_activity": True
        },
        "apothecary": {
            "symbol": "A",
            "type": "craft",
            "capacity": 5,
            "resources": ["potions", "remedies"],
            "buff": {"health_regen": 1.5}
        },
        "garden": {
            "symbol": "G",
            "type": "natural",
            "capacity": 8,
            "resources": ["flowers", "vegetables"],
            "buff": {"happiness": 1.1, "stress_reduction": 1.2}
        },
        "waterfront": {
            "symbol": "W",
            "type": "natural",
            "capacity": 25,
            "resources": ["fish", "trade"],
            "economic_activity": True
        },
        "docks": {
            "symbol": "D",
            "type": "commercial",
            "capacity": 15,
            "resources": ["goods", "news"],
            "economic_hub": True
        },
        "path": {
            "symbol": ".",
            "type": "transit",
            "capacity": 100,
            "resources": []
        }
    })
    
    # Time of day effects
    time_effects: dict = Field(default={
        "dawn": {"energy": 1.1, "happiness": 1.05},
        "morning": {"energy": 1.0, "productivity": 1.2},
        "noon": {"social": 1.3, "trade": 1.2},
        "afternoon": {"energy": 0.9, "social": 1.1},
        "dusk": {"stress": 0.9, "social": 1.4},
        "night": {"energy": 0.7, "rest": 1.5}
    })
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class Settings:
    """Main settings aggregator"""
    def __init__(self):
        self.llm = LLMConfig()
        self.neo4j = Neo4jConfig()
        self.simulation = SimulationConfig()
        self.rate_limit = RateLimitConfig()
        self.personality = PersonalityConfig()
        self.world = WorldConfig()
        
        # Add sage-specific config
        self.SAGE_SEARCH_LIMIT_PER_DAY = int(os.getenv("SAGE_SEARCH_LIMIT_PER_DAY", "1"))
        
        # Backward compatibility
        self.aws = self.llm  # For backward compatibility
        
        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        self.checkpoints_dir = self.project_root / "checkpoints"
        
        # Create directories if they don't exist
        for dir_path in [self.logs_dir, self.data_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()