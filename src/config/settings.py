"""
Core configuration and settings management for the simulation.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig(BaseSettings):
    """LLM Provider configuration"""
    # Pydantic v2 Settings config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model_id: str = Field(default="gpt-5-nano", validation_alias="OPENAI_MODEL_ID")
    openai_sage_model_id: str = Field(default="gpt-5-nano", validation_alias="OPENAI_SAGE_MODEL_ID")
    openai_embedding_model: str = Field(default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL")
    # Retry/timeout
    openai_max_retries: int = Field(default=5, validation_alias="OPENAI_MAX_RETRIES")
    openai_timeout_seconds: float = Field(default=60.0, validation_alias="OPENAI_TIMEOUT_SECONDS")
    
    # AWS/Bedrock settings (backup)
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, validation_alias="AWS_SECRET_ACCESS_KEY")
    bedrock_region: str = Field(default="us-east-1", validation_alias="BEDROCK_REGION")
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-haiku-20240307",
        validation_alias="BEDROCK_MODEL_ID"
    )
    bedrock_sage_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229", 
        validation_alias="BEDROCK_SAGE_MODEL_ID"
    )

class Neo4jConfig(BaseSettings):
    """Neo4j database configuration"""
    # Use Pydantic v2-style settings config with an env prefix
    # so NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE are respected
    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=".env", extra="ignore")

    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="simulation")

class SimulationConfig(BaseSettings):
    """Core simulation parameters"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    name: str = Field(default="smallville_v1", validation_alias="SIMULATION_NAME")
    max_agents: int = Field(default=5, validation_alias="MAX_AGENTS")
    max_days: int = Field(default=10, validation_alias="MAX_DAYS")
    ticks_per_day: int = Field(default=24, validation_alias="TICKS_PER_DAY")
    tick_duration_ms: int = Field(default=5000, validation_alias="TICK_DURATION_MS")
    
    # World configuration
    world_size: tuple = (10, 10)

class RateLimitConfig(BaseSettings):
    """Token budget and rate limiting"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    daily_token_budget: int = Field(default=1000000, validation_alias="DAILY_TOKEN_BUDGET")
    per_agent_token_limit: int = Field(default=200000, validation_alias="PER_AGENT_TOKEN_LIMIT")
    max_conversation_turns: int = Field(default=5, validation_alias="MAX_CONVERSATION_TURNS")

class PersonalityConfig(BaseSettings):
    """Agent personality trait ranges"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
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
    

class WorldConfig(BaseSettings):
    """World and location configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
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