"""
Agent personality system based on Big Five model with dynamic emotional states.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random
import numpy as np
from enum import Enum

class PersonalityTrait(Enum):
    """Big Five personality traits"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"

class EmotionalState(Enum):
    """Dynamic emotional states"""
    HAPPINESS = "happiness"
    ANGER = "anger"
    FEAR = "fear"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    ENERGY = "energy"
    STRESS = "stress"

@dataclass
class Personality:
    """Agent personality profile with Big Five traits"""
    
    # Big Five traits (0.0 to 1.0)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    # Additional personality aspects
    curiosity: float = 0.5
    creativity: float = 0.5
    ambition: float = 0.5
    patience: float = 0.5
    humor: float = 0.5
    
    def __post_init__(self):
        """Validate trait ranges"""
        for trait_name, trait_value in self.__dict__.items():
            if not 0.0 <= trait_value <= 1.0:
                raise ValueError(f"{trait_name} must be between 0.0 and 1.0")
    
    def to_prompt_description(self) -> str:
        """Convert personality to natural language for LLM prompt"""
        descriptions = []
        
        # Openness
        if self.openness > 0.7:
            descriptions.append("very curious and imaginative")
        elif self.openness < 0.3:
            descriptions.append("practical and conventional")
        
        # Conscientiousness
        if self.conscientiousness > 0.7:
            descriptions.append("organized and disciplined")
        elif self.conscientiousness < 0.3:
            descriptions.append("spontaneous and flexible")
        
        # Extraversion
        if self.extraversion > 0.7:
            descriptions.append("outgoing and energetic")
        elif self.extraversion < 0.3:
            descriptions.append("reserved and introspective")
        
        # Agreeableness
        if self.agreeableness > 0.7:
            descriptions.append("friendly and compassionate")
        elif self.agreeableness < 0.3:
            descriptions.append("competitive and skeptical")
        
        # Neuroticism
        if self.neuroticism > 0.7:
            descriptions.append("sensitive and emotionally reactive")
        elif self.neuroticism < 0.3:
            descriptions.append("calm and emotionally stable")
        
        return ", ".join(descriptions) if descriptions else "balanced personality"
    
    def interaction_affinity(self, other: 'Personality') -> float:
        """Calculate natural affinity between two personalities"""
        # Similar personalities tend to get along
        similarity = 1.0 - np.mean([
            abs(self.openness - other.openness),
            abs(self.conscientiousness - other.conscientiousness),
            abs(self.extraversion - other.extraversion),
            abs(self.agreeableness - other.agreeableness),
            abs(self.neuroticism - other.neuroticism)
        ])
        
        # Complementary traits can also work
        complement_bonus = 0.0
        if (self.extraversion > 0.7 and other.extraversion < 0.3) or \
           (self.extraversion < 0.3 and other.extraversion > 0.7):
            complement_bonus += 0.1
        
        if (self.conscientiousness > 0.7 and other.creativity > 0.7):
            complement_bonus += 0.1
        
        return min(1.0, similarity + complement_bonus)

@dataclass
class EmotionalProfile:
    """Dynamic emotional state that changes over time"""
    
    happiness: float = 0.5
    anger: float = 0.0
    fear: float = 0.0
    sadness: float = 0.0
    surprise: float = 0.0
    energy: float = 1.0
    stress: float = 0.0
    
    # Emotional memory
    recent_triggers: List[Tuple[str, float]] = field(default_factory=list)
    baseline_happiness: float = 0.5
    
    def __post_init__(self):
        """Validate emotional ranges"""
        for emotion_name, emotion_value in self.__dict__.items():
            if emotion_name in ["recent_triggers", "baseline_happiness"]:
                continue
            if emotion_name == "energy":
                if not 0.0 <= emotion_value <= 1.0:
                    self.energy = max(0.0, min(1.0, emotion_value))
            else:
                if not 0.0 <= emotion_value <= 1.0:
                    setattr(self, emotion_name, max(0.0, min(1.0, emotion_value)))
    
    def update(self, event: str, impact: Dict[str, float]):
        """Update emotional state based on an event"""
        for emotion, change in impact.items():
            if hasattr(self, emotion):
                current = getattr(self, emotion)
                new_value = max(0.0, min(1.0, current + change))
                setattr(self, emotion, new_value)
        
        # Record trigger
        self.recent_triggers.append((event, sum(abs(v) for v in impact.values())))
        if len(self.recent_triggers) > 10:
            self.recent_triggers.pop(0)
    
    def decay(self, decay_rates: Dict[str, float]):
        """Apply natural emotional decay over time"""
        for emotion, rate in decay_rates.items():
            if hasattr(self, emotion) and emotion != "energy":
                current = getattr(self, emotion)
                # Emotions decay toward baseline
                if emotion == "happiness":
                    target = self.baseline_happiness
                else:
                    target = 0.0
                
                new_value = current + (target - current) * rate
                setattr(self, emotion, new_value)
        
        # Energy regenerates slowly
        if self.energy < 1.0:
            self.energy = min(1.0, self.energy + 0.01)
    
    def get_dominant_emotion(self) -> str:
        """Return the strongest current emotion"""
        emotions = {
            "happy": self.happiness,
            "angry": self.anger,
            "fearful": self.fear,
            "sad": self.sadness,
            "surprised": self.surprise,
            "stressed": self.stress
        }
        
        # Filter out near-zero emotions
        active_emotions = {k: v for k, v in emotions.items() if v > 0.1}
        
        if not active_emotions:
            return "neutral"
        
        return max(active_emotions, key=active_emotions.get)
    
    def to_prompt_description(self) -> str:
        """Convert emotional state to natural language"""
        dominant = self.get_dominant_emotion()
        intensity = getattr(self, dominant.replace("ful", "").replace("ped", "e").replace("sed", "s"), 0.5)
        
        if intensity > 0.7:
            prefix = "very"
        elif intensity > 0.4:
            prefix = "somewhat"
        else:
            prefix = "slightly"
        
        energy_desc = ""
        if self.energy < 0.3:
            energy_desc = " and exhausted"
        elif self.energy < 0.5:
            energy_desc = " and tired"
        elif self.energy > 0.8:
            energy_desc = " and energetic"
        
        return f"{prefix} {dominant}{energy_desc}"

@dataclass
class SocialProfile:
    """Track relationships and social dynamics"""
    
    relationships: Dict[str, 'Relationship'] = field(default_factory=dict)
    reputation: float = 0.5
    social_energy: float = 1.0
    last_interaction_tick: int = 0
    
    def get_relationship(self, agent_id: str) -> Optional['Relationship']:
        """Get relationship with another agent"""
        return self.relationships.get(agent_id)
    
    def update_relationship(self, agent_id: str, changes: Dict[str, float]):
        """Update relationship metrics"""
        if agent_id not in self.relationships:
            self.relationships[agent_id] = Relationship(agent_id=agent_id)
        
        rel = self.relationships[agent_id]
        for metric, change in changes.items():
            if hasattr(rel, metric):
                current = getattr(rel, metric)
                new_value = max(-1.0, min(1.0, current + change))
                setattr(rel, metric, new_value)
        
        rel.interaction_count += 1
        rel.last_interaction = self.last_interaction_tick

@dataclass
class Relationship:
    """Relationship between two agents"""
    
    agent_id: str
    trust: float = 0.0
    friendship: float = 0.0
    respect: float = 0.0
    familiarity: float = 0.0
    attraction: float = 0.0
    hostility: float = 0.0
    
    interaction_count: int = 0
    last_interaction: int = 0
    shared_memories: List[str] = field(default_factory=list)
    
    def get_overall_sentiment(self) -> float:
        """Calculate overall relationship quality (-1 to 1)"""
        positive = (self.trust + self.friendship + self.respect + 
                   self.familiarity + self.attraction) / 5
        negative = self.hostility
        return positive - negative
    
    def to_prompt_description(self) -> str:
        """Convert relationship to natural language"""
        sentiment = self.get_overall_sentiment()
        
        if sentiment > 0.6:
            base = "close friend"
        elif sentiment > 0.3:
            base = "friendly acquaintance"
        elif sentiment > -0.3:
            base = "neutral acquaintance"
        elif sentiment > -0.6:
            base = "uncomfortable acquaintance"
        else:
            base = "adversary"
        
        modifiers = []
        if self.trust > 0.7:
            modifiers.append("trusted")
        if self.respect > 0.7:
            modifiers.append("respected")
        if self.hostility > 0.5:
            modifiers.append("tense")
        
        if modifiers:
            return f"{', '.join(modifiers)} {base}"
        return base

class PersonalityGenerator:
    """Generate random but coherent personalities"""
    
    @staticmethod
    def generate_random() -> Personality:
        """Generate a random personality with some coherence"""
        # Start with random base
        traits = {
            "openness": random.gauss(0.5, 0.2),
            "conscientiousness": random.gauss(0.5, 0.2),
            "extraversion": random.gauss(0.5, 0.2),
            "agreeableness": random.gauss(0.5, 0.2),
            "neuroticism": random.gauss(0.5, 0.2),
        }
        
        # Clamp to valid range
        for key in traits:
            traits[key] = max(0.0, min(1.0, traits[key]))
        
        # Derive related traits
        traits["curiosity"] = max(0.0, min(1.0, traits["openness"] + random.gauss(0, 0.1)))
        traits["creativity"] = max(0.0, min(1.0, traits["openness"] + random.gauss(0, 0.1)))
        traits["ambition"] = max(0.0, min(1.0, traits["conscientiousness"] + random.gauss(0, 0.1)))
        traits["patience"] = max(0.0, min(1.0, 1.0 - traits["neuroticism"] + random.gauss(0, 0.1)))
        traits["humor"] = max(0.0, min(1.0, traits["extraversion"] + random.gauss(0, 0.1)))
        
        return Personality(**traits)
    
    @staticmethod
    def generate_archetype(archetype: str) -> Personality:
        """Generate personality based on archetype"""
        archetypes = {
            "sage": {
                "openness": 0.9, "conscientiousness": 0.8,
                "extraversion": 0.3, "agreeableness": 0.7,
                "neuroticism": 0.2, "curiosity": 0.95,
                "creativity": 0.7, "ambition": 0.6,
                "patience": 0.9, "humor": 0.4
            },
            "farmer": {
                "openness": 0.3, "conscientiousness": 0.8,
                "extraversion": 0.4, "agreeableness": 0.7,
                "neuroticism": 0.3, "curiosity": 0.3,
                "creativity": 0.2, "ambition": 0.5,
                "patience": 0.8, "humor": 0.5
            },
            "merchant": {
                "openness": 0.6, "conscientiousness": 0.7,
                "extraversion": 0.8, "agreeableness": 0.6,
                "neuroticism": 0.4, "curiosity": 0.6,
                "creativity": 0.5, "ambition": 0.8,
                "patience": 0.5, "humor": 0.6
            },
            "artist": {
                "openness": 0.95, "conscientiousness": 0.3,
                "extraversion": 0.6, "agreeableness": 0.6,
                "neuroticism": 0.7, "curiosity": 0.9,
                "creativity": 0.95, "ambition": 0.6,
                "patience": 0.4, "humor": 0.7
            },
            "guard": {
                "openness": 0.3, "conscientiousness": 0.9,
                "extraversion": 0.5, "agreeableness": 0.4,
                "neuroticism": 0.3, "curiosity": 0.3,
                "creativity": 0.2, "ambition": 0.6,
                "patience": 0.7, "humor": 0.3
            }
        }
        
        if archetype not in archetypes:
            return PersonalityGenerator.generate_random()
        
        # Add small random variations
        traits = archetypes[archetype].copy()
        for key in traits:
            traits[key] = max(0.0, min(1.0, traits[key] + random.gauss(0, 0.05)))
        
        return Personality(**traits)