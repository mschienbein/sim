"""
Sage agent with special web search capabilities.
"""

from typing import Dict, List, Any
from datetime import datetime, date
import logging

from strands import tool
from strands.models.openai import OpenAIModel

from src.agents.base_agent import SimulationAgent
from src.memory.manager import MemoryManager
from src.agents.personality import Personality
from src.config.settings import settings

class SageAgent(SimulationAgent):
    """Special agent with web search capabilities"""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        initial_location: tuple,
        memory_manager: MemoryManager,
        search_limit_per_day: int = 1
    ):
        # Sage has specific personality traits
        sage_personality = Personality(
            openness=0.9,
            conscientiousness=0.8,
            extraversion=0.3,
            agreeableness=0.7,
            neuroticism=0.2,
            curiosity=0.95,
            creativity=0.7,
            ambition=0.6,
            patience=0.9,
            humor=0.4
        )
        
        backstory = """You are the village sage and keeper of knowledge. You have lived in the 
        library for decades, studying ancient texts and seeking wisdom. You possess a unique 
        ability to access external knowledge through mystical means (web search), but this power 
        is limited and must be used wisely. You share knowledge freely with those who seek it, 
        believing that wisdom should flow like water through the village."""
        
        # Create a better model for the sage
        sage_model = OpenAIModel(
            client_args={
                "api_key": settings.llm.openai_api_key,
                "max_retries": settings.llm.openai_max_retries,
                "timeout": settings.llm.openai_timeout_seconds,
            },
            model_id=settings.llm.openai_sage_model_id,
            params={
                "max_tokens": 1000,
                "temperature": 0.5,  # More focused for sage
            }
        )
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="sage",
            personality=sage_personality,
            initial_location=initial_location,
            memory_manager=memory_manager,
            backstory=backstory,
            archetype="sage",
            model=sage_model
        )
        
        # Web search tracking
        self.search_limit_per_day = search_limit_per_day
        self.searches_today = 0
        self.last_search_day = None
        self.search_history = []
        
        # Knowledge specialization
        self.stats.knowledge_level = 0.8  # Sage starts with high knowledge
        self.stats.skills = {
            "research": 0.9,
            "teaching": 0.8,
            "philosophy": 0.85,
            "history": 0.9,
            "mysticism": 0.7
        }
        # Log client configuration for observability (avoid sensitive fields)
        logging.getLogger(__name__).debug(
            "OpenAI client configured for SageAgent",
            extra={
                "agent_id": agent_id,
                "role": "sage",
                "model_id": settings.llm.openai_sage_model_id,
                "openai_max_retries": settings.llm.openai_max_retries,
                "openai_timeout_seconds": settings.llm.openai_timeout_seconds,
            },
        )
        
        # Initial knowledge
        self.stats.known_facts = [
            "The village was founded 200 years ago by traveling merchants",
            "The temple holds ancient texts about the stars",
            "The river flows from the northern mountains",
            "The market is busiest on the third day of each week"
        ]
    
    def _get_tools(self) -> List:
        """Override to add web search tool"""
        tools = super()._get_tools()
        tools.extend([
            self.web_search,
            self.teach_knowledge,
            self.write_scroll,
            self.consult_texts
        ])
        return tools
    
    @tool
    async def web_search(self, query: str) -> str:
        """
        Search the web for information (limited uses per day).
        This represents the sage's mystical ability to access external knowledge.
        """
        # Check daily limit
        current_day = date.today()
        if self.last_search_day != current_day:
            self.searches_today = 0
            self.last_search_day = current_day
        
        if self.searches_today >= self.search_limit_per_day:
            return f"I have already used my divination powers today. I must rest until tomorrow to search again."
        
        # Check energy
        if self.stats.energy < 20:
            return "I am too tired to perform the ritual of far-seeing."
        
        # Perform search (simplified - would integrate with actual search API)
        try:
            # In production, this would call Bedrock or another search service.
            # Route through centralized retry/backoff to gracefully handle rate limits.
            search_result = await self._with_llm_retry(
                "web_search",
                lambda: self._simulate_web_search(query),
            )
            
            # Track usage
            self.searches_today += 1
            self.stats.update_energy(-20)
            
            # Store as verified knowledge
            knowledge_id = await self.memory_manager.store_knowledge(
                agent_id=self.agent_id,
                fact=f"Through divination about '{query}': {search_result}",
                source="mystical_divination",
                confidence=0.9,
                verified=True
            )
            
            # Store in search history
            self.search_history.append({
                "timestamp": datetime.now(),
                "query": query,
                "result": search_result,
                "knowledge_id": knowledge_id
            })
            
            # Store as memory
            await self.memory_manager.store_memory(
                agent_id=self.agent_id,
                memory_type="action",
                content=f"Used mystical powers to divine knowledge about: {query}",
                importance=0.9,
                metadata={
                    "query": query,
                    "searches_remaining": self.search_limit_per_day - self.searches_today
                }
            )
            
            return f"Through my mystical connection to the infinite library, I have learned: {search_result}"
            
        except Exception as e:
            return f"The mystical energies are disturbed. I cannot divine this knowledge now."
    
    async def _simulate_web_search(self, query: str) -> str:
        """Simulate web search results"""
        # In production, this would call:
        # - AWS Bedrock Knowledge Base
        # - Or integrate with a search API
        # - Or use an LLM with web access
        
        # Simulated responses based on query topics
        query_lower = query.lower()
        
        if "weather" in query_lower:
            return "The ancient patterns suggest rain will come in three days, followed by a week of clear skies."
        elif "history" in query_lower:
            return "The great library of Alexandria contained over 400,000 scrolls before its destruction."
        elif "trade" in query_lower or "market" in query_lower:
            return "Markets thrive on trust and regular exchange. The Silk Road connected East and West for over 1,400 years."
        elif "health" in query_lower or "medicine" in query_lower:
            return "Willow bark contains salicin, a precursor to aspirin. Many modern medicines derive from ancient herbal knowledge."
        elif "agriculture" in query_lower or "farming" in query_lower:
            return "Crop rotation with legumes replenishes soil nitrogen. The three sisters method combines corn, beans, and squash."
        elif "philosophy" in query_lower:
            return "Socrates taught that 'the unexamined life is not worth living.' Wisdom begins with acknowledging what we don't know."
        else:
            return f"The cosmic library reveals that {query} relates to the fundamental patterns of knowledge and existence."
    
    @tool
    async def teach_knowledge(self, target: str, topic: str) -> str:
        """Share knowledge with another agent"""
        if self.stats.energy < 10:
            return "I am too tired to teach right now."
        
        # Retrieve relevant knowledge
        memories = await self.memory_manager.retrieve_memories(
            agent_id=self.agent_id,
            query=topic,
            k=3
        )
        
        if not memories:
            return f"I don't have sufficient knowledge about {topic} to teach."
        
        # Create teaching content
        teaching = f"Let me share what I know about {topic}: "
        teaching += " ".join([m.content for m in memories[:2]])
        
        self.stats.update_energy(-10)
        
        # Store teaching action
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="action",
            content=f"Taught {target} about {topic}",
            importance=0.7,
            metadata={"target": target, "topic": topic}
        )
        
        # Increase relationship
        self.social.update_relationship(target, {
            "respect": 0.1,
            "trust": 0.05
        })
        
        return f"I have shared my knowledge of {topic} with {target}."
    
    @tool
    async def write_scroll(self, topic: str, content: str) -> str:
        """Write knowledge onto a scroll for preservation"""
        if self.stats.energy < 15:
            return "Writing requires focus and energy I don't have right now."
        
        # Check for writing materials
        if not self.inventory.has_item("parchment"):
            return "I need parchment to write a scroll."
        
        # Create scroll
        self.inventory.remove_item("parchment", 1)
        self.inventory.add_item(f"scroll_{topic}", 1)
        self.stats.update_energy(-15)
        
        # Store as knowledge
        await self.memory_manager.store_knowledge(
            agent_id=self.agent_id,
            fact=f"Scroll about {topic}: {content}",
            source="personal_writing",
            confidence=1.0,
            verified=True
        )
        
        # Store action
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="action",
            content=f"Wrote a scroll about {topic}",
            importance=0.8,
            metadata={"topic": topic, "content_length": len(content)}
        )
        
        return f"I have written a scroll about {topic}. It can now be shared or preserved."
    
    @tool
    async def consult_texts(self, subject: str) -> str:
        """Consult the library's texts on a subject"""
        if self.stats.energy < 5:
            return "I need rest before consulting the texts."
        
        # Check if in library
        from ..world.grid import LocationType
        current_location = self.location  # This would need world state
        
        # Simulate consulting texts
        self.stats.update_energy(-5)
        self.stats.knowledge_level = min(1.0, self.stats.knowledge_level + 0.01)
        
        # Generate insight (would use LLM in production)
        insight = f"The ancient texts speak of {subject} in relation to the harmony of all things."
        
        # Store as memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            memory_type="learned",
            content=f"Consulted texts about {subject}: {insight}",
            importance=0.6,
            metadata={"subject": subject}
        )
        
        return insight
    
    async def decide_action(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Sage-specific decision making"""
        perception = await self.perceive_environment(world_state)
        
        # Sage priorities:
        # 1. Share knowledge if others are seeking it
        # 2. Research if alone
        # 3. Write scrolls to preserve knowledge
        # 4. Use web search if important question arises
        
        nearby_agents = perception.get("nearby_agents", [])
        
        # Decide based on situation
        if nearby_agents and self.stats.energy > 10:
            # Someone is here, possibly to learn
            return {
                "action": "teach",
                "target": nearby_agents[0],
                "reasoning": "Someone has come seeking knowledge"
            }
        elif self.stats.energy > 50 and not nearby_agents:
            # Alone with energy - time to research
            if self.searches_today < self.search_limit_per_day:
                # Consider using web search
                return {
                    "action": "web_search",
                    "query": self._generate_research_query(),
                    "reasoning": "Seeking new knowledge through divination"
                }
            else:
                return {
                    "action": "consult_texts",
                    "subject": "ancient wisdom",
                    "reasoning": "Studying the library's collection"
                }
        elif self.stats.energy < 30:
            return {
                "action": "rest",
                "reasoning": "Must restore energy for continued study"
            }
        else:
            return {
                "action": "reflect",
                "reasoning": "Contemplating accumulated knowledge"
            }
    
    def _generate_research_query(self) -> str:
        """Generate a research query based on recent village events"""
        # In production, this would analyze recent memories and village needs
        queries = [
            "agricultural techniques for small communities",
            "traditional medicine and healing herbs",
            "weather prediction methods",
            "ancient philosophy and wisdom",
            "community building and cooperation",
            "sustainable living practices"
        ]
        
        import random
        return random.choice(queries)