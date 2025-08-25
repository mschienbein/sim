"""
Rate limiting and token budget management for cost control.
"""

from typing import Dict, Optional, Any
from datetime import datetime, date
import json 

from src.config.settings import settings

class TokenBudgetManager:
    """Manages token budgets to control LLM costs"""
    
    def __init__(self):
        self.daily_budget = settings.rate_limit.daily_token_budget
        self.per_agent_limit = settings.rate_limit.per_agent_token_limit
        
        # Tracking
        self.used_today = 0
        self.agent_usage: Dict[str, int] = {}
        self.last_reset_date = date.today()
        self.total_used = 0
        
        # History for analytics
        self.usage_history = []
        
    def can_call_llm(self, agent_id: str, estimated_tokens: int) -> bool:
        """Check if an LLM call is within budget"""
        # Check if we need to reset daily budget
        self._check_daily_reset()
        
        # Check daily budget
        if self.used_today + estimated_tokens > self.daily_budget:
            return False
        
        # Check per-agent limit
        current_usage = self.agent_usage.get(agent_id, 0)
        if current_usage + estimated_tokens > self.per_agent_limit:
            return False
        
        return True
    
    def track_usage(self, agent_id: str, actual_tokens: int):
        """Track actual token usage"""
        self.used_today += actual_tokens
        self.total_used += actual_tokens
        self.agent_usage[agent_id] = self.agent_usage.get(agent_id, 0) + actual_tokens
        
        # Record in history
        self.usage_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "tokens": actual_tokens,
            "daily_total": self.used_today
        })
    
    def _check_daily_reset(self):
        """Check if we need to reset daily counters"""
        current_date = date.today()
        if current_date > self.last_reset_date:
            self.reset_daily_budget()
            self.last_reset_date = current_date
    
    def reset_daily_budget(self):
        """Reset daily budget counters"""
        # Save daily summary before reset
        if self.used_today > 0:
            self._save_daily_summary()
        
        self.used_today = 0
        self.agent_usage = {}
    
    def _save_daily_summary(self):
        """Save daily usage summary"""
        summary = {
            "date": self.last_reset_date.isoformat(),
            "total_tokens": self.used_today,
            "agent_breakdown": self.agent_usage.copy(),
            "cost_estimate": self.used_today * 0.00001  # Rough estimate
        }
        
        # Save to file
        summary_file = settings.logs_dir / f"token_usage_{self.last_reset_date.isoformat()}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_remaining_budget(self) -> Dict[str, int]:
        """Get remaining budgets"""
        return {
            "daily_remaining": self.daily_budget - self.used_today,
            "daily_used": self.used_today,
            "daily_percentage": (self.used_today / self.daily_budget) * 100
        }
    
    def get_agent_usage(self, agent_id: str) -> Dict[str, int]:
        """Get usage for specific agent"""
        used = self.agent_usage.get(agent_id, 0)
        return {
            "used": used,
            "remaining": self.per_agent_limit - used,
            "percentage": (used / self.per_agent_limit) * 100
        }
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get comprehensive usage report"""
        return {
            "daily": {
                "budget": self.daily_budget,
                "used": self.used_today,
                "remaining": self.daily_budget - self.used_today
            },
            "total": {
                "tokens": self.total_used,
                "estimated_cost": self.total_used * 0.00001
            },
            "by_agent": self.agent_usage.copy(),
            "top_users": sorted(
                self.agent_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }

class ConversationLimiter:
    """Limits conversation length to control costs"""
    
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.active_conversations: Dict[str, int] = {}
    
    def can_continue_conversation(self, conversation_id: str) -> bool:
        """Check if conversation can continue"""
        turns = self.active_conversations.get(conversation_id, 0)
        return turns < self.max_turns
    
    def track_turn(self, conversation_id: str):
        """Track a conversation turn"""
        self.active_conversations[conversation_id] = \
            self.active_conversations.get(conversation_id, 0) + 1
    
    def end_conversation(self, conversation_id: str):
        """End a conversation"""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
    
    def reset_all(self):
        """Reset all conversation tracking"""
        self.active_conversations = {}

class ActionThrottler:
    """Throttle agent actions to prevent runaway costs"""
    
    def __init__(self):
        self.action_cooldowns: Dict[str, Dict[str, datetime]] = {}
        self.action_limits = {
            "web_search": 86400,  # 1 day in seconds
            "reflect": 3600,       # 1 hour
            "work": 1800,          # 30 minutes
            "trade": 600,          # 10 minutes
            "speak": 60            # 1 minute
        }
    
    def can_perform_action(self, agent_id: str, action: str) -> bool:
        """Check if agent can perform action"""
        if action not in self.action_limits:
            return True
        
        if agent_id not in self.action_cooldowns:
            self.action_cooldowns[agent_id] = {}
        
        last_performed = self.action_cooldowns[agent_id].get(action)
        if not last_performed:
            return True
        
        elapsed = (datetime.now() - last_performed).total_seconds()
        return elapsed >= self.action_limits[action]
    
    def record_action(self, agent_id: str, action: str):
        """Record that an action was performed"""
        if agent_id not in self.action_cooldowns:
            self.action_cooldowns[agent_id] = {}
        self.action_cooldowns[agent_id][action] = datetime.now()
    
    def get_cooldown_remaining(self, agent_id: str, action: str) -> Optional[float]:
        """Get remaining cooldown time in seconds"""
        if action not in self.action_limits:
            return None
        
        if agent_id not in self.action_cooldowns:
            return 0
        
        last_performed = self.action_cooldowns[agent_id].get(action)
        if not last_performed:
            return 0
        
        elapsed = (datetime.now() - last_performed).total_seconds()
        cooldown = self.action_limits[action]
        
        if elapsed >= cooldown:
            return 0
        
        return cooldown - elapsed

class CostOptimizer:
    """Optimize LLM usage for cost efficiency"""
    
    def __init__(self):
        self.model_costs = {
            "gpt-5-nano": 0.00001,      # Extremely cheap model (cheaper than gpt-3.5)
            "gpt-4o-mini": 0.00015,     # $0.15 per 1M input tokens = $0.00015 per 1k tokens
            "gpt-4o": 0.0025,           # $2.50 per 1M input tokens = $0.0025 per 1k tokens
            "gpt-4": 0.03               # $30 per 1M input tokens = $0.03 per 1k tokens (expensive)
        }
        
        self.action_priorities = {
            "rest": 0,          # No LLM needed
            "observe": 1,       # Minimal LLM
            "move": 1,          # Minimal LLM
            "work": 2,          # Simple LLM
            "speak": 3,         # Full LLM
            "trade": 3,         # Full LLM
            "reflect": 4,       # Heavy LLM
            "web_search": 5     # Most expensive
        }
    
    def should_use_llm(self, action: str, agent_importance: float = 0.5) -> bool:
        """Decide if LLM should be used for action"""
        priority = self.action_priorities.get(action, 2)
        
        # Always use LLM for high priority actions
        if priority >= 4:
            return True
        
        # Never use LLM for zero priority
        if priority == 0:
            return False
        
        # Probabilistic for medium priority
        import random
        threshold = 0.3 + (agent_importance * 0.4)
        return random.random() < threshold
    
    def select_model(self, action: str, budget_remaining: float) -> str:
        """Select appropriate model based on action and budget - always use gpt-5-nano for maximum cost efficiency"""
        # Always return gpt-5-nano for all actions - it's extremely cheap and effective
        return "gpt-5-nano"
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        cost_per_1k = self.model_costs.get(model, 0.001)
        return (tokens / 1000) * cost_per_1k
    
    def optimize_prompt(self, prompt: str, max_length: int = 500) -> str:
        """Optimize prompt length to reduce tokens"""
        if len(prompt) <= max_length:
            return prompt
        
        # Truncate intelligently
        # Keep beginning and end, remove middle
        keep_start = max_length // 2
        keep_end = max_length // 2
        
        return prompt[:keep_start] + "\n...[truncated]...\n" + prompt[-keep_end:]