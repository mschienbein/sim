"""
Manages agent-to-agent conversations using Strands A2A protocol.
"""

from typing import Dict, List, Optional, Any, Tuple
import asyncio
from datetime import datetime
import json

from strands.multiagent.a2a import A2AServer, A2AClientToolProvider

from ..agents.base_agent import SimulationAgent
from ..config.settings import settings

class ConversationManager:
    """
    Manages multi-agent conversations using Strands A2A protocol.
    """
    
    def __init__(self):
        self.active_conversations: Dict[str, 'Conversation'] = {}
        self.a2a_servers: Dict[str, A2AServer] = {}
        self.conversation_history = []
        
    async def setup_a2a_servers(self, agents: Dict[str, SimulationAgent]):
        """Set up A2A servers for all agents"""
        base_port = 8000
        
        for idx, (agent_id, agent) in enumerate(agents.items()):
            port = base_port + idx
            
            # Create A2A server for each agent
            server = A2AServer(
                agent=agent,
                port=port,
                host="127.0.0.1"
            )
            
            self.a2a_servers[agent_id] = server
            
            # Start server in background
            asyncio.create_task(server.serve())
            
        print(f"✓ Started {len(self.a2a_servers)} A2A servers")
    
    async def initiate_conversation(
        self,
        initiator_id: str,
        target_id: str,
        context: Dict[str, Any],
        max_turns: int = 5
    ) -> 'Conversation':
        """Start a new conversation between two agents"""
        
        # Create conversation ID
        conv_id = f"{initiator_id}_{target_id}_{datetime.now().timestamp()}"
        
        # Create conversation object
        conversation = Conversation(
            conversation_id=conv_id,
            initiator_id=initiator_id,
            target_id=target_id,
            context=context,
            max_turns=max_turns
        )
        
        self.active_conversations[conv_id] = conversation
        
        # Get A2A servers
        initiator_server = self.a2a_servers.get(initiator_id)
        target_server = self.a2a_servers.get(target_id)
        
        if not initiator_server or not target_server:
            raise ValueError(f"A2A servers not found for agents")
        
        # Set up A2A client tool provider for cross-agent communication
        target_port = 8000 + list(self.a2a_servers.keys()).index(target_id)
        
        client_provider = A2AClientToolProvider(
            client_name=initiator_id,
            servers={
                target_id: f"http://127.0.0.1:{target_port}"
            }
        )
        
        # Store provider in conversation
        conversation.client_provider = client_provider
        
        return conversation
    
    async def execute_turn(
        self,
        conversation: 'Conversation',
        speaker_id: str,
        message: str
    ) -> str:
        """Execute a single conversation turn"""
        
        # Check turn limit
        if conversation.turn_count >= conversation.max_turns:
            return "Conversation limit reached"
        
        # Get the listener
        listener_id = (conversation.target_id 
                      if speaker_id == conversation.initiator_id 
                      else conversation.initiator_id)
        
        # Use A2A to send message
        if conversation.client_provider:
            # Call the other agent as a tool
            response = await conversation.client_provider.call_tool(
                tool_name=f"{listener_id}.speak",
                arguments={
                    "message": message,
                    "context": conversation.context
                }
            )
        else:
            # Fallback to direct response
            response = f"I heard you say: {message}"
        
        # Record turn
        conversation.add_turn(speaker_id, listener_id, message, response)
        
        return response
    
    async def run_full_conversation(
        self,
        initiator_id: str,
        target_id: str,
        opening_message: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Run a complete conversation between two agents"""
        
        # Initialize conversation
        conversation = await self.initiate_conversation(
            initiator_id=initiator_id,
            target_id=target_id,
            context=context,
            max_turns=settings.rate_limit.max_conversation_turns
        )
        
        dialogue = []
        current_speaker = initiator_id
        current_message = opening_message
        
        # Run conversation turns
        for turn in range(conversation.max_turns):
            # Execute turn
            response = await self.execute_turn(
                conversation=conversation,
                speaker_id=current_speaker,
                message=current_message
            )
            
            # Record in dialogue
            dialogue.append({
                "speaker": current_speaker,
                "message": current_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check for conversation end signals
            if any(end_phrase in response.lower() 
                   for end_phrase in ["goodbye", "farewell", "see you later"]):
                dialogue.append({
                    "speaker": target_id if current_speaker == initiator_id else initiator_id,
                    "message": response,
                    "timestamp": datetime.now().isoformat()
                })
                break
            
            # Switch speakers
            current_speaker = target_id if current_speaker == initiator_id else initiator_id
            current_message = response
        
        # End conversation
        self.end_conversation(conversation.conversation_id)
        
        # Store in history
        self.conversation_history.append({
            "conversation_id": conversation.conversation_id,
            "participants": [initiator_id, target_id],
            "dialogue": dialogue,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        
        return dialogue
    
    def end_conversation(self, conversation_id: str):
        """End an active conversation"""
        if conversation_id in self.active_conversations:
            conv = self.active_conversations[conversation_id]
            # Clean up A2A client if needed
            if hasattr(conv, 'client_provider'):
                # Close client connections
                pass
            del self.active_conversations[conversation_id]
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs"""
        return list(self.active_conversations.keys())
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversations"""
        total_conversations = len(self.conversation_history)
        
        if total_conversations == 0:
            return {
                "total": 0,
                "active": len(self.active_conversations),
                "average_turns": 0,
                "most_talkative": None
            }
        
        # Calculate statistics
        total_turns = sum(len(conv["dialogue"]) 
                         for conv in self.conversation_history)
        
        agent_talk_count = {}
        for conv in self.conversation_history:
            for participant in conv["participants"]:
                agent_talk_count[participant] = agent_talk_count.get(participant, 0) + 1
        
        most_talkative = max(agent_talk_count.items(), 
                            key=lambda x: x[1])[0] if agent_talk_count else None
        
        return {
            "total": total_conversations,
            "active": len(self.active_conversations),
            "average_turns": total_turns / total_conversations if total_conversations > 0 else 0,
            "most_talkative": most_talkative,
            "talk_counts": agent_talk_count
        }
    
    async def cleanup(self):
        """Clean up all A2A servers"""
        for server in self.a2a_servers.values():
            # Stop server gracefully
            pass
        self.a2a_servers.clear()
        self.active_conversations.clear()


class Conversation:
    """Represents an active conversation between agents"""
    
    def __init__(
        self,
        conversation_id: str,
        initiator_id: str,
        target_id: str,
        context: Dict[str, Any],
        max_turns: int = 5
    ):
        self.conversation_id = conversation_id
        self.initiator_id = initiator_id
        self.target_id = target_id
        self.context = context
        self.max_turns = max_turns
        
        self.turns: List[Turn] = []
        self.turn_count = 0
        self.started_at = datetime.now()
        self.ended_at: Optional[datetime] = None
        
        # A2A client provider (set during initialization)
        self.client_provider: Optional[A2AClientToolProvider] = None
    
    def add_turn(self, speaker: str, listener: str, message: str, response: str):
        """Add a turn to the conversation"""
        turn = Turn(
            turn_number=self.turn_count,
            speaker=speaker,
            listener=listener,
            message=message,
            response=response,
            timestamp=datetime.now()
        )
        self.turns.append(turn)
        self.turn_count += 1
    
    def get_dialogue(self) -> List[Dict[str, str]]:
        """Get conversation as dialogue list"""
        dialogue = []
        for turn in self.turns:
            dialogue.append({
                "speaker": turn.speaker,
                "message": turn.message
            })
            if turn.response:
                dialogue.append({
                    "speaker": turn.listener,
                    "message": turn.response
                })
        return dialogue
    
    def end(self):
        """Mark conversation as ended"""
        self.ended_at = datetime.now()
    
    def duration(self) -> float:
        """Get conversation duration in seconds"""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return (datetime.now() - self.started_at).total_seconds()


class Turn:
    """Represents a single turn in a conversation"""
    
    def __init__(
        self,
        turn_number: int,
        speaker: str,
        listener: str,
        message: str,
        response: str,
        timestamp: datetime
    ):
        self.turn_number = turn_number
        self.speaker = speaker
        self.listener = listener
        self.message = message
        self.response = response
        self.timestamp = timestamp