# Simulation TODO List

## Current Configuration
- **Checkpoints**: Daily (saved at end of each simulated day)
- **Ticks per day**: 24 (configured in settings)

## ✅ Fixed Issues

### 1. ~~Fix 'store_observation' AttributeError~~ ✅
**Solution**: Added `store_observation` method to GraphitiMemoryManager that wraps `ingest_observation`

### 2. ~~Empty Content in Episodic Nodes~~ ✅
**Solutions Implemented**:
- Fixed truncation logic bug (was using post-truncation length)
- Added empty content checks with placeholder generation
- Enhanced conversation storage to include human-readable format
- Improved episode_body construction with rich context
- All episodes now include: agent ID, content, type, location, and timestamp

### 3. ~~UI Action Column Shows Last Action~~ ✅
**Solutions Implemented**:
- Added `last_action` and `last_action_time` tracking to agents
- Dashboard now shows current action (bold) or last action
- Actions properly tracked through execution cycle

## 🟡 Improvements

### 4. Performance Monitoring
- Add metrics for episode processing time
- Track timeout occurrences
- Monitor memory operation batch sizes

### 5. Memory System Enhancements
- Verify group_id isolation is working correctly
- Check if entity_edges are being properly created
- Ensure content is preserved through truncation

### 6. Checkpoint System
- Add checkpoint validation on load
- Implement checkpoint rotation (keep last N checkpoints)
- Add checkpoint metadata (agent count, memory stats)

## 🟢 Documentation Updates

### 7. Configuration Documentation
- Document all environment variables
- Create config.json.example file
- Document tick duration and its effects

### 8. Debugging Guide
- Common errors and solutions
- How to inspect Neo4j data
- Troubleshooting memory issues

## 📝 Code Quality

### 9. Error Handling
- Add more descriptive error messages
- Implement proper logging for failures
- Add retry logic for transient failures

### 10. Testing
- Create unit tests for memory operations
- Test checkpoint save/load cycle
- Validate JSON truncation edge cases

## 🚀 Future Features

### 11. Advanced Monitoring
- Real-time memory graph visualization
- Token usage prediction
- Agent behavior analytics

### 12. Simulation Controls
- Pause/resume during simulation
- Speed controls (tick duration)
- Skip to specific day/tick

### 13. Memory Analysis
- Memory importance scoring
- Relationship strength visualization
- Knowledge propagation tracking

## Quick Fixes Needed Now

```python
# 1. Add to GraphitiMemoryManager class:
async def store_observation(self, agent_id: str, observation: str, **kwargs):
    """Store an observation for an agent."""
    return await self.add_episode(
        episode_data={
            "agent_id": agent_id,
            "content": observation,
            "type": "observation",
            **kwargs
        },
        group_id=self.group_id,
        source_description=f"{agent_id} observation"
    )

# 2. Fix content storage in _add_episode_safe:
# Ensure content is included in episode_data before truncation

# 3. Update dashboard to show current action:
# Track agent.current_action separately from agent.state
```