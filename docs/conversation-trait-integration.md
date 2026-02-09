# ConversationTrait Integration Design

## Overview

ConversationTrait adds conversation history and automatic compaction to agents. When attached, the
agent maintains context across `run_once()` or `ask()` calls.

## Architecture

```
Agent
├── SAIATrait (execution)
├── LearnTrait (memory/solutions)
├── ToolsTrait (tools)
└── ConversationTrait (history/compaction) ← NEW
```

## Integration Points

### 1. Context Retrieval (before execution)

**Current code** (agent.py:125-129):
```python
# Recall past solutions for context
context = self._recall_context(learn_trait, task, recall_strategy, recall_limit)

# Execute task
prompt = saia_trait.saia.compose(context, task)
saia_result = await saia_trait.saia.complete(prompt)
```

**With ConversationTrait**:
```python
# Get context from conversation or learning
context = self._get_context(learn_trait, conv_trait, task, recall_strategy, recall_limit)

# Execute task (conversation history included in context)
prompt = saia_trait.saia.compose(context, task)
saia_result = await saia_trait.saia.complete(prompt)
```

### 2. Turn Recording (after execution)

**Current code** (agent.py:140-144):
```python
# Persist successful outcomes
if result.success and learn_trait is not None:
    await self._persist_outcome(learn_trait, saia_trait, task, result)

return result
```

**With ConversationTrait**:
```python
# Persist successful outcomes
if result.success and learn_trait is not None:
    await self._persist_outcome(learn_trait, saia_trait, task, result)

# Add turn to conversation
if conv_trait is not None:
    conv_trait.add_turn(task, result.content)

return result
```

## Context Strategy

### Without ConversationTrait (current)
```
Context = LearnTrait.recall(task) → past solutions
```

### With ConversationTrait
```
Context = ConversationTrait.get_context() → conversation history
```

### With Both
Two options:

**Option A: Conversation replaces learning recall**
```python
if conv_trait:
    context = conv_trait.get_context()  # Use conversation
else:
    context = learn_trait.recall(...)   # Fallback to learning
```

**Option B: Conversation supplements learning** (more complex)
```python
context_parts = []
if conv_trait:
    context_parts.append(conv_trait.get_context())
if learn_trait:
    context_parts.append(learn_trait.recall(...))
context = merge_contexts(context_parts)
```

**Recommendation**: Start with Option A (conversation replaces recall)

## Usage Example

### Stateless agent (current)
```python
agent = Agent(lg, identity, "Check for new data")
agent.add_trait(SAIATrait(...))
agent.add_trait(LearnTrait(...))
agent.start()

# Each run is independent
agent.run_once()  # No context from previous runs
agent.run_once()  # No context from previous runs
```

### Conversational agent (with ConversationTrait)
```python
agent = Agent(lg, identity, "You are a helpful assistant")
agent.add_trait(SAIATrait(...))
agent.add_trait(ConversationTrait(agent, ConversationTraitConfig()))
agent.start()

# Conversation maintains context
result = agent.ask("What is 2+2?")
# → "4"

result = agent.ask("What about multiplying that by 3?")
# → "12" (knows "that" refers to 4 from previous turn)
```

## Implementation Steps

1. ✅ Create `ConversationTrait` in `core/traits/builtin/conv.py`
2. ⏸️ Update `agent.handle_task()` to:
   - Get conversation trait
   - Use conversation context if present
   - Add turn after execution
3. ⏸️ Add helper method `_get_context()` to choose context strategy
4. ⏸️ Export ConversationTrait from public API
5. ⏸️ Write tests for conversation integration
6. ⏸️ Add example showing conversational agent

## Open Questions

1. **Context strategy**: Conversation replaces recall, or supplements it?
2. **SAIA integration**: Does SAIA compose() work with Message objects, or do we format them?
3. **System prompt**: Should conversation include system prompt, or is that SAIA's job?
4. **Reset behavior**: When should conversation be reset? Manual only, or automatic triggers?

## Next Steps

Discuss integration approach, then implement the agent changes.
