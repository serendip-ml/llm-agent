# Learning Agent Model

Notes on how the agent should work and learn over time.

## Mental Model

The agent works like Claude Code solving a problem - but autonomously, without a user in the loop.

1. **Identity** - Who the agent is (defined in YAML)
2. **Method** - How the agent operates (can evolve over time)
3. **Task** - Problem or question to work through
4. **Conversation** - Agent works through the problem across multiple LLM requests, maintaining context
5. **Conclusion** - Agent determines if task is solved, unsolvable, or needs help
6. **Learning** - Conclusions and feedback accumulate in llm-learn

## Key Insight: Failure is Training Data

The agent will struggle heavily in the beginning. But through user feedback and 1000+ iterations,
it should become significantly better.

- Claude Code has the user as real-time feedback loop
- This agent develops its own feedback loop, augmented by occasional user feedback

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     Single Session                          │
│  Identity + Method + Task → Conversation → Conclusion       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  User Feedback  │  (optional, async)
                    │  "wrong because │
                    │   X" / "good"   │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     llm-learn                               │
│  - Facts about codebase                                     │
│  - What worked / what didn't                                │
│  - User corrections                                         │
│  - Preference pairs (agent output vs user correction)       │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼ (recall on next session)
                    ┌─────────────────┐
                    │  Next Session   │
                    │  Agent is       │
                    │  slightly better│
                    └─────────────────┘
```

## Implementation Status

### Done

1. **Conversation persistence** - `llm_agent/core/conversation/`
   - Maintains context across `run_once()` cycles
   - Token tracking and compaction when approaching limit
   - `SlidingWindowCompactor` (drops old) and `SummarizingCompactor` (LLM summary)

### To Do

2. **Session completion** - Agent needs to know when it's done with current attempt
   - Even if conclusion is wrong, it should recognize "I'm done trying"
   - Possible states: WORKING, SOLVED, UNSOLVABLE, STUCK

3. **Conclusion storage** - Save what agent concluded to llm-learn
   - Store the conclusion with context
   - Tag with task type, confidence, etc.

4. **Feedback integration** - User feedback flows to llm-learn
   - "That was wrong because X" → negative signal + correction
   - "Good insight" → positive signal
   - Creates preference pairs for future fine-tuning

5. **Recall at session start** - Inject relevant past learnings into context
   - Query llm-learn for relevant facts before starting
   - Include in system prompt or early conversation

## Open Questions

- How does agent recognize it's "done" with a task attempt?
- Should this be a tool (`complete_task(status, conclusion)`)?
- Or detected from structured output?
- Or inferred from LLM response patterns?

- How often should user feedback be solicited?
- How do we handle conflicting learnings over time?
