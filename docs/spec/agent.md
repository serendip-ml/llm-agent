# Agent Architecture

**Status:** Draft
**Date:** 2026-01-25

---

## Overview

A learning agent that improves through feedback. The core thesis: true personalization requires
changing model behavior, not just retrieving context. This agent collects feedback, builds
preference pairs, and enables fine-tuning.

### Stack

```
appinfra → llm-infer → llm-learn → llm-agent
```

| Layer | Responsibility |
|-------|----------------|
| `appinfra` | Logging, configuration, tracing |
| `llm-infer` | LLM serving, adapter loading |
| `llm-learn` | Facts, feedback, preferences, embeddings, training |
| `llm-agent` | Agent coordination, LLM completion, conversation management |

### Goals

1. **Remember** - Store facts about the user (via `llm-learn`)
2. **Recall** - Retrieve relevant facts when building context (via `llm-learn`)
3. **Learn** - Collect feedback, generate preference pairs (via `llm-learn`)
4. **Improve** - Load fine-tuned adapters that reflect learned preferences

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Agent                                       │
│                                                                          │
│  ┌──────────────┐  ┌────────────────────────────────────────────────┐   │
│  │  LLMBackend  │  │              LearnClient (llm-learn)           │   │
│  │              │  │                                                 │   │
│  │  - complete  │  │  ┌──────────┐ ┌──────────┐ ┌───────────────┐   │   │
│  │  - adapter   │  │  │  facts   │ │ feedback │ │  preferences  │   │   │
│  │              │  │  └──────────┘ └──────────┘ └───────────────┘   │   │
│  └──────────────┘  │                                                 │   │
│                    │  ┌──────────────────┐  ┌──────────────────┐    │   │
│                    │  │  ContextBuilder  │  │     Embedder     │    │   │
│                    │  └──────────────────┘  └──────────────────┘    │   │
│                    └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**What `llm-agent` provides:**
- `Agent` - Coordinator that wires LLM and learning together
- `LLMBackend` - Protocol for LLM inference with adapter support
- `HTTPBackend` - OpenAI-compatible HTTP implementation

**What comes from `llm-learn`:**
- `LearnClient` - Main entry point for learning data
- `FactsClient` - Fact storage and retrieval
- `FeedbackClient` - Feedback signal recording
- `PreferencesClient` - Preference pair storage
- `ContextBuilder` - System prompt construction with fact injection
- `Embedder` - Embedding generation for RAG
- Training export and LoRA fine-tuning

---

## Components

### Agent

The main entry point. Coordinates LLM backend with learning infrastructure.

```python
from llm_learn import LearnClient
from llm_learn.inference import ContextBuilder, Embedder, RAGArgs

class Agent:
    """Learning agent that improves through feedback."""

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMBackend,
        learn: LearnClient,
        embedder: Embedder | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            config: Agent configuration.
            llm: LLM backend for completions.
            learn: Learning client from llm-learn.
            embedder: Embedder for RAG (optional, enables semantic search).
        """
        ...

    # === Core operations ===

    def complete(
        self,
        query: str,
        system_prompt: str | None = None,
        rag: RAGArgs | None = None,
    ) -> CompletionResult:
        """Generate a response with context-augmented prompt.

        Args:
            query: User input.
            system_prompt: System prompt (uses default if None).
            rag: RAG configuration for fact retrieval (None = include all facts).

        Returns:
            Completion result with response and metadata.
        """
        ...

    # === Memory (delegates to llm-learn) ===

    def remember(self, fact: str, category: str = "general") -> int:
        """Store a fact. Returns fact ID."""
        return self._learn.facts.add(fact, category=category)

    def forget(self, fact_id: int) -> None:
        """Remove a stored fact."""
        self._learn.facts.delete(fact_id)

    def recall(self, query: str, top_k: int = 5) -> list[ScoredFact]:
        """Retrieve relevant facts for a query using RAG."""
        ...

    # === Feedback (delegates to llm-learn) ===

    def feedback(
        self,
        response_id: str,
        signal: Literal["positive", "negative"],
        correction: str | None = None,
    ) -> None:
        """Record feedback on a response.

        Args:
            response_id: ID from CompletionResult.
            signal: Whether response was good or bad.
            correction: If negative, the preferred response (creates preference pair).
        """
        ...

    # === Training data ===

    def export_preferences(
        self,
        output_path: str,
        format: Literal["dpo", "sft"] = "dpo",
    ) -> ExportResult:
        """Export preference pairs for training."""
        ...

    def training_stats(self) -> TrainingStats:
        """Get statistics about accumulated training data."""
        ...

    # === Adapter ===

    def load_adapter(self, adapter_path: str) -> None:
        """Load a fine-tuned adapter."""
        self._llm.load_adapter(adapter_path)

    def unload_adapter(self) -> None:
        """Revert to base model."""
        self._llm.unload_adapter()
```

### AgentConfig

```python
class AgentConfig(BaseModel):
    """Agent configuration."""

    name: str                              # Agent identifier
    default_prompt: str = "You are a helpful assistant."
    model: str = "default"                 # Model identifier for LLM backend

    # Fact injection mode
    fact_injection: Literal["all", "rag", "none"] = "all"
    max_facts: int = 20                    # Max facts to inject

    # RAG defaults (when fact_injection = "rag")
    rag_top_k: int = 5
    rag_min_similarity: float = 0.3
```

---

### LLMBackend

Interface for LLM inference. Supports adapter loading for personalization.

**Note:** This is the primary contribution of `llm-agent` - a clean protocol for LLM backends
that can work with `llm-infer` or any OpenAI-compatible API.

```python
class LLMBackend(Protocol):
    """LLM inference backend."""

    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Generate a completion."""
        ...

    def load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter."""
        ...

    def unload_adapter(self) -> None:
        """Unload current adapter, revert to base model."""
        ...


class Message(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionResult(BaseModel):
    """Result from LLM completion."""

    id: str                                # Unique response ID (for feedback)
    content: str                           # Generated text
    model: str                             # Model used
    tokens_used: int                       # Total tokens (prompt + completion)
    latency_ms: int                        # Response time
```

#### Implementations

```python
class HTTPBackend(LLMBackend):
    """Backend that calls an OpenAI-compatible HTTP API.

    Works with llm-infer or any OpenAI-compatible endpoint.
    Uses persistent HTTP client for connection reuse.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        default_model: str = "default",
    ) -> None: ...

    def close(self) -> None:
        """Close HTTP client and release resources."""
        ...
```

---

### Components from llm-learn

These are used by Agent but defined in `llm-learn`. See llm-learn documentation for details.

| Component | Purpose |
|-----------|---------|
| `LearnClient` | Main entry point, scoped to a profile |
| `FactsClient` | Store and retrieve facts |
| `FeedbackClient` | Record explicit feedback signals |
| `PreferencesClient` | Store preference pairs for DPO |
| `ContextBuilder` | Build system prompts with injected facts |
| `Embedder` | Generate embeddings for RAG |
| `RAGArgs` | Configuration for RAG retrieval |
| `export_preferences_dpo` | Export to DPO training format |
| `train_lora` / `train_dpo` | Fine-tune adapters |

---

## Usage Example

```python
from llm_learn import LearnClient
from llm_learn.inference import Embedder, RAGArgs

from llm_agent import Agent, AgentConfig
from llm_agent.llm import HTTPBackend

# Create LLM backend (connects to llm-infer or OpenAI-compatible API)
llm = HTTPBackend(base_url="http://localhost:8000/v1")

# Create learning client (from llm-learn)
learn = LearnClient(profile_id=1)
learn.migrate()  # Ensure database tables exist

# Optional: embedder for RAG
embedder = Embedder(base_url="http://localhost:8000/v1")

# Create agent
agent = Agent(
    config=AgentConfig(
        name="assistant",
        default_prompt="You are a helpful coding assistant.",
        fact_injection="rag",
    ),
    llm=llm,
    learn=learn,
    embedder=embedder,
)

# Store facts (persisted via llm-learn)
agent.remember("User prefers concise responses", category="preferences")
agent.remember("User is a Python expert", category="background")

# Generate response (facts automatically injected via RAG)
result = agent.complete(
    "How do I read a file in Python?",
    rag=RAGArgs(top_k=5),
)
print(result.content)

# Record feedback
agent.feedback(result.id, "positive")

# Or with correction (creates preference pair for training)
agent.feedback(
    result.id,
    "negative",
    correction="Use pathlib: Path('file.txt').read_text()",
)

# Check training data accumulation
stats = agent.training_stats()
print(f"Preference pairs: {stats.preference_count}")

# Export for training when enough feedback collected
if stats.preference_count >= 100:
    agent.export_preferences("training_data.jsonl", format="dpo")

# After training externally, load the adapter
agent.load_adapter("/models/adapters/my-adapter")
```

---

## Implementation Phases

### Phase 1: Core Agent (Done)

- [x] `Agent` class with config
- [x] `LLMBackend` protocol + `HTTPBackend` implementation
- [x] `Message`, `CompletionResult` types
- [x] Basic completion flow (no facts, no feedback)

### Phase 2: llm-learn Integration

- [ ] Add `llm-learn` as dependency
- [ ] Update `Agent.__init__` to take `LearnClient`
- [ ] Implement `remember()` / `forget()` delegating to `learn.facts`
- [ ] Implement `feedback()` delegating to `learn.feedback` / `learn.preferences`
- [ ] Wire `ContextBuilder` for fact injection in `complete()`

### Phase 3: RAG Support

- [ ] Add optional `Embedder` to Agent
- [ ] Implement `recall()` using `learn.facts.search_similar()`
- [ ] Support `RAGArgs` in `complete()` for semantic fact retrieval
- [ ] Embed facts on `remember()` when embedder is available

### Phase 4: Training Integration

- [ ] Implement `export_preferences()` using `llm-learn` exporters
- [ ] Implement `training_stats()`
- [ ] Add response tracking for feedback collection

### Phase 5: Adapter Management

- [ ] Wire `load_adapter()` / `unload_adapter()` to `llm-infer` endpoints
- [ ] Track active adapter in Agent
- [ ] Add adapter info to completion metadata

---

## Open Questions

| Question | Status | Notes |
|----------|--------|-------|
| Storage backend | Resolved | PostgreSQL via llm-learn |
| Embedding model | Resolved | Via llm-infer |
| Fact deduplication | Resolved | Handled by llm-learn |
| Training orchestration | Open | In-agent vs external workflow |

---

## Success Criteria

| Criterion | How to verify |
|-----------|---------------|
| Facts persist | Restart agent, facts still there (via llm-learn DB) |
| Facts injected | Check system prompt includes facts |
| Feedback tracked | Query `learn.feedback` shows recorded signals |
| Preferences created | Negative feedback with correction creates pair |
| Preferences exported | File contains valid DPO format |
| Adapter loads | Responses change after loading adapter |
