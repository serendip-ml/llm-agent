# Agent Architecture

**Status:** Draft
**Date:** 2026-01-24

---

## Overview

A learning agent that improves through feedback. The core thesis: true personalization requires
changing model behavior, not just retrieving context. This agent collects feedback, builds
preference pairs, and enables fine-tuning.

### Goals

1. **Remember** - Store facts about the user (preferences, background, rules)
2. **Recall** - Retrieve relevant facts when building context
3. **Learn** - Collect feedback, generate preference pairs for training
4. **Improve** - Load fine-tuned adapters that reflect learned preferences

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Agent                                  │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  LLMBackend │  │  FactStore  │  │  FeedbackCollector      │  │
│  │             │  │             │  │                         │  │
│  │  - complete │  │  - add      │  │  - track_response       │  │
│  │  - adapter  │  │  - search   │  │  - record_feedback      │  │
│  │             │  │  - delete   │  │  - export_preferences   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    ContextBuilder                          │  │
│  │                                                            │  │
│  │  base_prompt + relevant_facts + query → system_prompt      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### Agent

The main entry point. Coordinates all components.

```python
class Agent:
    """Learning agent that improves through feedback."""

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMBackend,
        facts: FactStore,
        feedback: FeedbackCollector,
    ) -> None: ...

    # === Core operations ===

    def complete(
        self,
        query: str,
        base_prompt: str | None = None,
        rag: RAGConfig | None = None,
    ) -> CompletionResult:
        """Generate a response with context-augmented prompt.

        Args:
            query: User input
            base_prompt: System prompt (uses default if None)
            rag: RAG configuration for fact retrieval (None = include all facts)

        Returns:
            Completion result with response and metadata
        """
        ...

    # === Memory ===

    def remember(self, fact: str, category: str = "general") -> int:
        """Store a fact. Returns fact ID."""
        ...

    def forget(self, fact_id: int) -> None:
        """Remove a stored fact."""
        ...

    def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        """Retrieve relevant facts for a query."""
        ...

    # === Feedback ===

    def feedback(
        self,
        response_id: str,
        signal: Literal["positive", "negative"],
        correction: str | None = None,
    ) -> None:
        """Record feedback on a response.

        Args:
            response_id: ID from CompletionResult
            signal: Whether response was good or bad
            correction: If negative, the preferred response
        """
        ...

    # === Training data ===

    def export_preferences(self, output_path: str) -> ExportResult:
        """Export preference pairs for DPO training."""
        ...

    # === Adapter ===

    def load_adapter(self, adapter_path: str) -> None:
        """Load a fine-tuned adapter."""
        ...

    def unload_adapter(self) -> None:
        """Revert to base model."""
        ...
```

### AgentConfig

```python
class AgentConfig(BaseModel):
    """Agent configuration."""

    name: str                              # Agent identifier
    default_prompt: str = "You are a helpful assistant."
    model: str = "default"                 # Model identifier for LLM backend

    # Fact injection
    fact_injection: Literal["all", "rag", "none"] = "all"
    max_facts: int = 20                    # Max facts to inject

    # RAG defaults (when fact_injection = "rag")
    rag_top_k: int = 5
    rag_min_similarity: float = 0.3
```

---

### LLMBackend

Interface for LLM inference. Supports adapter loading for personalization.

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
    """Backend that calls an OpenAI-compatible HTTP API."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None: ...


class LocalBackend(LLMBackend):
    """Backend that calls local llm-infer server."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None: ...
```

---

### FactStore

Stores facts about the user. Supports semantic search for RAG.

```python
class FactStore(Protocol):
    """Storage for user facts."""

    def add(
        self,
        content: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a fact. Returns fact ID."""
        ...

    def get(self, fact_id: int) -> Fact | None:
        """Get a fact by ID."""
        ...

    def delete(self, fact_id: int) -> None:
        """Delete a fact."""
        ...

    def list(
        self,
        category: str | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """List facts, optionally filtered by category."""
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[ScoredFact]:
        """Semantic search for relevant facts."""
        ...

    def count(self) -> int:
        """Total number of facts."""
        ...


class Fact(BaseModel):
    """A stored fact."""

    id: int
    content: str
    category: str
    metadata: dict[str, Any]
    created_at: datetime


class ScoredFact(BaseModel):
    """Fact with similarity score from search."""

    fact: Fact
    similarity: float
```

#### Implementations

```python
class SQLiteFactStore(FactStore):
    """SQLite-backed fact storage with optional embeddings."""

    def __init__(
        self,
        db_path: str,
        embedder: Embedder | None = None,
    ) -> None: ...


class MemoryFactStore(FactStore):
    """In-memory fact storage for testing."""

    def __init__(self) -> None: ...
```

#### Embedder (for RAG)

```python
class Embedder(Protocol):
    """Generate embeddings for semantic search."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class HTTPEmbedder(Embedder):
    """Embedder that calls an HTTP API."""

    def __init__(self, base_url: str, model: str = "default") -> None: ...
```

---

### FeedbackCollector

Tracks responses and collects feedback for training.

```python
class FeedbackCollector:
    """Collects feedback and generates preference pairs."""

    def __init__(self, storage: FeedbackStorage) -> None: ...

    def track_response(
        self,
        response_id: str,
        context: str,
        query: str,
        response: str,
        model: str,
    ) -> None:
        """Track a response for later feedback.

        Call this after generating a response to enable feedback collection.
        """
        ...

    def record_feedback(
        self,
        response_id: str,
        signal: Literal["positive", "negative"],
        correction: str | None = None,
    ) -> None:
        """Record feedback on a tracked response.

        Args:
            response_id: ID of the tracked response
            signal: Whether response was good or bad
            correction: If negative, the preferred response (creates preference pair)
        """
        ...

    def export_preferences(
        self,
        output_path: str,
        format: Literal["dpo", "jsonl"] = "dpo",
    ) -> ExportResult:
        """Export preference pairs for training.

        Args:
            output_path: Where to write the export
            format: Output format (dpo = DPO training format)

        Returns:
            Export statistics
        """
        ...

    def stats(self) -> FeedbackStats:
        """Get feedback statistics."""
        ...


class FeedbackStats(BaseModel):
    """Statistics about collected feedback."""

    total_tracked: int                     # Responses tracked
    positive_feedback: int                 # Thumbs up
    negative_feedback: int                 # Thumbs down
    preference_pairs: int                  # Pairs with corrections
    last_feedback_at: datetime | None
```

#### Storage

```python
class FeedbackStorage(Protocol):
    """Storage for tracked responses and feedback."""

    def save_response(self, record: ResponseRecord) -> None: ...
    def get_response(self, response_id: str) -> ResponseRecord | None: ...
    def save_feedback(self, feedback: FeedbackRecord) -> None: ...
    def list_preferences(self) -> list[PreferencePair]: ...
    def stats(self) -> FeedbackStats: ...


class ResponseRecord(BaseModel):
    """Tracked response awaiting feedback."""

    id: str
    context: str                           # System prompt used
    query: str                             # User input
    response: str                          # Generated response
    model: str
    created_at: datetime


class FeedbackRecord(BaseModel):
    """Feedback on a response."""

    response_id: str
    signal: Literal["positive", "negative"]
    correction: str | None                 # Preferred response if negative
    created_at: datetime


class PreferencePair(BaseModel):
    """Preference pair for DPO training."""

    context: str                           # System prompt + query
    chosen: str                            # Preferred response (correction)
    rejected: str                          # Original response
    margin: float = 1.0                    # Preference strength
```

---

### ContextBuilder

Builds context-augmented prompts by injecting relevant facts.

```python
class ContextBuilder:
    """Builds prompts with fact injection."""

    def __init__(self, facts: FactStore) -> None: ...

    def build(
        self,
        base_prompt: str,
        query: str | None = None,
        mode: Literal["all", "rag", "none"] = "all",
        max_facts: int = 20,
        rag_config: RAGConfig | None = None,
    ) -> str:
        """Build a system prompt with facts injected.

        Args:
            base_prompt: Base system prompt
            query: User query (required for RAG mode)
            mode: How to select facts
                - "all": Include all facts (up to max_facts)
                - "rag": Semantic search based on query
                - "none": No fact injection
            max_facts: Maximum facts to include
            rag_config: Configuration for RAG mode

        Returns:
            System prompt with facts section
        """
        ...


class RAGConfig(BaseModel):
    """Configuration for RAG-based fact retrieval."""

    top_k: int = 5
    min_similarity: float = 0.3
    categories: list[str] | None = None   # Filter by category
```

#### Prompt Format

```
{base_prompt}

## Known facts about the user

- User prefers concise responses
- User is a Python expert with 10 years experience
- User dislikes excessive emoji

## Conversation
```

---

## Storage Schema

### SQLite

```sql
-- Facts
CREATE TABLE facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    metadata    TEXT,  -- JSON
    embedding   BLOB,  -- Optional, for RAG
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_facts_category ON facts(category);

-- Tracked responses
CREATE TABLE responses (
    id          TEXT PRIMARY KEY,
    context     TEXT NOT NULL,
    query       TEXT NOT NULL,
    response    TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Feedback
CREATE TABLE feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id TEXT NOT NULL REFERENCES responses(id),
    signal      TEXT NOT NULL,  -- 'positive' or 'negative'
    correction  TEXT,           -- Preferred response if negative
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_feedback_response ON feedback(response_id);
```

---

## Usage Example

```python
from llm_agent import Agent, AgentConfig
from llm_agent.llm import LocalBackend
from llm_agent.facts import SQLiteFactStore
from llm_agent.feedback import FeedbackCollector, SQLiteFeedbackStorage

# Create components
llm = LocalBackend(base_url="http://localhost:8000")
facts = SQLiteFactStore("agent.db")
feedback = FeedbackCollector(SQLiteFeedbackStorage("agent.db"))

# Create agent
agent = Agent(
    config=AgentConfig(
        name="assistant",
        default_prompt="You are a helpful coding assistant.",
    ),
    llm=llm,
    facts=facts,
    feedback=feedback,
)

# Store facts
agent.remember("User prefers concise responses", category="preferences")
agent.remember("User is a Python expert", category="background")

# Generate response (facts automatically injected)
result = agent.complete("How do I read a file in Python?")
print(result.content)

# Record feedback
agent.feedback(result.id, "positive")

# Or with correction
agent.feedback(
    result.id,
    "negative",
    correction="Use pathlib: Path('file.txt').read_text()",
)

# Export for training when enough feedback collected
if feedback.stats().preference_pairs >= 100:
    agent.export_preferences("training_data.jsonl")
```

---

## Implementation Phases

### Phase 1: Core Agent

- [ ] `Agent` class with config
- [ ] `LLMBackend` protocol + `HTTPBackend` implementation
- [ ] `Message`, `CompletionResult` types
- [ ] Basic completion flow (no facts, no feedback)

### Phase 2: Memory

- [ ] `FactStore` protocol
- [ ] `MemoryFactStore` for testing
- [ ] `SQLiteFactStore` for persistence
- [ ] `ContextBuilder` with "all" mode
- [ ] Fact injection into prompts

### Phase 3: Feedback

- [ ] `FeedbackCollector` with response tracking
- [ ] `FeedbackStorage` protocol + SQLite implementation
- [ ] Preference pair generation from corrections
- [ ] Export to DPO format

### Phase 4: RAG

- [ ] `Embedder` protocol + HTTP implementation
- [ ] Embedding storage in SQLiteFactStore
- [ ] Semantic search in `FactStore.search()`
- [ ] `ContextBuilder` with "rag" mode

### Phase 5: Adapters

- [ ] `LLMBackend.load_adapter()` / `unload_adapter()`
- [ ] Adapter path management in Agent
- [ ] Integration with llm-infer adapter endpoints

---

## Open Questions

| Question | Options | Notes |
|----------|---------|-------|
| Embedding storage | SQLite BLOB vs separate vector DB | Start with SQLite, migrate if needed |
| Embedding model | Local vs API | Depends on llm-infer capabilities |
| Fact deduplication | Exact match vs semantic | Start with exact, add semantic later |
| Feedback expiry | Keep forever vs rolling window | Keep forever, prune manually |

---

## Success Criteria

| Criterion | How to verify |
|-----------|---------------|
| Facts persist | Restart agent, facts still there |
| Facts injected | Check system prompt includes facts |
| Feedback tracked | Query shows response was tracked |
| Preferences exported | File contains valid DPO format |
| Adapter loads | Responses change after loading adapter |
