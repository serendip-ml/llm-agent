# Review Agent Experiment

**Status:** Specification
**Date:** 2026-01-28

---

## Hypothesis

**Specialization beats generalization.** A LoRA fine-tuned medium-sized model (7B-70B) specialized
for code review can match a static prompt-driven agent running on a commercial model (Claude).

---

## Experiment Design

### Variants

| # | Variant | Model | Fine-tuned | Tools | Orchestration |
|---|---------|-------|------------|-------|---------------|
| 1 | Prompt Agent | Claude | No | Yes | Prompt-driven |
| 2 | Prompt Agent | Local (Qwen 7B-70B) | No | Yes | Prompt-driven |
| 3 | Prompt Agent + LoRA | Local | Yes | Yes | Prompt-driven |
| 4 | Programmatic Agent + LoRA | Local | Yes | Yes | Code-driven |

### Expected Outcomes

- **2 << 1**: Raw local model significantly worse than Claude
- **3 < 1**: LoRA fine-tuning narrows the gap but still trails Claude
- **4 ≈ 1**: LoRA + code orchestration matches Claude

---

## Variant Descriptions

### Variant 1: Prompt Agent (Claude)

The baseline. Uses `/pr` skill with Claude.

- **Model:** Claude (via Claude Code)
- **Tools:** Claude's built-in (git, grep, read files)
- **Learning:** None
- **Orchestration:** Prompt instructs model what to do; model decides workflow
- **Reference:** [review-code skill](https://github.com/serendip-ml/claude-code-skills/blob/main/skills/review-code/SKILL.md) (review logic)

### Variant 2: Prompt Agent (Local)

Same prompt, local model. Establishes the capability gap between Claude and local models.

- **Model:** Qwen 2.5 (7B or 70B) via Ollama
- **Tools:** Same as Variant 1 (shell, grep, file read)
- **Learning:** None
- **Orchestration:** Prompt-driven

### Variant 3: Prompt Agent + LoRA (Local)

Same prompt-driven approach, but model is fine-tuned on code review feedback.

- **Model:** Qwen 2.5 + LoRA adapter trained on review feedback
- **Tools:** Yes
- **Learning:** Facts (preferences), feedback collection, LoRA fine-tuning
- **Orchestration:** Prompt-driven

**Training signal:**
- Positive: Reviews accepted without iteration
- Negative: Reviews requiring corrections, false positives flagged by user

### Variant 4: Programmatic Agent + LoRA (Local)

Code orchestrates the workflow; model handles individual steps.

- **Model:** Qwen 2.5 + LoRA adapter
- **Tools:** Yes
- **Learning:** Facts, feedback, LoRA fine-tuning
- **Orchestration:** Code-driven workflow

**Key difference from Variant 3:**
Instead of prompting "review this code", code defines:
1. Get diff
2. For each file: run pattern checks
3. For each potential issue: verify with tools
4. Format and return findings

The model handles subtasks; code handles planning. Reduces reasoning burden on the model.

---

## Metrics

### Primary Metrics

| Metric | Definition | How to Measure |
|--------|------------|----------------|
| **Precision** | Issues flagged that are real | TP / (TP + FP) |
| **Recall** | Real issues that were found | TP / (TP + FN) |
| **Iteration Count** | Reviews until "ready to merge" | Count per PR |
| **F1 Score** | Harmonic mean of precision/recall | 2 * (P * R) / (P + R) |

### Secondary Metrics

| Metric | Definition | How to Measure |
|--------|------------|----------------|
| **Latency** | Time to complete review | Wall clock seconds |
| **User Satisfaction** | Subjective quality rating | 1-5 scale after each review |

---

## Measurement Methodology

### Evaluation Dataset

**Source:** Real PRs from active development, not synthetic examples.

**Dataset composition:**
- Minimum 30 PRs per evaluation round (statistical significance)
- Mix of PR sizes: small (<100 lines), medium (100-500), large (500+)
- Mix of change types: features, bug fixes, refactors
- From repositories with known coding standards

**Candidate repositories:**
- This repo (llm-agent) and related repos (llm-learn, llm-infer)
- Open source projects with good issue/bug tracking (for ground truth validation)

### Ground Truth Collection

Ground truth is established through **expert annotation** after each review:

**Process:**
1. Variant produces review with N findings
2. Expert (human reviewer) evaluates each finding:
   - **True Positive (TP):** Real issue that should be fixed
   - **False Positive (FP):** Not a real issue, noise
   - **Severity Correct:** Did variant assign correct severity?
3. Expert identifies missed issues:
   - **False Negative (FN):** Real issue the variant missed
4. Expert rates overall quality (1-5 scale)

**Ground truth sources for FN detection:**
- Expert's own review of the diff
- Issues found later in production (if tracked)
- Other variant's findings (if validated as TP)

### Evaluation Protocol

**Blind evaluation:**
- Expert evaluates reviews without knowing which variant produced them
- Reviews are anonymized and presented in random order

**Same-PR comparison:**
- Each PR in the dataset is reviewed by ALL variants
- Enables direct comparison on identical inputs

**Evaluation session:**
```
For each PR in dataset:
    1. Run all variants on the PR
    2. Collect reviews (anonymized)
    3. Expert evaluates each review
    4. Record metrics per variant
    5. Expert provides overall ranking (optional)
```

### Issue Matching

To compare variants, need to determine when two findings refer to the "same" issue.

**Matching criteria:**
- Same file AND
- Same line (±5 lines tolerance) AND
- Same issue category (bug, security, style, etc.)

**Example:**
```
Variant 1: "SQL injection in users.py:42"
Variant 2: "Unsanitized input in users.py:44"
→ Matched (same file, nearby lines, both security)
```

### Severity Weighting

Not all issues are equal. Weighted scoring:

| Severity | Weight | Rationale |
|----------|--------|-----------|
| Critical | 4 | Bugs, security issues - must catch |
| Important | 2 | Real problems - should catch |
| Minor | 1 | Quality improvements - nice to catch |
| Nitpick | 0.5 | Observations - optional |

**Weighted precision:** Σ(TP × weight) / Σ(flagged × weight)
**Weighted recall:** Σ(TP × weight) / Σ(actual × weight)

This rewards catching critical issues more than nitpicks.

### PR Lifecycle Logging

The `/pr` skill tracks the complete PR lifecycle with explicit state transitions.

**Log location:** `~/.config/claude-code-pr-skill/prs.jsonl`

**Commands:**

| Command | Action |
|---------|--------|
| `/pr start <base>` | Create PR record, set base branch |
| `/pr review` | Run code review, log as iteration |
| `/pr commit` | Commit changes (after fixing issues) |
| `/pr push` | Push branch, create GitHub PR |
| `/pr merge` | Mark PR complete |

**Log structure:**

```json
{
  "pr_id": "serendip-ml/llm-agent/feature-foo/1706435200",
  "repo": "serendip-ml/llm-agent",
  "branch": "feature/foo",
  "base": "develop",
  "started_at": "2026-01-28T10:00:00Z",
  "events": [
    {"type": "start", "ts": "2026-01-28T10:00:00Z", "commit": "a1b2c3d", "base_commit": "f4e5d6c"},
    {"type": "review", "ts": "2026-01-28T10:30:00Z", "commit": "a1b2c3d", "findings": [...], "latency_s": 4.2},
    {"type": "commit", "ts": "2026-01-28T11:00:00Z", "commit": "b2c3d4e", "message": "Fix null check"},
    {"type": "review", "ts": "2026-01-28T11:15:00Z", "commit": "b2c3d4e", "findings": [...], "latency_s": 3.8},
    {"type": "push", "ts": "2026-01-28T11:30:00Z", "commit": "b2c3d4e", "pr_url": "https://github.com/..."},
    {"type": "merge", "ts": "2026-01-28T14:00:00Z", "commit": "b2c3d4e"}
  ],
  "iterations": 2,
  "outcome": "merged"
}
```

**Event types:**

| Event | Fields | Description |
|-------|--------|-------------|
| `start` | `ts`, `commit`, `base_commit` | PR cycle begins |
| `review` | `ts`, `commit`, `findings`, `latency_s` | Code review iteration |
| `commit` | `ts`, `commit`, `message` | Fix committed |
| `push` | `ts`, `commit`, `pr_url` | Branch pushed, PR created |
| `merge` | `ts`, `commit` | PR merged, cycle complete |

**Finding structure:**

```json
{"file": "src/agent.py", "line": 42, "severity": "critical", "category": "bug", "description": "..."}
```

**Derived metrics:**

| Metric | Derivation |
|--------|------------|
| `iterations` | Count of "review" events |
| `cycle_time` | `merge.ts - start.ts` |
| Implicit TP | Findings that disappear between reviews (fixed) |
| Implicit FP | Findings that persist across all reviews (ignored) |

*Note: Implicit TP/FP are approximations useful for automated analysis. Expert annotation provides
definitive classification, as users may fix disagreed findings or defer valid ones.*

### Evaluation Data Recording

Each review session records (after expert annotation):

```json
{
  "review_id": "uuid",
  "timestamp": "2026-01-28T12:00:00Z",
  "variant": 1,
  "pr": {
    "repo": "llm-agent",
    "branch": "feature/foo",
    "diff_lines": 150,
    "files_changed": 3
  },
  "findings": [
    {
      "id": "f1",
      "file": "src/agent.py",
      "line": 42,
      "severity": "critical",
      "category": "bug",
      "description": "...",
      "evaluation": "TP",
      "severity_correct": true
    }
  ],
  "missed_issues": [
    {
      "file": "src/agent.py",
      "line": 88,
      "severity": "important",
      "category": "security",
      "description": "..."
    }
  ],
  "metrics": {
    "tp": 3,
    "fp": 1,
    "fn": 1,
    "precision": 0.75,
    "recall": 0.75,
    "f1": 0.75,
    "weighted_f1": 0.82
  },
  "latency_ms": 4500,
  "tokens_used": 2300,
  "expert_rating": 4
}
```

### Statistical Significance

- **Minimum sample size:** 30 PRs per comparison
- **Significance test:** Paired t-test or Wilcoxon signed-rank (non-parametric)
- **Significance threshold:** p < 0.05

**Comparison protocol:**
1. Calculate metric (e.g., F1) for each variant on each PR
2. Compute paired differences
3. Test if mean difference is significantly different from 0

**Reporting:**
- Mean ± standard deviation for each metric
- 95% confidence intervals
- p-values for pairwise comparisons

### Iteration Tracking (Longitudinal)

For variants with learning (3, 4), track improvement over time:

**Checkpoints:**
- After 10 reviews (early)
- After 30 reviews (baseline comparison point)
- After 100 reviews (if applicable)
- After each LoRA training round

**Learning curve:** Plot F1 score vs. number of reviews to show improvement trajectory.

---

## Implementation Phases

### Phase 1: Baseline Measurement (Variants 1 & 2)

**Goal:** Establish baseline metrics for Claude and raw local model.

1. Use existing `/review-code` skill (Variant 1)
2. Create equivalent for local model (Variant 2)
3. Run both on same set of PRs
4. Collect metrics, establish the gap

**Deliverables:**
- Local prompt agent that mirrors Claude skill
- Evaluation harness for collecting metrics
- Baseline numbers for both variants

### Phase 2: Learning Integration (Variant 3)

**Goal:** Add learning to local prompt agent, measure improvement.

1. Integrate llm-learn for fact storage
2. Build feedback collection (mark findings as valid/invalid)
3. Implement preference extraction from feedback
4. Set up LoRA fine-tuning pipeline
5. Train on accumulated feedback
6. Measure improvement over Variant 2

**Deliverables:**
- Prompt agent with learning (facts, preferences)
- Feedback collection UI/flow
- LoRA training pipeline
- Metrics showing improvement

### Phase 3: Programmatic Agent (Variant 4)

**Goal:** Replace prompt-driven workflow with code-driven orchestration.

1. Design code-driven review workflow
2. Implement using tool infrastructure (ToolRegistry, ToolExecutor)
3. Integrate learning from Phase 2
4. Compare against Variants 1-3

**Deliverables:**
- Programmatic review agent
- Side-by-side comparison with all variants
- Final metrics and analysis

---

## Architecture

### Local Agent Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code Skill                     │
│                        /pr                               │
│         (start, review, commit, push, merge)             │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP (for variants 2-4)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     Review Agent                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ HTTPTrait   │  │ DirectiveTrait│  │ LearnTrait     │  │
│  │ /v1/review  │  │ (review prompt)│ │ (feedback)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │ Ollama   │   │ llm-learn│   │ Tool Registry│
    │ (Qwen)   │   │ (facts,  │   │ (shell, grep,│
    │          │   │  feedback)│   │  file read)  │
    └──────────┘   └──────────┘   └──────────────┘
```

### PR Lifecycle Flow

```
1. User: /pr start develop     → Creates PR record in log
2. User: /pr review            → Runs review, logs findings
3. User: (fixes issues)
4. User: /pr commit            → Commits fix, logs commit event
5. User: /pr review            → Another review iteration
6. User: /pr push              → Pushes, creates GitHub PR, logs
7. User: /pr merge             → Marks complete, logs final state
```

### Data Flow (Review Step)

```
Variant 1 (Claude):
  /pr review → Claude runs review → Log findings

Variants 2-4 (Local):
  /pr review → POST /v1/review → Agent reviews → Log findings
```

---

## Open Questions

Decisions needed before Phase 2 implementation.

| Question | Options | Decision |
|----------|---------|----------|
| Where does diff gathering happen? | Skill vs Agent | TBD |
| How to handle large diffs? | Truncate, chunk, or summarize | TBD |
| Feedback collection UX | Inline in Claude Code vs separate UI | TBD |
| LoRA training frequency | Per-session, daily, manual | TBD |
| Model size for experiments | 7B, 14B, 32B, 70B | Start with 7B |

---

## Success Criteria

The experiment succeeds if:

1. **Variant 3 < Variant 1** on F1 score, but within 20% (fine-tuning helps)
2. **Variant 4 ≈ Variant 1** on F1 score, within 10% (orchestration closes the gap)

Stretch goal: Variant 4 achieves higher precision (fewer false positives) than Variant 1.
