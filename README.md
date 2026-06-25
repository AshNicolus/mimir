# Mimir

[![PyPI](https://img.shields.io/pypi/v/mimir-learn.svg)](https://pypi.org/project/mimir-learn/)
[![Python](https://img.shields.io/pypi/pyversions/mimir-learn.svg)](https://pypi.org/project/mimir-learn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Experience-driven memory for autonomous agents.** Mimir helps agents learn from their past successes and failures instead of starting from scratch on every task.

> Named after Mímir, the keeper of wisdom in Norse mythology.

---

## The problem

Today's agents have memory, but they don't really *learn*.

Most frameworks store one of two things:

- **Conversation history** (LangGraph memory, buffer memory)
- **Vector embeddings of documents** (RAG, AGENTS.md, CLAUDE.md)

Both let an agent **remember information**. Neither lets it **remember experience**.

```
Task:     Fix authentication latency
Action:   Added Redis cache
Outcome:  Success
```

A month later, the agent has no meaningful understanding that this strategy worked. It solves the same class of problem from zero, every time.

## The idea

Instead of storing documents, embeddings, and metadata, Mimir stores **experiences**:

```
Problem  →  Action  →  Outcome  →  Confidence  →  Context  →  Time
```

From a stream of experiences, Mimir reflects, extracts reusable strategies, and recommends actions for new tasks, so the agent gets measurably better over time.

```python
from mimir import Mimir

memory = Mimir()

# Record what happened
memory.record(
    task="Fix authentication timeout",
    action="Implemented Redis caching",
    outcome="success",
    score=0.95,
)

# Recall relevant past experience
past = memory.recall("authentication latency")

# Get a recommended strategy with confidence
strategy = memory.recommend("login timeout")
# → Strategy: "Redis caching"  |  confidence: 0.87  |  based on 23 successes / 2 failures
```

## How it differs from AGENTS.md / CLAUDE.md

| | AGENTS.md / CLAUDE.md | Mimir |
|---|---|---|
| Knowledge type | Static, hand-written rules | Dynamic, learned from outcomes |
| Updates | Manually edited | Updates itself from results |
| Example | "Use FastAPI. Use PostgreSQL." | "Redis caching solved auth latency 23/25 times (92%)." |
| Failures | Not tracked | First-class: agents stop repeating mistakes |

AGENTS.md answers *"What should the agent remember?"*
Mimir answers *"How does an agent accumulate experience and become wiser over time?"*

## Design principles

Mimir is built as a **modular monolith Python library**, not a microservice swarm or managed cloud product. The library is the product.

- **No LLM and no web server required for v1.** Storage, retrieval, and ranking come first. Reflection via an LLM is added later, behind an interface.
- **Pluggable seams.** Storage, embeddings, and the write path are interfaces, so scaling up (SQLite → Postgres → async reflection → Redis cache) is a swap, never a rewrite.
- **Derived knowledge is rebuildable.** Strategies and reflections are computed from raw experiences and can always be regenerated.
- **Failures are first-class.** Learning from what *didn't* work is treated as importantly as what did.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Public API   Mimir()  .record() .recall() .recommend()     │
├────────────────────────────────────────────────────────────┤
│  Write chokepoint   ──►  [validation / provenance hook]      │   single write path
├──────────────┬───────────────┬─────────────────────────────┤
│  Episodic    │  Reflection   │  Recommendation             │
│  Engine      │  Engine       │  Engine                     │
│  (record/    │  (reflect/    │  (recommend / rank /         │
│   recall)    │   extract)    │   confidence)               │
├──────────────┴───────────────┴─────────────────────────────┤
│  Retrieval layer        (keyword + optional vector hybrid)   │
├────────────────────────────────────────────────────────────┤
│  Storage interface      SQLite (v1) · Postgres (v2) · …      │   pluggable
├────────────────────────────────────────────────────────────┤
│  Embedding provider     none (default) · local · API        │   pluggable
└────────────────────────────────────────────────────────────┘
```

### Data model

```
Experience
  id, task, action, outcome (success|failure|partial),
  score (0..1), context (json), embedding (nullable),
  created_at, superseded_by (nullable)

Strategy   (derived)  problem_pattern, recommended_action, confidence,
                      success_count, failure_count, source_experience_ids
Reflection (derived)  summary, pattern, supporting_experience_ids, created_at
```

## Installation

```bash
pip install mimir-learn
```

The distribution is named `mimir-learn` on PyPI, but you import it as `mimir`:

```python
from mimir import Mimir
```

For development:

```bash
git clone https://github.com/AshNicolus/mimir.git
cd mimir
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, tested on 3.10, 3.11, and 3.12 (Linux, macOS, Windows). This matters for agents, which often run on the Python version their host ships, and 3.10 is still the default on several current Linux distributions. v1 has no required external services: storage is a local SQLite file. Semantic search and reflection are optional extras.

## Quick start

```python
from mimir import Mimir

memory = Mimir(db_path="mimir.db")

memory.record(
    task="Fix login latency",
    action="Added Redis cache in front of session lookups",
    outcome="success",
    score=0.9,
    context={"service": "auth", "language": "python"},
)

memory.record_failure(
    task="Throttle abusive clients",
    action="Added a fixed-window rate limiter",
    reason="WebSocket traffic wasn't handled; limiter only saw HTTP",
)

for exp in memory.recall("authentication is slow", k=5):
    print(exp.action, exp.outcome, exp.score)

print(memory.recommend("login times out under load"))
```

## Recommendations

`recommend()` aggregates past experiences for a task and returns the action with
the strongest track record, ranked by a relevance-weighted Wilson lower bound. An
action proven on closely matching tasks outranks an equally successful one proven
on loosely related tasks. Reported counts always cover the full matching
population; weighting only affects ranking, and you can turn it off to compare:

```python
memory.recommend("login times out under load", weight_by_relevance=False)
```

By default actions are grouped by normalized text, so "Added Redis cache" and
"use redis caching" count as separate strategies. Plug in a clusterer to merge
equivalent phrasings and pool their evidence:

```python
from mimir import Mimir, EmbeddingClusterer

memory = Mimir(clusterer=EmbeddingClusterer(my_embedder))
```

`ExactClusterer` is the default and needs no embeddings. Any other strategy can
implement `ActionClusterer`.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| **1: Episodic memory** | `record()` / `recall()`, outcome tracking, SQLite backend | ✅ Done |
| **2: Failure memory** | `record_failure()`, failures queried separately | ✅ Done |
| **3: Reflection engine** | `reflect()`: cluster experiences, synthesize patterns (LLM) | Planned |
| **4: Strategy extraction** | Turn experiences into reusable strategies with confidence | Planned |
| **5: Recommendation engine** | `recommend()`: rank strategies for a new task | 🛠️ Relevance-weighted aggregation, pluggable action clustering (non-LLM) |
| **6: Shared org memory** | Multiple agents learn from a shared store | Future |
| **Runtime support** | Run on the Python versions agent hosts actually ship, across Linux, macOS, and Windows |  Python 3.10–3.12 |

## Scaling path

Mimir starts as a single SQLite file and grows by swapping seams, no rewrites:

1. **v1**: SQLite, in-process, single agent.
2. **v2**: Postgres + pgvector backend for concurrent multi-agent writes.
3. **v3**: extract the (slow, batch) reflection engine into an async worker.
4. **v4**: Redis cache for hot/recent experiences on the read path.

## Status

Alpha (`0.1.2`): **published on PyPI** as [`mimir-learn`](https://pypi.org/project/mimir-learn/). Phase 1 (episodic + failure memory) is complete and tested; the recommendation engine works today via relevance-weighted outcome aggregation with pluggable action clustering (no LLM). APIs may still change before `1.0`. Feedback and ideas welcome.

## License

MIT
