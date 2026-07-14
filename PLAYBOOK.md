# Mimir Playbook

This is the hands-on guide to Mimir. It starts from zero and builds up to real
agent integrations, so a first-time reader can follow along and someone shipping
a production agent can find the advanced patterns. For the project overview and
design rationale, see the [README](README.md).

Mimir gives an agent memory of its own experience. It stores what was tried, on
which task, and how it turned out, then recalls and ranks that experience when a
new task comes in. It is a plain Python library over a local SQLite file. There
is no server and no LLM to run, and the only required dependency is Pydantic.

## Contents

1. [What Mimir is](#what-mimir-is)
2. [Install](#install)
3. [Your first five minutes](#your-first-five-minutes)
4. [The mental model](#the-mental-model)
5. [Recording experiences](#recording-experiences)
6. [Recalling experiences](#recalling-experiences)
7. [Recommending an action](#recommending-an-action)
8. [A worked example: an agent that improves](#a-worked-example-an-agent-that-improves)
9. [Keeping memory fresh](#keeping-memory-fresh)
10. [Semantic recall with embeddings](#semantic-recall-with-embeddings)
11. [Grouping equivalent actions](#grouping-equivalent-actions)
12. [Persistence, concurrency, and lifecycle](#persistence-concurrency-and-lifecycle)
13. [Integrations](#integrations)
14. [Distilling conversations into experiences](#distilling-conversations-into-experiences)
15. [Patterns and best practices](#patterns-and-best-practices)
16. [Testing your integration](#testing-your-integration)
17. [Troubleshooting and FAQ](#troubleshooting-and-faq)
18. [Extending Mimir](#extending-mimir)
19. [API reference](#api-reference)

## What Mimir is

Most agent memory stores one of two things. It keeps conversation history, or it
keeps vector embeddings of documents. Both help an agent remember information.
Neither helps it remember what worked.

Mimir stores experience instead:

```
task  ->  action  ->  outcome  ->  score  ->  context  ->  time
```

An agent records what it tried and how it went. Later, faced with a similar task,
it can ask Mimir what has worked before and what to avoid. Over many runs the
actions that keep succeeding rise to the top, and the ones that fail get
remembered as mistakes.

You will use three methods most of the time:

- `record(...)` stores what happened.
- `recall(...)` finds relevant past experiences.
- `recommend(...)` returns the action with the best track record for a task.

A good way to keep them straight: `recall` answers "what have I seen like this
before?" and `recommend` answers "what should I do about it?".

## Install

```bash
pip install mimir-learn
```

The package is named `mimir-learn` on PyPI, but you import it as `mimir`:

```python
from mimir import Mimir
```

Keyword recall and recommendations work with no extra dependencies. Two optional
extras add semantic search, which you can skip until you need it:

```bash
pip install "mimir-learn[embeddings]"  # local embeddings (sentence-transformers, numpy)
pip install "mimir-learn[vector]"      # sqlite-vec index for fast vector search
```

Mimir needs Python 3.10 or newer and runs on Linux, macOS, and Windows.

## Your first five minutes

Here is a complete program. Save it as `first.py` and run it.

```python
from mimir import Mimir

# A file-backed store. Use ":memory:" for a throwaway one.
memory = Mimir(db_path="mimir.db")

# Record a couple of experiences.
memory.record(
    task="Fix login latency under load",
    action="Added a Redis cache in front of session lookups",
    outcome="success",
    score=0.9,
)
memory.record_failure(
    task="Throttle abusive clients",
    action="Added a fixed-window rate limiter",
    reason="WebSocket traffic wasn't handled; the limiter only saw HTTP",
)

# Recall the most relevant experiences for a related task.
for exp in memory.recall("login latency", k=5):
    print(exp.outcome.value, "|", exp.action)

# Ask what to do.
rec = memory.recommend("login times out under load")
print(rec)

memory.close()
```

Running it prints something like:

```
success | Added a Redis cache in front of session lookups
Recommended strategy: 'Added a Redis cache in front of session lookups'
  confidence: 0.23
  based on 1 experiences (1 success / 0 failure)
```

Confidence is only 0.23 here because a single success is thin evidence; it climbs
as results accumulate. Note that recall matched on the shared word "login". To
match a paraphrase with no shared words, add an embedder (see
[Semantic recall](#semantic-recall-with-embeddings)).

That is the whole loop in miniature: record, recall, recommend. Everything else
in this guide builds on those three calls.

`Mimir` is also a context manager, so you can let it close itself:

```python
with Mimir(db_path="mimir.db") as memory:
    memory.record("task", "action", outcome="success")
```

## The mental model

Everything Mimir stores is an `Experience`. It has these fields:

- `task`: the problem you were solving, in plain words.
- `action`: what you actually did.
- `outcome`: one of `success`, `failure`, or `partial`.
- `score`: a number from 0 to 1 for how well it went.
- `context`: a free-form dict for tags like service, environment, or agent id.
- `created_at`: set for you, in UTC.
- `id`: a unique id you can use to fetch or supersede the experience later.

You create experiences with `record`, read them back with `recall`, `get`, and
`recent`, and turn them into advice with `recommend`. When you are done, call
`close()` or use a `with` block.

## Recording experiences

### Outcomes and scores

Every experience has an outcome and a score. If you leave the score out, Mimir
fills it in from the outcome: success becomes 1.0, partial becomes 0.5, and
failure becomes 0.0.

```python
memory.record("resize images", "used Pillow", outcome="success")             # score 1.0
memory.record("resize images", "shelled out to ImageMagick", outcome="partial", score=0.6)
```

Outcome accepts a plain string or the `Outcome` enum, whichever you prefer:

```python
from mimir import Outcome
memory.record("task", "action", outcome=Outcome.SUCCESS)
```

If a score clearly contradicts its outcome, such as a failure scored 0.9, the
record still lands but Mimir raises an `OutcomeScoreWarning` so bad data is easy
to notice. To make it an error instead:

```python
import warnings
from mimir import OutcomeScoreWarning

warnings.simplefilter("error", OutcomeScoreWarning)
```

### Recording failures

`record_failure` is a shortcut for an outcome of failure. It also saves the
reason under `context["failure_reason"]`, which is exactly what an agent wants to
read back when deciding what not to repeat:

```python
memory.record_failure(
    task="Speed up the report",
    action="Added an index on created_at",
    reason="The slow query filtered on status, not created_at",
)
```

### Context

`context` is a free-form dict for anything you want to filter on later: tags,
environment, team, region, agent id, language. It has to be JSON-serializable,
otherwise `record` raises a `ValueError` with a clear message.

```python
memory.record(
    "deploy the service", "blue-green rollout", outcome="success",
    context={"env": "prod", "team": "platform", "region": "us-east"},
)
```

`record` returns the stored `Experience`. Keep its `id` if you want to look it up
or supersede it later:

```python
exp = memory.record("task", "action")
memory.get(exp.id)      # fetch it back
memory.delete(exp.id)   # remove it
```

## Recalling experiences

`recall(query, k=5)` returns up to `k` experiences most relevant to the query,
best first.

```python
hits = memory.recall("database is slow", k=5)
for exp in hits:
    print(exp.task, "->", exp.action, f"({exp.outcome.value})")
```

### Filter by outcome

Ask for successes only, or study just the failures:

```python
wins = memory.recall("rate limiting", outcome="success")
mistakes = memory.recall("rate limiting", outcome="failure")
```

### Filter by context

Pass a `context` dict and Mimir returns only experiences whose context matches
every key and value you give. Scalar values are matched inside SQLite; nested
values like lists are matched in Python, so both work:

```python
memory.recall("speed things up", context={"service": "auth"})
memory.recall("speed things up", context={"tags": ["auth", "cache"]})
```

### How matching works

With no embedder configured, recall is keyword search. It uses SQLite FTS5 when
your build has it and falls back to a portable token-overlap scorer when it does
not. Common stopwords are ignored, so "what is the latency" still matches on
"latency". Matching is on the task and action text together.

Keyword recall is fast and needs nothing extra, but it only finds shared words.
To match by meaning, add an embedder (see
[Semantic recall](#semantic-recall-with-embeddings)).

## Recommending an action

`recommend(task)` gathers every past experience that matches the task, groups
them by action, and returns the action with the best track record. It returns
`None` when there is nothing relevant to go on.

```python
rec = memory.recommend("login times out under load")
if rec:
    print(rec.recommended_action)
    print(rec.confidence)          # 0 to 1, read as a success rate
    print(rec.success_count, rec.failure_count, rec.partial_count)
    print(rec.based_on)            # how many experiences were considered
    print(rec.supporting_ids)      # ids of the experiences behind it
```

`str(rec)` prints a readable summary:

```
Recommended strategy: 'add a redis cache'
  confidence: 0.79
  based on 8 experiences (8 success / 0 failure)
```

### What confidence means

Confidence is the lower bound of a Beta posterior on the action's success rate,
computed from its raw counts, with a Jeffreys prior so small samples stay honest.
Read it as a conservative success rate. Some intuition:

- 1 success out of 1 gives about 0.23. One data point is barely evidence.
- 8 out of 8 gives about 0.79. A clean track record earns real confidence.
- 90 out of 100 gives about 0.83. More consistent evidence tightens the estimate.
- An action that has only ever failed is never recommended.

The number means the same thing no matter how the query was phrased, so you can
compare it across tasks and threshold on it (for example, only auto-apply an
action above 0.7).

### Relevance weighting

When more than one action could apply, Mimir has to choose. By default it favors
actions whose supporting experiences closely match your task, so an action proven
on very similar tasks outranks an equally confident one proven on loosely related
tasks. This affects only the ranking, not the reported confidence.

You can turn it off to rank purely on track record:

```python
memory.recommend("login times out under load", weight_by_relevance=False)
```

A concrete case. Suppose one action has 6 clean successes on a loosely related
task, and another has 3 clean successes on a task almost identical to your query.
With weighting on, Mimir recommends the closely matching one. With it off, it
recommends the more proven one. In both cases the confidence it reports is that
action's own honest success rate.

Set the instance default with `Mimir(weight_by_relevance=False)`.

### Exploring instead of exploiting

The default mode always returns the best-supported action, which means an
almost-as-good newcomer never gets a chance to prove itself. Pass `explore=True`
and the winner is drawn by Thompson sampling instead: each action's rank is a
random draw from its posterior, so a less-proven action wins a share of calls
proportional to how likely it is to actually be better.

```python
action = memory.recommend("api latency high", explore=True).recommended_action
```

Use it when the agent will record the outcome afterwards, because that is what
makes exploration pay: trying the newcomer either builds its track record or
rules it out. Three guarantees hold either way: actions that have only ever
failed are never recommended, the reported confidence is always the honest
bound rather than the random draw, and leaving `explore` off keeps `recommend`
fully deterministic. Pass `rng=random.Random(seed)` to make draws reproducible
in tests.

A simple policy that works well: explore while evidence is thin, exploit once
it is strong.

```python
rec = memory.recommend(task)
if rec is None or rec.confidence < 0.6:
    rec = memory.recommend(task, explore=True)
```

## A worked example: an agent that improves

This runnable script shows the whole point of Mimir: consult memory, act, record
the result, and get better over time. It simulates a few runs against the same
kind of task.

```python
from mimir import Mimir

memory = Mimir(db_path=":memory:")

# Pretend these came from real runs over the past weeks.
for _ in range(6):
    memory.record("api latency is high under load", "add a redis cache", outcome="success")
memory.record("api latency is high under load", "raise the thread count", outcome="failure")

def handle(task: str) -> str:
    rec = memory.recommend(task)
    if rec and rec.confidence > 0.6:
        print(f"[reuse] {rec.recommended_action} (confidence {rec.confidence:.2f})")
        return rec.recommended_action
    # Not confident yet: explore, and steer around known failures.
    for past in memory.recall(task, outcome="failure"):
        print(f"[avoid] {past.action}")
    print("[explore] profiling the endpoint first")
    return "profile the slow endpoint first"

# A new, related task arrives.
action = handle("api latency high under load")

# Whatever the agent does, record how it went so the next run is wiser.
memory.record("api latency high under load", action, outcome="success")
print("stored", memory.count(), "experiences")
memory.close()
```

This prints:

```
[reuse] add a redis cache (confidence 0.74)
stored 8 experiences
```

Because the cache fix has a solid track record on matching tasks, `recommend`
returns it with high confidence and the agent reuses it instead of solving from
scratch. If the record were thin or the confidence low, the `handle` function
would fall to the explore branch and print the actions it has seen fail, so it
does not repeat them.

## Keeping memory fresh

Old knowledge goes stale. Mimir gives you two tools, and they work together.

### Superseding

Mark an old experience as replaced. It drops out of recall and recommendation by
default but stays retrievable by id, so nothing is lost.

```python
old = memory.record("auth is slow", "add a write-through cache", outcome="failure")
new = memory.record("auth is slow", "add a read cache", supersedes=old.id)

# or link two that already exist
memory.supersede(old.id, new.id)
```

Pass `include_superseded=True` to `recall` or `recommend` to see superseded rows
anyway, which helps when you want to study how a strategy changed over time.

### Time decay

For gradual staleness rather than a hard cutoff, set a half-life. Evidence then
loses half its weight every `half_life_days`, so recent results count for more.
This is off by default.

```python
memory = Mimir(half_life_days=30)   # evidence from 30 days ago counts for half
```

The half-life influences ranking in both engines. In `recommend` it discounts
older evidence, so an action that has been winning lately can overtake one that
was proven long ago. In `recall` it reweights by age, so a fresh experience
outranks an equally relevant but older one. The reported counts and confidence
stay based on the full record; decay only changes the ordering.

## Semantic recall with embeddings

Keyword recall only finds shared words. "kitten care" will not match "adopt a
feline companion" because they share no tokens. Give Mimir an embedder and recall
becomes hybrid: keyword results and vector results are combined with
reciprocal-rank fusion, so an experience can be found by meaning as well as by
words.

An embedder is any object with an `embed(text) -> list[float]` method. Here is one
backed by a local sentence-transformers model (from the `embeddings` extra):

```python
from mimir import Mimir
from mimir.embeddings import Embedder
from sentence_transformers import SentenceTransformer

class LocalEmbedder(Embedder):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

memory = Mimir(db_path="mimir.db", embedder=LocalEmbedder())
memory.record("adopt a feline companion", "visit the shelter", outcome="success")

# Matches by meaning, with no shared words:
print(memory.recall("kitten care tips"))
```

You can wrap an API embedder the same way. Only the `embed` method matters:

```python
from anthropic import Anthropic          # example provider
from mimir.embeddings import Embedder

class APIEmbedder(Embedder):
    def __init__(self):
        self.client = SomeEmbeddingClient()

    def embed(self, text: str) -> list[float]:
        return self.client.create_embedding(text)   # return a list of floats
```

### The vector index

With the `vector` extra installed, vector search runs on a
[sqlite-vec](https://github.com/asg017/sqlite-vec) index for speed. Without it,
Mimir falls back to a plain Python cosine scan, which is faster when numpy is
present. The results are identical either way, so you can develop without the
extra and add it later when the store grows.

### The query embedding cache

Every `recall` embeds the query text, and embedding is the expensive step with a
real model or an API. Agents tend to repeat queries during retries and loops, so
Mimir keeps a small per-instance cache from query text to vector. A repeated query
then costs nothing to embed. Resize or disable it:

```python
memory = Mimir(embedder=LocalEmbedder(), query_cache_size=512)  # default 256, 0 disables
```

## Grouping equivalent actions

`recommend` groups experiences by action so a strategy accumulates evidence. By
default it groups by normalized text, so "Added Redis cache" and "add redis
cache" merge into one strategy, but "use redis caching" stays separate.

To pool actions that mean the same thing but read differently, plug in a
clusterer:

```python
from mimir import Mimir, EmbeddingClusterer

memory = Mimir(clusterer=EmbeddingClusterer(my_embedder))
```

`ExactClusterer` is the default and needs no embeddings. `EmbeddingClusterer`
merges an action into the nearest existing cluster within a cosine threshold
(0.85 by default). You can write your own by implementing the `ActionClusterer`
interface.

## Persistence, concurrency, and lifecycle

- File or memory. `Mimir(db_path="mimir.db")` saves to disk and survives
  restarts. `Mimir(":memory:")` is ephemeral, which is ideal for tests. Missing
  parent directories are created for you.
- Concurrency. A file-backed store runs in SQLite WAL mode. Each reader thread
  gets its own connection so reads scale across threads, while writes are
  serialized. Reading and writing from several threads is safe.
- Schema. The database is versioned with a forward-only migration runner, so
  opening a file written by an older version upgrades it in place.
- Lifecycle. Call `close()` when done, or use a `with` block. `count()` returns
  how many experiences are stored, and `recent(n)` returns the newest.

```python
with Mimir("mimir.db") as memory:
    ...
    print(memory.count())
    print(memory.recent(5))
```

## Integrations

The rest of the guide shows how to wire Mimir into a real agent. Every pattern is
the same underneath: consult memory before acting, and record the outcome after.
The framework only changes where those two calls sit.

### A plain agent loop

Start here even if you use a framework. This is framework-free and shows the
whole cycle.

```python
from mimir import Mimir

memory = Mimir("agent-memory.db", half_life_days=60)

def run_task(task: str):
    # 1. Consult experience.
    rec = memory.recommend(task)
    if rec and rec.confidence > 0.7:
        action = rec.recommended_action            # reuse a proven strategy
    else:
        action = plan_from_scratch(task)           # explore when unsure
        for past in memory.recall(task, outcome="failure"):
            note_to_avoid(past.action)             # do not repeat known mistakes

    # 2. Act.
    result = execute(action)

    # 3. Record the outcome so the next run knows more.
    memory.record(
        task=task,
        action=action,
        outcome="success" if result.ok else "failure",
        score=result.quality,
        context={"agent": "worker-1", "env": "prod"},
    )
    return result
```

### With an LLM (Claude or OpenAI)

When an LLM agent picks an action or a tool, that decision has an outcome you can
learn from. Feed recalled experience into the prompt so the model sees what has
worked, then record how its choice turned out. This example uses the Anthropic
SDK, but the shape is the same for any provider.

```python
from anthropic import Anthropic
from mimir import Mimir

client = Anthropic()
memory = Mimir("agent-memory.db")

def solve(task: str) -> str:
    # Pull in prior experience as grounding for the model.
    rec = memory.recommend(task)
    hint = ""
    if rec:
        hint = (
            f"\n\nPast experience: '{rec.recommended_action}' has worked "
            f"{rec.success_count}/{rec.based_on} times for tasks like this "
            f"(confidence {rec.confidence:.2f}). Prefer it unless you have a reason not to."
        )
    failures = memory.recall(task, outcome="failure", k=3)
    if failures:
        avoid = "; ".join(f.action for f in failures)
        hint += f"\nKnown failures to avoid: {avoid}."

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Task: {task}{hint}\n\nWhat should we do?"}],
    )
    action = message.content[0].text
    return action

def solve_and_learn(task: str):
    action = solve(task)
    result = carry_out(action)               # your execution + evaluation
    memory.record(
        task=task,
        action=action,
        outcome="success" if result.ok else "failure",
        score=result.score,
        context={"model": "claude-sonnet-5"},
    )
    return result
```

The pattern is: `recommend` and `recall` shape the prompt, and `record` closes
the loop after the model's choice plays out. Over time the agent stops rediscovering
the same fixes and stops repeating the same mistakes, without any fine-tuning.

### With LangChain or LangGraph

In a graph-based agent, memory fits naturally as two touchpoints: a node that
injects experience before the model runs, and a hook that records the outcome
after a tool runs. The Mimir calls below are exact; adapt the surrounding
framework glue to your version.

```python
from mimir import Mimir

memory = Mimir("agent-memory.db", half_life_days=90)

# LangGraph-style node: enrich the state with recalled experience.
def recall_node(state: dict) -> dict:
    task = state["task"]
    rec = memory.recommend(task)
    state["experience"] = str(rec) if rec else "no prior experience"
    state["known_failures"] = [e.action for e in memory.recall(task, outcome="failure", k=3)]
    return state

# After a tool runs, record how it went.
def record_outcome(task: str, tool_name: str, ok: bool):
    memory.record(
        task=task,
        action=f"used tool: {tool_name}",
        outcome="success" if ok else "failure",
        context={"tool": tool_name},
    )
```

If you use LangChain tools, wrap the tool's `run` so every call records its
result:

```python
def remembering_tool(task: str, tool):
    def wrapped(*args, **kwargs):
        try:
            out = tool.run(*args, **kwargs)
            memory.record(task, f"used tool: {tool.name}", outcome="success")
            return out
        except Exception as exc:
            memory.record_failure(task, f"used tool: {tool.name}", reason=str(exc))
            raise
    return wrapped
```

You now have a memory that spans sessions and even different agents, since it is
just a shared SQLite file.

### Semantic recall end to end

A complete, runnable example that adds meaning-based recall with a local model.
Install the extra first with `pip install "mimir-learn[embeddings]"`.

```python
from mimir import Mimir
from mimir.embeddings import Embedder
from sentence_transformers import SentenceTransformer

class LocalEmbedder(Embedder):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

memory = Mimir(db_path="semantic.db", embedder=LocalEmbedder())

memory.record("reduce cloud spend", "moved batch jobs to spot instances", outcome="success")
memory.record("cut the AWS bill", "rightsized the database instances", outcome="success")

# "lower infrastructure costs" shares no words with either task, but matches by meaning.
for exp in memory.recall("lower infrastructure costs", k=2):
    print(exp.action)

memory.close()
```

## Distilling conversations into experiences

Everything so far assumed your loop knows the task and action as strings. Often
what you actually have at the end of a run is a transcript. `record_conversation`
bridges the two: give it the messages and a distiller, and Mimir stores one
experience distilled from the conversation.

```python
from mimir import CallableDistiller, Draft, Mimir

memory = Mimir("agent-memory.db")

def summarize(messages) -> Draft | None:
    ...  # summarize the transcript; return None when no clean task emerges

memory.record_conversation(
    messages,                                  # the transcript you already have
    distiller=CallableDistiller(summarize),
    outcome="success",                         # ground truth: tests, exit code, user acceptance
)
```

Three rules keep distilled memory trustworthy:

- **Ground truth wins.** An `outcome` or `score` you pass overrides whatever the
  distiller inferred. If neither you nor the distiller knows the outcome,
  `record_conversation` raises rather than assume success, because a wrong
  outcome label is the one thing the confidence engine cannot recover from.
- **Abstention is fine.** A distiller that returns `None` stores nothing. A
  memory with gaps stays trustworthy; one padded with bad extractions does not.
- **Every distilled row carries provenance.** The context gains
  `source="transcript"` and the distiller's name, and the experience id is
  derived from the transcript, so ingesting the same conversation twice replaces
  the row instead of duplicating it, and you can audit or supersede everything a
  given distiller wrote.

### An LLM distiller

Any `messages -> Draft | None` function works. Here is one backed by the
Anthropic SDK; the shape is the same for any provider.

```python
import json
from anthropic import Anthropic
from mimir import CallableDistiller, Draft

client = Anthropic()

PROMPT = (
    "This transcript shows an agent completing one task. Reply with JSON: "
    '{"task": the problem phrased generally, "action": what was done, one '
    "sentence, no transcript quotes}. Reply with null if no clear completed task."
)

def distill(messages):
    reply = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[{"role": "user", "content": f"{PROMPT}\n\n{json.dumps(messages)}"}],
    )
    data = json.loads(reply.content[0].text)
    return Draft(**data) if data else None

distiller = CallableDistiller(distill, name="claude-sonnet-5")
```

Two constraints in that prompt earn their place. The action must be a
generalizable summary rather than quoted transcript text, which keeps secrets
out of the store and lets equivalent actions cluster. And the task should be
phrased generally, so future recall matches the whole family of similar tasks,
not just this exact instance.

Let the distiller fill in the outcome only when you have no better signal; a
test result or an exit code is always worth more than the model's own reading
of how the conversation went.

## Patterns and best practices

- Close the loop every time. The value compounds only if you record outcomes,
  including failures. A recall-only setup is just search.
- Write tasks the way you will query them. Mimir matches task and action text, so
  consistent, descriptive phrasing ("api latency high under load") recalls better
  than terse labels ("bug123").
- Record failures generously, with reasons. Failure memory is what stops an agent
  repeating mistakes, and the reason is what the next run reads.
- Use confidence as a gate. A simple policy works well: reuse the recommendation
  when confidence is high, explore when it is low. Pick the threshold to match how
  costly a wrong action is.
- Keep weighting on unless you have a reason not to. It surfaces the most relevant
  experience. Turn it off when you want the single most proven action regardless
  of how closely it matches.
- Set a half-life for domains that drift. If what works changes over months, a
  half-life keeps recommendations current without manual cleanup.
- Put stable facts in `context`. Service, environment, language, and agent id in
  context let you slice recall precisely later.

## Testing your integration

Use an in-memory store so tests are fast and isolated, and assert on outcomes
rather than internal state.

```python
from mimir import Mimir

def test_agent_reuses_a_proven_fix():
    memory = Mimir(":memory:")
    for _ in range(5):
        memory.record("db is slow", "add an index", outcome="success")

    rec = memory.recommend("the database is slow")
    assert rec.recommended_action == "add an index"
    assert rec.confidence > 0.5
    memory.close()
```

Because `":memory:"` stores are independent, each test starts from a clean slate
with no files to clean up.

## Troubleshooting and FAQ

**`recommend` returns `None`.** Nothing in memory matched the task text. Check
that stored tasks and your query share words, or add an embedder for meaning-based
matching.

**Confidence looks low for a single success.** That is expected. One data point is
weak evidence, so a 1/1 action sits around 0.23. Confidence climbs as consistent
results accumulate.

**A less proven action gets recommended.** With relevance weighting on (the
default), a closely matching action can outrank a more proven but loosely related
one. Pass `weight_by_relevance=False` to rank on track record alone.

**Recall finds nothing for a paraphrase.** Keyword recall only matches shared
words. Add an embedder for semantic recall.

**Do I need the extras?** No. Keyword recall and recommendations work with only
Pydantic installed. Add `embeddings` for meaning-based recall and `vector` for a
faster vector index.

**Is it safe from multiple threads?** Yes. A file-backed store uses WAL mode:
reads scale across threads and writes are serialized.

**Where is my data?** In the SQLite file at `db_path`. Delete the file to reset.

## Extending Mimir

Mimir is built around three swappable parts, so you can grow it without rewriting
the rest. Pass any of them to the constructor.

- `Storage` decides where experiences live. `SQLiteStorage` ships by default.
  Implement the `Storage` interface for another backend, such as Postgres for
  concurrent multi-agent writes.
- `Embedder` decides how text becomes a vector. `NullEmbedder` (keyword only) is
  the default. Implement `embed(text)` for local or API embeddings.
- `ActionClusterer` decides how equivalent actions are grouped. `ExactClusterer`
  is the default, and `EmbeddingClusterer` or your own class can merge synonyms.
- `Distiller` decides how a transcript becomes a draft experience. It is passed
  per call to `record_conversation` rather than to the constructor.

```python
memory = Mimir(storage=MyStorage(), embedder=MyEmbedder(), clusterer=MyClusterer())
```

## API reference

### `Mimir(...)`

| Parameter | Default | Purpose |
|---|---|---|
| `db_path` | `"mimir.db"` | SQLite path, or `":memory:"` for ephemeral. |
| `storage` | `None` | Custom `Storage`; defaults to `SQLiteStorage(db_path)`. |
| `embedder` | `None` | Custom `Embedder`; defaults to `NullEmbedder` (keyword only). |
| `clusterer` | `None` | Custom `ActionClusterer`; defaults to `ExactClusterer`. |
| `weight_by_relevance` | `True` | Let relevance and recency steer ranking. |
| `half_life_days` | `None` | Evidence and recall half-life; `None` turns decay off. |
| `query_cache_size` | `256` | Query-embedding cache size; `0` disables it. |

### Methods

| Method | Returns | Purpose |
|---|---|---|
| `record(task, action, outcome="success", score=None, context=None, supersedes=None)` | `Experience` | Store an experience. |
| `record_failure(task, action, reason=None, score=0.0, context=None, supersedes=None)` | `Experience` | Store a failure with a reason. |
| `record_conversation(messages, *, distiller, outcome=None, score=None, context=None)` | `Experience` or `None` | Distill a transcript into one experience. |
| `recall(query, k=5, outcome=None, context=None, include_superseded=False)` | `list[Experience]` | Most relevant past experiences. |
| `recommend(task, *, weight_by_relevance=None, include_superseded=False, explore=False, rng=None)` | `Recommendation` or `None` | Best-supported action for a task; `explore=True` draws the winner by Thompson sampling. |
| `supersede(old_id, new_id)` | `bool` | Mark an experience as replaced. |
| `get(id)` | `Experience` or `None` | Fetch one experience by id. |
| `delete(id)` | `bool` | Remove one experience. |
| `recent(n=10)` | `list[Experience]` | The `n` newest experiences. |
| `count()` | `int` | Total stored experiences. |
| `write(exp)` | `Experience` | Low-level single write path. |
| `close()` | `None` | Release resources, or use `with`. |

### `Experience`

`id`, `task`, `action`, `outcome` (an `Outcome`), `score` (0 to 1), `context`
(dict), `embedding` (nullable), `created_at` (UTC), and `superseded_by`
(nullable). `text()` returns the task and action joined, which is what Mimir
indexes and embeds.

### `Recommendation`

`task`, `recommended_action`, `confidence` (0 to 1, read as a success rate),
`success_count`, `failure_count`, `partial_count`, `based_on`, and
`supporting_ids`. The `.total` property sums the counts, and `str(rec)` prints a
readable summary.

### `Distiller` and `Draft`

A `Distiller` implements `distill(messages) -> Draft | None`, where `None`
abstains. `CallableDistiller(fn, name="...")` wraps a plain function. A `Draft`
holds `task`, `action`, optional `outcome` and `score`, and a `context` dict;
`record_conversation` merges it with ground truth and provenance before writing.
