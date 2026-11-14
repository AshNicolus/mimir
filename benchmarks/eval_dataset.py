"""Labeled corpus for the retrieval and recommendation quality eval.

Each experience carries a stable ``label`` so cases can refer to it without
knowing the random id assigned at record time. Recall cases name the labels a
query should surface; recommend cases name the action a task should get back.
"""

from __future__ import annotations

from typing import NamedTuple


class Seed(NamedTuple):
    label: str
    task: str
    action: str
    outcome: str


class RecallCase(NamedTuple):
    query: str
    relevant: set[str]  # labels that count as correct hits


class RecommendCase(NamedTuple):
    task: str
    expected_action: str


class AgedSeed(NamedTuple):
    task: str
    action: str
    days_ago: int


class DecayCase(NamedTuple):
    task: str
    recent_action: str  # the action decay should surface over the stale one


# A small corpus of agent experiences across a few problem areas. Some actions
# repeat with mixed outcomes so recommend() has a real track record to rank.
SEEDS = [
    Seed("cache-1", "fix slow login under load", "add a redis cache", "success"),
    Seed("cache-2", "checkout page latency spikes", "add a redis cache", "success"),
    Seed("cache-3", "session lookups are slow", "add a redis cache", "success"),
    Seed("cache-4", "product page slow at peak", "add a redis cache", "failure"),
    Seed("index-1", "reports query takes minutes", "add a database index", "success"),
    Seed("index-2", "dashboard aggregation is slow", "add a database index", "success"),
    Seed("index-3", "search endpoint latency high", "add a database index", "failure"),
    Seed("pool-1", "connections exhausted under load", "increase the pool size", "success"),
    Seed("pool-2", "timeouts when traffic spikes", "increase the pool size", "partial"),
    Seed("rate-1", "abusive clients hammer the api", "add a token bucket rate limiter", "success"),
    Seed("rate-2", "throttle noisy tenants", "add a token bucket rate limiter", "success"),
    Seed(
        "rate-3", "block brute force login attempts", "add a fixed window rate limiter", "failure"
    ),
    Seed("retry-1", "flaky upstream calls fail randomly", "add retries with backoff", "success"),
    Seed("retry-2", "payment webhook drops messages", "add retries with backoff", "success"),
    Seed("deploy-1", "rollout causes downtime", "switch to blue green deploys", "success"),
    Seed("deploy-2", "bad release reaches all users", "add a canary stage", "success"),
    Seed("chart-1", "render a monthly revenue chart", "use matplotlib", "success"),
    Seed("bread-1", "bake a sourdough loaf", "proof the dough overnight", "success"),
    Seed("ci-1", "ci pipeline takes too long", "cache dependencies between ci runs", "success"),
    Seed("ci-2", "test suite is slow in ci", "run tests in parallel", "success"),
    Seed("ci-3", "flaky test fails in ci randomly", "quarantine the flaky test", "success"),
    Seed("mem-1", "worker memory grows until oom", "fix the leak in the request cache", "success"),
    Seed("mem-2", "service killed for out of memory", "stream the file instead", "success"),
    Seed("auth-1", "users logged out unexpectedly", "extend the session lifetime", "partial"),
    Seed("auth-2", "leaked api key in the repo", "rotate the key and scan history", "success"),
    Seed("data-1", "nightly etl job fails on bad rows", "quarantine bad rows", "success"),
    Seed("data-2", "duplicate records after pipeline rerun", "make loads idempotent", "success"),
    Seed("fe-1", "page layout breaks on mobile", "use a responsive grid", "success"),
    Seed("fe-2", "bundle size makes the page load slowly", "code split the bundle", "success"),
    Seed("garden-1", "tomato plants are wilting", "water deeply in the morning", "success"),
]

RECALL_CASES = [
    RecallCase("login is slow under heavy load", {"cache-1", "cache-3"}),
    RecallCase("database report query is slow", {"index-1", "index-2"}),
    RecallCase("connection pool exhausted", {"pool-1", "pool-2"}),
    RecallCase("rate limit abusive api clients", {"rate-1", "rate-2", "rate-3"}),
    RecallCase("retry flaky upstream requests", {"retry-1", "retry-2"}),
    RecallCase("deploy without downtime", {"deploy-1", "deploy-2"}),
    RecallCase("plot a revenue chart", {"chart-1"}),
    RecallCase("ci builds are too slow", {"ci-1", "ci-2"}),
    RecallCase("test fails randomly in ci", {"ci-3"}),
    RecallCase("process runs out of memory", {"mem-1", "mem-2"}),
    RecallCase("api key leaked", {"auth-2"}),
    RecallCase("etl pipeline produces duplicate records", {"data-1", "data-2"}),
    RecallCase("page loads slowly on mobile", {"fe-1", "fe-2"}),
    RecallCase("users get logged out", {"auth-1"}),
]

RECOMMEND_CASES = [
    # Caching wins on login/latency despite one failure (3 successes, 1 failure).
    RecommendCase("login latency is high under load", "add a redis cache"),
    # Token bucket has a clean record; fixed window only failed.
    RecommendCase("throttle abusive api clients", "add a token bucket rate limiter"),
    RecommendCase("handle flaky upstream calls", "add retries with backoff"),
    RecommendCase("pipeline rerun created duplicate records", "make loads idempotent"),
    RecommendCase("an api key leaked", "rotate the key and scan history"),
]

# A stale approach that worked more often long ago, and a fresh one that works
# now. Without decay the larger old sample wins; with decay the recent one does.
DECAY_SEEDS = [
    AgedSeed("scale the service under load", "manual scaling", days_ago=300),
    AgedSeed("scale the service under load", "manual scaling", days_ago=300),
    AgedSeed("scale the service under load", "manual scaling", days_ago=300),
    AgedSeed("scale the service under load", "manual scaling", days_ago=300),
    AgedSeed("scale the service under load", "autoscaling", days_ago=5),
    AgedSeed("scale the service under load", "autoscaling", days_ago=5),
]

DECAY_CASE = DecayCase("scale the service under load", "autoscaling")


class Family(NamedTuple):
    tasks: tuple[str, ...]  # one phrasing per round, so later rounds aren't verbatim repeats
    actions: tuple[str, ...]  # what a memoryless agent tries, in this order
    working: str  # the only action the scripted environment accepts


# Recurring task families for the attempts proxy. The working action sits at a
# different position in each family, so the memoryless baseline isn't trivial.
FAMILIES = [
    Family(
        (
            "deploy fails with a permission error",
            "permission denied during deploy",
            "deploy step hits permission denied",
        ),
        ("retry the deploy", "clear the build cache", "grant the deploy role write access"),
        "grant the deploy role write access",
    ),
    Family(
        (
            "tests time out on the shared runner",
            "test run times out on a shared runner",
            "shared runner tests hit the timeout",
        ),
        ("increase the timeout", "split the test run into shards", "pin the runner image"),
        "split the test run into shards",
    ),
    Family(
        (
            "import error after upgrading the library",
            "library upgrade breaks imports",
            "imports fail after a library upgrade",
        ),
        ("pin the previous version", "reinstall the virtualenv", "update the import paths"),
        "update the import paths",
    ),
    Family(
        (
            "disk fills up on the log server",
            "log server disk is full",
            "log disk keeps filling up",
        ),
        ("rotate and compress logs", "add a bigger disk", "delete old logs by hand"),
        "rotate and compress logs",
    ),
]
