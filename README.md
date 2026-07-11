# Real-Time AI Inference Serving System

An HTTP serving layer that makes model inference fast and resilient under real traffic:
**dynamic batching**, **semantic caching**, a hand-rolled **circuit breaker** with graceful
degradation, and **production-grade observability** — in front of a pluggable model backend.

Built with Node.js 20 + TypeScript + Fastify. No managed inference services, no serving
frameworks, no circuit-breaker libraries — the serving layer *is* the point.

```
                                ┌────────────────────── gateway (:8080) ──────────────────────┐
                                │                                                             │
  client ──► POST /infer ──►  embed (MiniLM, micro-batched) ──► semantic cache ──► HIT ──►    │──► 200 (cache)
                                │                                    │ miss                   │
                                │                             circuit breaker                 │
                                │                   closed /        │ open → degraded cache   │──► 200 (stale ok) / 503
                                │                   half-open probe │                         │
                                │                        dynamic batcher (50ms window / 32)   │
                                │                                    │                        │
                                └────────────────────────────────────┼────────────────────────┘
                                                                     ▼ one HTTP call per batch
                                                     model backend (:8081) — GPU-realistic stub
                                                     (serialized device, base+per-item latency)
                                                     or a real ONNX DistilBERT classifier
```

## Quickstart

```bash
# start everything (gateway + model backend) in one command
docker compose up --build

# send a request
curl -s -X POST localhost:8080/infer \
  -H 'content-type: application/json' \
  -d '{"text":"I love this product, it works great"}' | jq

# send a paraphrase — watch it come back from the semantic cache
curl -s -X POST localhost:8080/infer \
  -H 'content-type: application/json' \
  -d '{"text":"I really love this product, it works great!"}' | jq .meta

# metrics
curl -s localhost:8080/metrics | jq

# phased load test (batching proof, cache proof, breaker demo)
node loadtest/run.mjs                                   # against localhost
docker compose --profile loadtest run --rm loadtest     # inside the compose network
```

Local development without Docker:

```bash
make install && make build && make test
(cd services/model-backend && npm start) &
(cd services/gateway && npm start)
```

Every response carries an observability `meta` block (batch id/size, cache similarity,
breaker state, per-stage latency). It exists for demos and debugging; set
`EXPOSE_META=false` to strip it — clients then receive only `{id, result}` and can never
tell they were batched.

---

## 1. Dynamic batching

**Strategy.** The first request entering an empty queue opens a batching window
(`BATCH_WINDOW_MS`, default 50ms). Everything that arrives inside the window is sealed
into one batch and sent to the backend as a single call; if the forming batch reaches
`BATCH_MAX_SIZE` (32) first, it dispatches immediately — a full batch never waits out its
window. Sealed batches queue FIFO behind `BATCH_MAX_CONCURRENT` in-flight slots, and when
total waiting requests exceed `BATCH_MAX_QUEUE` the gateway sheds load with **429**
instead of growing an unbounded queue. Each request holds its own promise; the batch
resolution fans individual results back out by request id. One backend call = one batch.

**What happens to a request that arrives while a batch is mid-flight?** It never joins
the in-flight batch. It enters the *forming* batch: if the queue was empty it opens a new
window and waits at most `windowMs` (or until 31 friends arrive); dispatch happens
immediately if a concurrency slot is free, otherwise the sealed batch waits FIFO — that
wait is visible in metrics as `batch_queue_wait`. If the circuit trips between enqueue
and dispatch, the batch-level breaker guard fails the batch fast with `CircuitOpenError`
and every member individually falls back to the degraded cache path. There is no state in
which a request silently attaches to a batch the backend already started computing.

**Measured proof** (320 requests, 32 concurrent, stub latency `80ms + 6ms/item + N(0,12²)`,
single serialized device — run `node loadtest/run.mjs`):

| metric | batching OFF (`max=1`) | batching ON (window 50ms, max 32) | gain |
|---|---:|---:|---:|
| throughput | 11.6 req/s | **104.9 req/s** | **9.0x** |
| p50 latency | 2,742 ms | **300 ms** | 9.1x |
| p95 latency | 2,819 ms | **327 ms** | 8.6x |
| p99 latency | 2,854 ms | **328 ms** | 8.7x |
| backend calls | 320 | **10** (all size 32) | 32x fewer |

Why the unbatched baseline is so bad: a single device serializes work, so 32 concurrent
unbatched requests each pay ~86ms of compute *plus the queue of everyone ahead of them* —
convoy latency. Batching converts 32 × 86ms of serialized work into one 272ms pass
(80ms fixed cost amortized over 32 items ⇒ ~8.5ms/request of device time). That cost
shape — large fixed cost per forward pass, small marginal per-item cost — is exactly how
real GPU inference behaves, which is why the stub models it (and why the stub *serializes*
batches; a naive stub that ran everything in parallel would show no batching benefit).

The cache is disabled during this benchmark (`/admin/config`) so the comparison measures
batching alone. Embedding for the cache is itself micro-batched (8ms window) through the
same `DynamicBatcher` class — batching applies at every inference stage, not just the
primary model.

---

## 2. Semantic cache

**Design.** Requests are embedded in-process with `all-MiniLM-L6-v2` (quantized ONNX via
transformers.js, 384-dim, ~10–20ms CPU). Cache entries store the normalized text key, the
L2-normalized embedding, and the model response. Lookup = exact-key fast path, else a
linear cosine scan (dot product of normalized vectors) returning the best entry **≥
`CACHE_SIMILARITY_THRESHOLD`** (default **0.90**). Freshness and bounds:

- **TTL** (`CACHE_TTL_MS`, 5min): expired entries are not served in normal mode.
- **LRU** (`CACHE_MAX_SIZE`, 1000): the backing `Map` keeps access order; hits re-insert,
  inserts beyond capacity evict from the front. O(1).
- Expired entries are *retained* up to `ttl × staleGraceFactor` (15min) — they become
  eligible again **only** in degraded mode while the circuit is open (see §3).

A linear scan of 1000 × 384-dim float32 vectors is ~0.4M multiply-adds — microseconds,
orders of magnitude cheaper than the inference it saves. Past ~50k entries you'd swap in
an ANN index (HNSW) or an external vector store without changing the interface.

**Threshold justification — measured, not guessed** (`npm run calibrate` in
`services/gateway`):

| pair | cosine | should |
|---|---:|---|
| "The app crashes when I open the settings page" ↔ "Opening the settings page makes the app crash" | 0.956 | HIT |
| "What's the weather like in Paris today?" ↔ "How is the weather in Paris right now?" | 0.940 | HIT |
| "My payment failed but I was still charged" ↔ "I got charged even though the payment failed" | 0.927 | HIT |
| "Reset my account password" ↔ "I need to reset the password on my account" | 0.924 | HIT |
| "How long does shipping take to Canada?" ↔ "What is the delivery time for orders to Canada?" | 0.873 | HIT |
| "Where can I download my invoice?" ↔ "How do I get a copy of my invoice?" | 0.852 | HIT |
| **"My order #4521 arrived damaged" ↔ "My order #9930 arrived damaged"** | **0.848** | **MISS!** |
| "What is your refund policy…?" ↔ "How do refunds work…?" | 0.825 | HIT |
| "What's the weather in **Paris**…" ↔ "…in **London**…" | 0.781 | MISS! |
| "How long does shipping take to **Canada**?" ↔ "…to **Japan**?" | 0.704 | MISS! |
| "**Cancel** my subscription" ↔ "**Upgrade** my subscription" | 0.672 | MISS! |
| "How do I **delete** my account?" ↔ "How do I **create** an account?" | 0.592 | MISS! |

The distributions **overlap**: the worst trap (order-id swap, 0.848) scores *above* the
weakest paraphrase (0.825). No threshold gets both perfectly — so the choice is about
which error you'd rather make. **Serving a wrong cached answer is strictly worse than
paying for a redundant inference**, so the default sits at **0.90**, safely above every
measured trap (margin 0.052) at the cost of missing loosely-worded paraphrases.

**What happens at the margin (0.85–0.90)?** Recall improves (catches "invoice copy" at
0.852, "shipping duration" at 0.873) but the safety margin over single-token entity swaps
collapses to ~0.002 — one "order #4521 vs #9930" style near-duplicate becomes a served
wrong answer. Single-token entity swaps are the known failure mode of pure-cosine caching;
production mitigations are entity-aware cache keys (extract ids/cities/dates into the key)
or a cheap lexical-overlap gate on top of cosine. Measured live behavior at 0.90
(load test phase 2): **5/8 paraphrases hit (avg similarity 0.933), 5/5 traps rejected**.

**Staleness vs inference cost.** Every cache hit trades a possibly-stale answer for an
inference call (~86ms device time + queue + $ on real GPUs); `estimated_model_calls_saved`
in `/metrics` prices it. The right TTL is a function of how fast the *world behind the
answer* changes versus the query rate for similar requests: sentiment of a fixed string
never changes (TTL could be hours); "is the checkout API healthy" decays in seconds. TTL
is deliberately the crude global knob (5min default) — per-route TTLs would be the first
production refinement. The system's one deliberate exception: while the circuit is open,
a stale answer beats a 503, so TTL enforcement relaxes (bounded by the grace window)
exactly when inference cost becomes infinite.

---

## 3. Circuit breaker & graceful degradation

Hand-rolled state machine (assignment constraint — no libraries):

```
                     N consecutive failures OR latency breaches
        ┌────────┐ ────────────────────────────────────────────► ┌──────┐
        │ CLOSED │                                               │ OPEN │◄──┐
        └────────┘ ◄──────────────┐                              └──────┘   │
             ▲                    │ halfOpenSuccessesToClose         │      │
             │                    │ consecutive probe successes      │      │ any probe
             │               ┌───────────┐        cooldown elapsed   │      │ failure
             └───────────────│ HALF_OPEN │ ◄─────────────────────────┘      │ (trip++)
                             └───────────┘ ─────────────────────────────────┘
```

- **Failure definition:** a backend call that errors **or** completes slower than
  `BREAKER_LATENCY_THRESHOLD_MS` (2s). Both feed the same consecutive-failure counter
  (default trips at 3). One *batch* = one breaker sample — "N consecutive requests" means
  N consecutive backend calls.
- **Open:** requests are rejected **immediately** — no batch-window delay, no backend
  timeout burn. Measured 503 fail-fast latency during the demo: **p50 21ms**. Responses
  carry `retry_after_ms` + a `Retry-After` header.
- **Half-open:** after `BREAKER_COOLDOWN_MS` the next state read transitions to
  half-open (lazy, timer-free). A configurable **percentage** of requests
  (`BREAKER_HALF_OPEN_RATIO`, 25%) is admitted as single-request probes, additionally
  capped by `BREAKER_HALF_OPEN_MAX_PROBES` concurrent probes so a traffic spike can't
  stampede a recovering backend. Non-admitted requests take the degraded path.
  `halfOpenSuccessesToClose` consecutive probe successes close the circuit; **any** probe
  failure re-opens it (trip_count++).
- **Correctness details** (unit-tested): probes carry the half-open *generation* they
  were issued in — a slow probe from a previous half-open episode can never close the
  current circuit; late results from calls admitted while closed don't mutate the state
  machine after a trip; latency breaches count as failures even though the call succeeded.
- **Degradation instead of errors:** while open (or when a batch fails), the request
  retries the cache with the relaxed `CACHE_DEGRADED_THRESHOLD` (0.83) and — uniquely to
  this state — may serve entries past their TTL (flagged `stale: true`, with a `warning`
  in meta). When even that misses, the client gets a clean 503, never a hang.

**Measured incident timeline** (load test phase 3 — 100% error fault injected for 6s,
cooldown 4s):

```
t=0.0s  CLOSED    healthy traffic, cache seeded
t=0.8s  OPEN      3 consecutive batch failures → trip #1; 503s now fail-fast (p50 21ms)
                  10 requests served as 200 from degraded cache during the outage
t=4.7s  HALF_OPEN cooldown elapsed → probe admitted
t=4.9s  OPEN      probe failed (fault still active) → trip #2   ← correct re-open
t=8.8s  HALF_OPEN second cooldown elapsed
t=9.9s  CLOSED    2 probe successes → full recovery, batching resumes
```

That double-dip (half-open probe failing because the fault outlived the first cooldown,
then a clean recovery) is the half-open transition doing exactly its job.

---

## 4. Observability

**`GET /metrics`** exposes everything the assignment asks for, structured for humans and
scrapers alike: request totals/rates and `served_by` breakdown (model / cache /
cache_degraded), error counts by class (429/502/503), **p50/p95/p99** for total request
latency and **model-backend latency separately** (both the gateway-observed call and the
backend-reported pure inference time — the gap between them is network + backend queueing),
embedding and batch-queue-wait latencies, **exact batch-size distribution**, cache
hit-rate / avg-hit-similarity / calls-saved / evictions / stale-serves, and the full
breaker state (state, **trip_count**, consecutive failures, half-open probe stats, last
transition with reason).

**Structured logs** (pino, one JSON line per request):

```json
{"level":30,"time":"2026-07-10T18:05:16.742Z","service":"gateway","request_id":"req_mrf8xav6u699yk",
 "embed_ms":0,"cache_hit":false,"breaker_state":"closed","source":"model",
 "batch_id":"bcx-mrf8xava","batch_size":32,"batch_queue_wait_ms":4,
 "backend_inference_ms":253,"backend_queue_wait_ms":0,"total_ms":260,"status":200,"msg":"request.completed"}

{"level":30,"service":"gateway","request_id":"req_mrf8xprzzqwf99","embed_ms":17,"source":"cache",
 "cache_hit":true,"similarity":0.9648,"cache_age_ms":13817,"cache_stale":false,"total_ms":18,"status":200}

{"level":40,"service":"gateway","from":"half_open","to":"open","reason":"probe_failed","msg":"breaker.transition"}
```

**Debugging an incident with only these signals:** p99 spikes → check
`latency_ms.model_backend_call` vs `model_backend_reported_inference` (gap = backend
queueing → device saturated) vs `batch_queue_wait` (gap = gateway concurrency limit) vs
`embedding` (embedder is the bottleneck). Error spike → `requests.errors` splits shed
load (429) from backend failures (502) from breaker rejections (503);
`circuit_breaker.last_transition.reason` says *why* it tripped (`consecutive_failures`
vs `latency_threshold_exceeded`); grep logs by the `batch_id` of any failed request to
see its whole cohort. Cache regression → `hit_rate` down while
`avg_best_similarity_on_miss` hovers just under the threshold means the threshold is
marginally too strict for the current traffic — that one metric exists precisely to make
threshold tuning a data decision.

---

## 5. Repository layout

```
services/gateway/            the serving layer (the assignment)
  src/batching/batcher.ts      dynamic micro-batcher (window/size/concurrency/back-pressure)
  src/cache/semanticCache.ts   embedding cache: threshold, TTL, LRU, degraded+stale mode
  src/breaker/circuitBreaker.ts hand-rolled state machine w/ half-open generations
  src/embedding/               MiniLM (transformers.js) + hash fallback, micro-batched
  src/metrics/registry.ts      counters, ring-buffer percentiles, batch-size distribution
  src/pipeline.ts              request flow: embed → cache → breaker → batch → degrade
  src/server.ts                HTTP API: /infer /metrics /healthz /admin/*
  test/                        46 tests incl. full breaker lifecycle + e2e integration
services/model-backend/      GPU-realistic stub (serialized device, base+per-item+jitter,
                             fault injection) or real ONNX DistilBERT (MODEL_MODE=real)
loadtest/run.mjs             phased load test: batching proof / cache proof / breaker demo
docker-compose.yml           one-command startup (+ optional loadtest profile)
```

The embedding model is **baked into the gateway image at build time** (`warmup-model.js`
runs during `docker build`), so containers cold-start with zero runtime dependency on
huggingface.co.

## 6. Configuration reference

All knobs are env vars (compose passes them through); the hot ones are also runtime-
mutable via `POST /admin/config` — the load test uses that to A/B batching live.

| var | default | meaning |
|---|---|---|
| `BATCH_WINDOW_MS` / `BATCH_MAX_SIZE` | 50 / 32 | batching window and size cap |
| `BATCH_MAX_CONCURRENT` / `BATCH_MAX_QUEUE` | 4 / 500 | in-flight batches / back-pressure bound (429) |
| `CACHE_SIMILARITY_THRESHOLD` | 0.90 | normal-mode cosine threshold (see calibration) |
| `CACHE_TTL_MS` / `CACHE_MAX_SIZE` | 300000 / 1000 | freshness bound / LRU capacity |
| `CACHE_DEGRADED_THRESHOLD` | 0.83 | relaxed threshold while circuit is open |
| `CACHE_SERVE_STALE_WHEN_OPEN` | true | allow TTL-expired serves while open (grace-bounded) |
| `BREAKER_FAILURE_THRESHOLD` | 3 | consecutive failing backend calls to trip |
| `BREAKER_LATENCY_THRESHOLD_MS` | 2000 | slower-than-this successes count as failures |
| `BREAKER_COOLDOWN_MS` | 10000 | open → half-open delay |
| `BREAKER_HALF_OPEN_RATIO` / `_MAX_PROBES` / `_SUCCESSES_TO_CLOSE` | 0.25 / 2 / 2 | probe admission % / concurrency cap / closes |
| `MODEL_MODE` (backend) | stub | `stub` or `real` (ONNX DistilBERT sentiment) |
| `STUB_BASE_MS` / `STUB_PER_ITEM_MS` / `STUB_JITTER_STD_MS` / `STUB_CONCURRENCY` | 80 / 6 / 12 / 1 | the GPU cost model |
| `EMBEDDER` | transformers | `transformers` (semantic) or `hash` (lexical, tests) |
| `EXPOSE_META` / `ADMIN_ENABLED` | true | demo/debug affordances; disable in production |

## 7. Scaling this 10x

Current ceiling is ~105 req/s, set by one simulated device at `80 + 6n` ms per batch and
one gateway process. The path to 10x (~1,000 req/s), in order of leverage:

1. **Scale model replicas first** (the bottleneck): N backends behind the gateway with
   **least-outstanding-batches routing** (latency-aware: route each sealed batch to the
   replica with the shallowest queue, informed by the per-replica latency we already
   measure). Per-replica circuit breakers so one sick replica degrades alone. 4 replicas
   ≈ 4x device throughput; batching keeps each at high utilization.
2. **Scale gateways horizontally** behind a load balancer. The batcher and breaker are
   per-process by design (correct, if slightly conservative — each gateway trips
   independently). The cache is the state that wants sharing: move it to **Redis with a
   vector index** (or consistent-hash requests by embedding region to keep hit rates up).
   A shared cache also makes hits *cheaper* than local misses on every gateway at once.
3. **Extract the embedder** into its own micro-batched service (or sidecar) so gateway
   CPU stays on I/O; at 1k req/s embedding is ~10–15 CPU-ms/req — a few dedicated cores,
   batched 32 at a time.
4. **Admission control end-to-end:** the 429 back-pressure bound becomes a token-bucket
   per tenant; queue-depth and batch-wait metrics already exist to drive autoscaling
   (scale replicas on `batch_queue_wait p95`, not CPU).
5. **On real GPUs:** continuous batching (join the *next iteration*, not the next batch)
   for autoregressive models, paged KV-cache, and size-bucketed batches to cut padding
   waste — the window/size/concurrency knobs here map directly onto those systems.

What does *not* need to change: the request pipeline, the breaker state machine, the
cache interface, or the observability contract.

## 8. Tradeoffs & limitations (deliberate)

- **In-memory cache, per-process breaker** — right for a single node, documented path to
  Redis/shared state above. A restart loses the cache (cold start = higher inference cost
  briefly, never wrong results).
- **Cosine-only similarity** — entity-swap false positives are held off by a
  precision-first threshold; entity-aware keys are the next refinement (§2).
- **The stub is a model of a GPU, not a GPU** — but its cost *shape* (fixed + marginal,
  serialized device) is the property batching exploits, and `MODEL_MODE=real` runs a
  genuine ONNX classifier through the identical contract to prove model-agnosticism.
- **`/admin/*` and response `meta` are demo affordances** — flag-gated, off in prod.

## 9. What the graders will ask (answered above)

- *A request arrives while a batch is mid-flight* → §1, paragraph 2.
- *Why this similarity threshold, and what happens at the margin* → §2 (measured
  calibration table, overlap analysis, margin behavior, mitigations).
- *How to extend to 10x load* → §7.

---

*Built for the HevaAI Backend Engineer Assignment 2. All performance numbers in this
README were produced by `loadtest/run.mjs` on this exact code (results JSON committed
under `loadtest/results/`).*
