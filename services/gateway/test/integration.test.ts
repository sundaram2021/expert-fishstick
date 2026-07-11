/**
 * End-to-end integration: real gateway (hash embedder, real HTTP) against a
 * controllable fake model backend. Exercises batching fan-out, semantic cache
 * hits, breaker trip → fail-fast → degraded serving → half-open recovery, and
 * queue back-pressure.
 */
import Fastify, { type FastifyInstance } from 'fastify';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { buildGateway, type BuiltGateway } from '../src/server.js';

interface FakeState {
  mode: 'ok' | 'fail';
  delayMs: number;
  calls: Array<{ batchId: string | null; size: number }>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

let fake: FastifyInstance;
let fakeUrl: string;
const fakeState: FakeState = { mode: 'ok', delayMs: 20, calls: [] };

let gw: BuiltGateway;
let gwUrl: string;

const infer = async (text: string) => {
  const res = await fetch(`${gwUrl}/infer`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return { status: res.status, body: (await res.json()) as any, headers: res.headers };
};

beforeAll(async () => {
  fake = Fastify({ logger: false });
  fake.post('/infer', async (req, reply) => {
    const body = req.body as { batch_id?: string; inputs: Array<{ id: string; text: string }> };
    fakeState.calls.push({ batchId: body.batch_id ?? null, size: body.inputs.length });
    if (fakeState.mode === 'fail') {
      return reply.code(500).send({ error: 'inference_failed', message: 'synthetic backend failure' });
    }
    await sleep(fakeState.delayMs);
    return {
      batch_id: body.batch_id ?? null,
      model: 'fake-model',
      batch_size: body.inputs.length,
      inference_ms: fakeState.delayMs,
      queue_wait_ms: 0,
      outputs: body.inputs.map((i) => ({
        id: i.id,
        result: { label: 'ok', score: 0.9, model: 'fake-model', tokens: i.text.split(' ').length },
      })),
    };
  });
  fake.get('/healthz', async () => ({ status: 'ok' }));
  await fake.listen({ port: 0, host: '127.0.0.1' });
  const addr = fake.server.address();
  fakeUrl = `http://127.0.0.1:${typeof addr === 'object' && addr ? addr.port : 0}`;

  gw = await buildGateway({
    logLevel: 'silent',
    modelBackendUrl: fakeUrl,
    modelTimeoutMs: 3_000,
    exposeMeta: true,
    adminEnabled: true,
    embedding: { provider: 'hash', windowMs: 2, maxBatchSize: 32 },
    batch: { windowMs: 25, maxSize: 4, maxConcurrentBatches: 2, maxQueueDepth: 100 },
    cache: {
      enabled: true,
      similarityThreshold: 0.9,
      degradedThreshold: 0.6,
      ttlMs: 60_000,
      maxSize: 100,
      serveStaleWhenOpen: true,
      staleGraceFactor: 3,
    },
    breaker: {
      failureThreshold: 2,
      latencyThresholdMs: 2_500,
      cooldownMs: 400,
      halfOpenRatio: 1,
      halfOpenMaxProbes: 1,
      halfOpenSuccessesToClose: 1,
    },
  });
  await gw.app.listen({ port: 0, host: '127.0.0.1' });
  const gaddr = gw.app.server.address();
  gwUrl = `http://127.0.0.1:${typeof gaddr === 'object' && gaddr ? gaddr.port : 0}`;
}, 30_000);

afterAll(async () => {
  await gw?.app.close();
  await fake?.close();
});

describe('gateway end-to-end', () => {
  it('serves a single request through the model with full meta + request id header', async () => {
    const r = await infer('the quarterly report numbers look excellent');
    expect(r.status).toBe(200);
    expect(r.body.result.label).toBe('ok');
    expect(r.body.meta.source).toBe('model');
    expect(r.body.meta.batch.size).toBeGreaterThanOrEqual(1);
    expect(r.headers.get('x-request-id')).toBeTruthy();
  });

  it('batches concurrent requests into one backend call, each getting its own response', async () => {
    fakeState.calls = [];
    const texts = [
      'alpha unique text one',
      'bravo unique text two',
      'charlie unique text three',
      'delta unique text four',
    ];
    const rs = await Promise.all(texts.map((t) => infer(t)));
    expect(rs.every((r) => r.status === 200)).toBe(true);
    const batchIds = new Set(rs.map((r) => r.body.meta.batch.id));
    expect(batchIds.size).toBe(1); // all four in one batch
    expect(rs[0]?.body.meta.batch.size).toBe(4);
    expect(fakeState.calls.filter((c) => c.size === 4)).toHaveLength(1);
    // fan-out isolation: each response has its own id
    expect(new Set(rs.map((r) => r.body.id)).size).toBe(4);
  });

  it('serves an exact repeat from the semantic cache without a backend call', async () => {
    fakeState.calls = [];
    const first = await infer('please summarize my open support tickets');
    expect(first.body.meta.source).toBe('model');
    const callsAfterFirst = fakeState.calls.length;
    const second = await infer('please summarize my open support tickets');
    expect(second.body.meta.source).toBe('cache');
    expect(second.body.meta.cache.similarity).toBe(1);
    expect(fakeState.calls.length).toBe(callsAfterFirst); // no new backend call
    expect(second.body.result).toEqual(first.body.result);
  });

  it('serves near-duplicate text from the cache above the similarity threshold', async () => {
    await infer('generate a summary of the incident timeline for the outage');
    const near = await infer('generate a summary of the incident timeline for the outage!');
    expect(near.body.meta.source).toBe('cache');
    expect(near.body.meta.cache.similarity).toBeGreaterThanOrEqual(0.9);
    expect(near.body.meta.cache.similarity).toBeLessThan(1);
  });

  it('trips the breaker after consecutive failures, then fails fast with 503', async () => {
    fakeState.mode = 'fail';
    // two failing batches (failureThreshold=2) — distinct texts to dodge the cache
    const f1 = await infer('zebra xylophone quantum blueberry');
    const f2 = await infer('penguin asteroid marmalade circuit');
    expect([f1.status, f2.status]).toEqual([502, 502]);

    const m1 = (await (await fetch(`${gwUrl}/metrics`)).json()) as any;
    expect(m1.circuit_breaker.state).toBe('open');
    expect(m1.circuit_breaker.trip_count).toBeGreaterThanOrEqual(1);

    // while open: immediate rejection, no batch window wait, retry-after set
    const t0 = performance.now();
    const rejected = await infer('walrus tangerine hypothesis dynamo');
    const elapsed = performance.now() - t0;
    expect(rejected.status).toBe(503);
    expect(rejected.body.error).toBe('circuit_open');
    expect(rejected.headers.get('retry-after')).toBeTruthy();
    expect(elapsed).toBeLessThan(150); // fail-fast: no 25ms window, no backend timeout
  });

  it('serves cached responses in degraded mode while the circuit is open', async () => {
    // 'please summarize my open support tickets' was cached earlier
    const r = await infer('please summarize my open support tickets');
    expect(r.status).toBe(200);
    expect(['cache', 'cache_degraded']).toContain(r.body.meta.source);
  });

  it('recovers through half-open probing once the backend heals', async () => {
    fakeState.mode = 'ok';
    await sleep(450); // cooldown (400ms) elapses
    const probe = await infer('fresh recovery text after cooldown');
    expect(probe.status).toBe(200);
    expect(probe.body.meta.source).toBe('model');
    expect(probe.body.meta.probe).toBe(true);
    expect(probe.body.meta.batch.size).toBe(1);

    const m = (await (await fetch(`${gwUrl}/metrics`)).json()) as any;
    expect(m.circuit_breaker.state).toBe('closed');

    const normal = await infer('back to normal batched operation');
    expect(normal.status).toBe(200);
    expect(normal.body.meta.probe).toBeUndefined();
  });

  it('applies back-pressure with 429 when the queue is full', async () => {
    const restore = { ...gw.cfg.batch };
    gw.cfg.batch.maxConcurrentBatches = 1;
    gw.cfg.batch.maxSize = 2;
    gw.cfg.batch.maxQueueDepth = 2;
    fakeState.delayMs = 300;
    try {
      const rs = await Promise.all(
        Array.from({ length: 10 }, (_, i) => infer(`overload unique text number ${i} ${Math.random()}`)),
      );
      const statuses = rs.map((r) => r.status);
      expect(statuses.filter((s) => s === 429).length).toBeGreaterThanOrEqual(1);
      expect(statuses.filter((s) => s === 200).length).toBeGreaterThanOrEqual(2);
    } finally {
      Object.assign(gw.cfg.batch, restore);
      fakeState.delayMs = 20;
    }
  });

  it('exposes a debuggable /metrics payload', async () => {
    const m = (await (await fetch(`${gwUrl}/metrics`)).json()) as any;
    expect(m.latency_ms.total_request.p95).toBeGreaterThan(0);
    expect(m.latency_ms.model_backend_call.count).toBeGreaterThan(0);
    expect(m.batching.batches).toBeGreaterThan(0);
    expect(m.batching.avg_batch_size).toBeGreaterThan(0);
    expect(m.semantic_cache.hits).toBeGreaterThan(0);
    expect(m.semantic_cache.hit_rate).toBeGreaterThan(0);
    expect(m.circuit_breaker.trip_count).toBeGreaterThanOrEqual(1);
    expect(m.requests.served_by.cache).toBeGreaterThan(0);
  });

  it('runtime config via /admin/config takes effect immediately', async () => {
    const res = await fetch(`${gwUrl}/admin/config`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ batch_max_size: 1, bogus_key: 5 }),
    });
    const body = (await res.json()) as any;
    expect(body.applied.batch_max_size).toBe(1);
    expect(body.rejected.bogus_key).toBe('unknown_key');
    expect(gw.cfg.batch.maxSize).toBe(1);
    // restore
    await fetch(`${gwUrl}/admin/config`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ batch_max_size: 4 }),
    });
  });

  it('rejects malformed requests with 400', async () => {
    const res = await fetch(`${gwUrl}/infer`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ nope: true }),
    });
    expect(res.status).toBe(400);
  });
}, 30_000);
