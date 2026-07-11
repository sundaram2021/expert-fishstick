#!/usr/bin/env node
/**
 * Phased load test for the inference gateway. Zero dependencies — plain Node.
 *
 *   Phase 1  batching   : identical workload with batching OFF (max_batch_size=1)
 *                         vs ON — proves the throughput/latency win with numbers.
 *   Phase 2  cache      : seeds prompts, replays paraphrases (should HIT) and
 *                         entity-swap/intent-flip traps (must MISS).
 *   Phase 3  breaker    : injects backend faults, watches closed → open →
 *                         half_open → closed, including degraded cache serving.
 *
 * Usage:
 *   node loadtest/run.mjs [--phase all|batching|cache|breaker]
 *                         [--requests 320] [--concurrency 32]
 *   GATEWAY_URL / BACKEND_URL env vars override targets.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  BREAKER_DEGRADED_QUERIES,
  BREAKER_SEEDS,
  PARAPHRASE_PAIRS,
  TRAP_PAIRS,
  uniqueText,
} from './scenarios.mjs';

const GATEWAY = process.env.GATEWAY_URL ?? 'http://localhost:8080';
const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8081';

const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1] ?? 'true'] : [])).filter((x) => x.length),
);
const PHASE = args.phase ?? 'all';
const TOTAL = Number(args.requests ?? 320);
const CONC = Number(args.concurrency ?? 32);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pctl = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))] ?? null;
const fmt = (v, d = 1) => (v === null || v === undefined ? 'n/a' : Number(v).toFixed(d));

async function post(pathname, body, base = GATEWAY) {
  const res = await fetch(`${base}${pathname}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  let json = null;
  try {
    json = await res.json();
  } catch {
    /* non-JSON error body */
  }
  return { status: res.status, body: json };
}

const admin = (patch) => post('/admin/config', patch);
const resetMetrics = () => post('/admin/metrics/reset', {});
const clearCache = () => post('/admin/cache/clear', {});
const setFault = (mode, opts = {}) => post('/admin/fault', { mode, ...opts }, BACKEND);
const getMetrics = async () => (await fetch(`${GATEWAY}/metrics`)).json();

function summarize(results, elapsedMs) {
  const ok = results.filter((r) => r.status === 200);
  const lat = ok.map((r) => r.ms).sort((a, b) => a - b);
  const statuses = {};
  for (const r of results) statuses[r.status] = (statuses[r.status] ?? 0) + 1;
  return {
    total: results.length,
    ok: ok.length,
    statuses,
    elapsed_s: +(elapsedMs / 1000).toFixed(2),
    throughput_rps: +(results.length / (elapsedMs / 1000)).toFixed(1),
    latency_ms: {
      p50: pctl(lat, 0.5),
      p95: pctl(lat, 0.95),
      p99: pctl(lat, 0.99),
      avg: lat.length ? +(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(1) : null,
      max: lat.length ? +lat[lat.length - 1].toFixed(1) : null,
    },
  };
}

/** Closed-loop load: `concurrency` workers each pull the next request index. */
async function runClosedLoop({ name, total, concurrency, genText, paceMs = 0 }) {
  process.stdout.write(`\n▶ ${name} (${total} requests, concurrency ${concurrency}) ... `);
  const results = [];
  let next = 0;
  const t0 = performance.now();
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= total) return;
      const start = performance.now();
      try {
        const r = await post('/infer', { text: genText(i) });
        results.push({ i, status: r.status, ms: performance.now() - start, meta: r.body?.meta ?? null });
      } catch (err) {
        results.push({ i, status: 0, ms: performance.now() - start, error: String(err) });
      }
      if (paceMs > 0) await sleep(paceMs);
    }
  };
  await Promise.all(Array.from({ length: concurrency }, worker));
  const elapsed = performance.now() - t0;
  const s = summarize(results, elapsed);
  console.log(`done in ${s.elapsed_s}s → ${s.throughput_rps} req/s`);
  return { results, summary: s };
}

function printComparison(off, on, offM, onM) {
  const rows = [
    ['throughput (req/s)', off.summary.throughput_rps, on.summary.throughput_rps],
    ['p50 latency (ms)', off.summary.latency_ms.p50, on.summary.latency_ms.p50],
    ['p95 latency (ms)', off.summary.latency_ms.p95, on.summary.latency_ms.p95],
    ['p99 latency (ms)', off.summary.latency_ms.p99, on.summary.latency_ms.p99],
    ['avg batch size', offM.batching.avg_batch_size, onM.batching.avg_batch_size],
    ['backend calls', offM.model_backend.calls, onM.model_backend.calls],
  ];
  console.log('\n  metric                      | batching OFF | batching ON');
  console.log('  ----------------------------|--------------|------------');
  for (const [label, a, b] of rows) {
    console.log(`  ${label.padEnd(28)}| ${String(fmt(a)).padStart(12)} | ${String(fmt(b)).padStart(11)}`);
  }
  const thr = on.summary.throughput_rps / off.summary.throughput_rps;
  const p50 = off.summary.latency_ms.p50 / on.summary.latency_ms.p50;
  const p99 = off.summary.latency_ms.p99 / on.summary.latency_ms.p99;
  console.log(
    `\n  ⇒ batching: ${thr.toFixed(1)}x throughput, ${p50.toFixed(1)}x lower p50, ${p99.toFixed(1)}x lower p99, ` +
      `${offM.model_backend.calls}→${onM.model_backend.calls} backend calls`,
  );
  console.log(`  batch size distribution (ON): ${JSON.stringify(onM.batching.distribution)}`);
  return { throughput_gain: +thr.toFixed(2), p50_gain: +p50.toFixed(2), p99_gain: +p99.toFixed(2) };
}

async function phaseBatching() {
  console.log('\n════════ PHASE 1: DYNAMIC BATCHING — OFF vs ON ════════');
  // Isolate the variable under test:
  //  - cache OFF (template workloads are near-duplicates; semantic hits would
  //    contaminate the batching comparison),
  //  - breaker latency threshold raised (the unbatched baseline deliberately
  //    drowns the backend in queueing; the breaker demo comes later).
  await admin({
    cache_enabled: false,
    breaker_latency_threshold_ms: 120_000,
    model_timeout_ms: 120_000,
    batch_max_queue: 5_000,
  });

  // OFF: every request is its own backend call; extra in-flight slots so the
  // gateway adds no artificial queueing — the (simulated) device is the limit.
  await admin({ batch_max_size: 1, batch_window_ms: 0, batch_max_concurrent: 64 });
  await resetMetrics();
  const off = await runClosedLoop({
    name: 'batching OFF (max_batch_size=1)',
    total: TOTAL,
    concurrency: CONC,
    genText: (i) => uniqueText(i),
  });
  const offM = await getMetrics();

  await admin({ batch_max_size: 32, batch_window_ms: 50, batch_max_concurrent: 4 });
  await resetMetrics();
  const on = await runClosedLoop({
    name: 'batching ON (window=50ms, max_batch_size=32)',
    total: TOTAL,
    concurrency: CONC,
    genText: (i) => uniqueText(10_000 + i),
  });
  const onM = await getMetrics();

  const gains = printComparison(off, on, offM, onM);
  await admin({ cache_enabled: true, breaker_latency_threshold_ms: 2_000, model_timeout_ms: 10_000 });
  return { off: off.summary, on: on.summary, gains, batch_distribution_on: onM.batching.distribution };
}

async function phaseCache() {
  console.log('\n════════ PHASE 2: SEMANTIC CACHE — paraphrases hit, traps miss ════════');
  await admin({ batch_max_size: 32, batch_window_ms: 50 });
  await clearCache();
  await resetMetrics();

  // Seed every pair's first text.
  const seeds = [...PARAPHRASE_PAIRS.map((p) => p[0]), ...TRAP_PAIRS.map((p) => p[0])];
  const uniqueSeeds = [...new Set(seeds)];
  for (const s of uniqueSeeds) await post('/infer', { text: s });

  const pairResults = [];
  console.log('\n  paraphrase queries (expected: HIT):');
  let hits = 0;
  for (const [seed, query] of PARAPHRASE_PAIRS) {
    const r = await post('/infer', { text: query });
    const hit = r.body?.meta?.source === 'cache';
    const sim = hit ? r.body.meta.cache.similarity : r.body?.meta?.cache?.best_similarity;
    if (hit) hits++;
    pairResults.push({ kind: 'paraphrase', seed, query, hit, similarity: sim });
    console.log(`   ${hit ? '✅ HIT ' : '❌ MISS'} sim=${fmt(sim, 4)}  "${query}"`);
  }
  console.log('\n  trap queries (expected: MISS — a hit here would serve a WRONG answer):');
  let trapMisses = 0;
  for (const [seed, query, kind] of TRAP_PAIRS) {
    const r = await post('/infer', { text: query });
    const hit = r.body?.meta?.source === 'cache';
    const sim = hit ? r.body.meta.cache.similarity : r.body?.meta?.cache?.best_similarity;
    if (!hit) trapMisses++;
    pairResults.push({ kind: `trap:${kind}`, seed, query, hit, similarity: sim });
    console.log(`   ${hit ? '❌ HIT (bad!)' : '✅ MISS'} sim=${fmt(sim, 4)}  [${kind}] "${query}"`);
  }

  const m = await getMetrics();
  console.log(
    `\n  ⇒ paraphrase hit rate: ${hits}/${PARAPHRASE_PAIRS.length}, trap rejection: ${trapMisses}/${TRAP_PAIRS.length}` +
      `\n  ⇒ cache: hit_rate=${fmt(m.semantic_cache.hit_rate, 3)} avg_hit_similarity=${fmt(m.semantic_cache.avg_hit_similarity, 4)} ` +
      `calls_saved=${m.semantic_cache.estimated_model_calls_saved} threshold=${m.semantic_cache.similarity_threshold}`,
  );
  return {
    paraphrase_hits: hits,
    paraphrase_total: PARAPHRASE_PAIRS.length,
    trap_misses: trapMisses,
    trap_total: TRAP_PAIRS.length,
    cache_metrics: m.semantic_cache,
    pairs: pairResults,
  };
}

async function phaseBreaker() {
  console.log('\n════════ PHASE 3: CIRCUIT BREAKER — trip, degrade, recover ════════');
  await admin({
    breaker_failure_threshold: 3,
    breaker_cooldown_ms: 4_000,
    breaker_latency_threshold_ms: 2_000,
    batch_max_size: 8,
    batch_window_ms: 20,
  });
  await clearCache();
  await resetMetrics();

  // Seed the cache while healthy so degraded mode has material.
  for (const s of BREAKER_SEEDS) await post('/infer', { text: s });

  const timeline = [];
  let lastState = null;
  let polling = true;
  const t0 = performance.now();
  const poller = (async () => {
    while (polling) {
      try {
        const m = await getMetrics();
        const st = m.circuit_breaker.state;
        if (st !== lastState) {
          timeline.push({ t_s: +((performance.now() - t0) / 1000).toFixed(2), state: st, trips: m.circuit_breaker.trip_count });
          console.log(`   [t=${((performance.now() - t0) / 1000).toFixed(1)}s] breaker → ${st.toUpperCase()} (trips=${m.circuit_breaker.trip_count})`);
          lastState = st;
        }
      } catch {
        /* gateway busy */
      }
      await sleep(150);
    }
  })();

  console.log('\n  injecting 100% error fault into the model backend for 6s ...');
  await setFault('error', { error_rate: 1, duration_ms: 6_000 });

  const statuses = { '200_model': 0, '200_cache': 0, '200_degraded': 0, '502': 0, '503': 0, other: 0 };
  const failFastSamples = [];
  const send = async (text) => {
    const start = performance.now();
    const r = await post('/infer', { text });
    const ms = performance.now() - start;
    if (r.status === 200) {
      const src = r.body?.meta?.source;
      if (src === 'cache_degraded') statuses['200_degraded']++;
      else if (src === 'cache') statuses['200_cache']++;
      else statuses['200_model']++;
    } else if (r.status === 502) statuses['502']++;
    else if (r.status === 503) {
      statuses['503']++;
      failFastSamples.push(ms);
    } else statuses.other++;
    return r;
  };

  // ~14s of steady traffic: alternate novel texts with degraded-cache queries.
  const start = performance.now();
  let i = 0;
  while (performance.now() - start < 14_000) {
    const batch = [
      send(uniqueText(50_000 + i * 3)),
      send(uniqueText(50_000 + i * 3 + 1)),
      send(BREAKER_DEGRADED_QUERIES[i % BREAKER_DEGRADED_QUERIES.length]),
    ];
    await Promise.all(batch);
    await sleep(250);
    i++;
  }
  polling = false;
  await poller;

  const m = await getMetrics();
  console.log(`\n  responses during the incident window: ${JSON.stringify(statuses)}`);
  if (failFastSamples.length) {
    const sorted = failFastSamples.sort((a, b) => a - b);
    console.log(`  503 fail-fast latency: p50=${fmt(pctl(sorted, 0.5))}ms p95=${fmt(pctl(sorted, 0.95))}ms (no batch-window or timeout waits)`);
  }
  console.log(`  breaker end state: ${m.circuit_breaker.state}, trip_count=${m.circuit_breaker.trip_count}`);
  console.log(`  state timeline: ${timeline.map((t) => `${t.t_s}s:${t.state}`).join(' → ')}`);
  console.log(`  degraded cache serves during outage: ${m.requests.served_by.cache_degraded}`);

  const closed = m.circuit_breaker.state === 'closed';
  console.log(closed ? '  ✅ full recovery: circuit closed again' : '  ⚠ circuit not closed yet');
  return { timeline, statuses, breaker: m.circuit_breaker, degraded_serves: m.requests.served_by.cache_degraded };
}

async function main() {
  console.log(`Load test → gateway=${GATEWAY} backend=${BACKEND} phase=${PHASE} requests=${TOTAL} concurrency=${CONC}`);

  // Wait for readiness.
  for (let i = 0; i < 120; i++) {
    try {
      const h = await (await fetch(`${GATEWAY}/healthz`)).json();
      if (h.embedder_ready) break;
    } catch {
      /* not up yet */
    }
    if (i === 119) throw new Error('gateway not ready after 120s');
    await sleep(1000);
  }
  await setFault('none').catch(() => {});

  // Warmup: first inference pass initializes ONNX kernels.
  await runClosedLoop({ name: 'warmup', total: Math.min(32, TOTAL), concurrency: 8, genText: (i) => `warmup text ${i}` });

  const out = { started_at: new Date().toISOString(), gateway: GATEWAY, config: { requests: TOTAL, concurrency: CONC } };
  if (PHASE === 'all' || PHASE === 'batching') out.batching = await phaseBatching();
  if (PHASE === 'all' || PHASE === 'cache') out.cache = await phaseCache();
  if (PHASE === 'all' || PHASE === 'breaker') out.breaker = await phaseBreaker();
  out.final_metrics = await getMetrics();

  const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'results');
  await mkdir(dir, { recursive: true });
  const file = path.join(dir, `run-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
  await writeFile(file, JSON.stringify(out, null, 2));
  console.log(`\nfull results written to ${file}`);
}

main().catch((err) => {
  console.error('loadtest failed:', err);
  process.exit(1);
});
