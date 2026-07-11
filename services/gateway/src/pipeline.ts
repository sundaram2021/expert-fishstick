/**
 * The request pipeline — what happens to every inference request:
 *
 *   1. Embed the request text (micro-batched, local CPU).
 *   2. Semantic cache lookup — a sufficiently similar cached response is
 *      returned without touching the model.
 *   3. Circuit breaker gate:
 *        - CLOSED     -> join the dynamic batcher (normal path).
 *        - OPEN       -> degraded path: relaxed-threshold (optionally stale)
 *                        cache lookup; else 503 + Retry-After. Requests are
 *                        rejected immediately — no batch window delay.
 *        - HALF_OPEN  -> a bounded percentage of requests go through as
 *                        single-request probes; everyone else takes the
 *                        degraded path.
 *   4. Batched model call, guarded by the breaker at the *batch* level
 *      (one backend call == one breaker sample). If the circuit trips while
 *      a request is already queued, the sealed batch fails fast with
 *      CircuitOpenError and each member falls back to the degraded path
 *      individually.
 *   5. Successful results are inserted into the cache and fanned back out —
 *      each request receives only its own result.
 */
import { DynamicBatcher } from './batching/batcher.js';
import { CircuitBreaker } from './breaker/circuitBreaker.js';
import { SemanticCache, type CacheHit } from './cache/semanticCache.js';
import type { GatewayConfig } from './config.js';
import { BatchError, CircuitOpenError, QueueFullError } from './errors.js';
import type { EmbeddingService } from './embedding/embeddingService.js';
import type { Logger } from './logger.js';
import type { GatewayMetrics } from './metrics/registry.js';
import type { ModelClient } from './modelClient.js';
import type { InferenceInput, ModelResult } from './types.js';

interface ItemOut {
  result: ModelResult;
  backend: { inference_ms: number; queue_wait_ms: number; model: string };
}

export interface HandleResult {
  status: number;
  body: Record<string, unknown>;
  headers?: Record<string, string>;
  logFields: Record<string, unknown>;
}

export interface PipelineDeps {
  cfg: GatewayConfig;
  log: Logger;
  embeddings: EmbeddingService;
  cache: SemanticCache;
  breaker: CircuitBreaker;
  modelClient: ModelClient;
  metrics: GatewayMetrics;
}

export class InferencePipeline {
  readonly batcher: DynamicBatcher<InferenceInput, ItemOut>;

  constructor(private readonly d: PipelineDeps) {
    this.batcher = new DynamicBatcher<InferenceInput, ItemOut>(
      (inputs, batchId) => this.callBackendGuarded(inputs, batchId),
      () => ({
        windowMs: d.cfg.batch.windowMs,
        maxBatchSize: d.cfg.batch.maxSize,
        maxConcurrentBatches: d.cfg.batch.maxConcurrentBatches,
        maxQueueDepth: d.cfg.batch.maxQueueDepth,
      }),
      {
        onDispatch: (batchId, size) => {
          d.metrics.batchSizes.record(size);
          d.log.debug({ batch_id: batchId, batch_size: size }, 'batch.dispatched');
        },
        onSettle: (batchId, size, ok, ms) => {
          d.log[ok ? 'debug' : 'warn']({ batch_id: batchId, batch_size: size, ok, ms }, 'batch.settled');
        },
        onReject: () => d.metrics.queueRejections.inc(),
      },
    );
  }

  /** One backend call == one circuit-breaker sample. */
  private async callBackendGuarded(inputs: InferenceInput[], batchId: string): Promise<ItemOut[]> {
    const acq = this.d.breaker.tryAcquire();
    if (!acq.allowed) {
      throw new CircuitOpenError(acq.retryAfterMs ?? this.d.cfg.breaker.cooldownMs);
    }
    const t0 = performance.now();
    try {
      const resp = await this.d.modelClient.inferBatch(inputs, batchId);
      const callMs = performance.now() - t0;
      this.d.breaker.record(acq, { ok: true, latencyMs: callMs });
      this.d.metrics.backendCalls.inc();
      this.d.metrics.modelCallLatency.record(callMs);
      if (typeof resp.inference_ms === 'number') {
        this.d.metrics.backendInferenceLatency.record(resp.inference_ms);
      }
      const byId = new Map(resp.outputs.map((o) => [o.id, o.result]));
      return inputs.map((i) => {
        const result = byId.get(i.id);
        if (!result) throw new Error(`backend response missing output for id ${i.id}`);
        return {
          result,
          backend: {
            inference_ms: resp.inference_ms,
            queue_wait_ms: resp.queue_wait_ms,
            model: resp.model,
          },
        };
      });
    } catch (err) {
      if (!(err instanceof CircuitOpenError)) {
        this.d.breaker.record(acq, { ok: false, latencyMs: performance.now() - t0 });
        this.d.metrics.backendFailures.inc();
      }
      throw err;
    }
  }

  async handle(text: string, requestId: string): Promise<HandleResult> {
    const t0 = performance.now();
    const { cfg, metrics } = this.d;
    metrics.requests.inc();
    metrics.rate.record();

    const lf: Record<string, unknown> = { request_id: requestId };

    // 1) Embed (needed for cache lookup and insertion).
    let vector: Float32Array | null = null;
    let embedMs = 0;
    if (cfg.cache.enabled) {
      try {
        const e = await this.d.embeddings.embedOne(text);
        vector = e.vector;
        embedMs = e.ms;
      } catch (err) {
        lf.embed_error = err instanceof Error ? err.message : String(err);
        this.d.log.warn({ request_id: requestId, err: lf.embed_error }, 'embed.failed — bypassing cache');
      }
    }
    lf.embed_ms = embedMs;

    // 2) Semantic cache.
    if (vector) {
      const outcome = this.d.cache.lookup(text, vector, 'normal');
      if (outcome.hit) {
        return this.respondCached(requestId, outcome.hit, { degraded: false, embedMs, t0, lf });
      }
      lf.cache_best_similarity = outcome.bestSimilarity;
    }
    lf.cache_hit = false;

    // 3) Circuit breaker gate — fail fast, no batch-window delay when open.
    const state = this.d.breaker.currentState();
    lf.breaker_state = state;
    if (state === 'open') {
      return this.degrade(requestId, text, vector, new CircuitOpenError(this.d.breaker.retryAfterMs()), {
        embedMs,
        t0,
        lf,
      });
    }
    if (state === 'half_open') {
      const acq = this.d.breaker.tryAcquire();
      if (!acq.allowed) {
        return this.degrade(
          requestId,
          text,
          vector,
          new CircuitOpenError(acq.retryAfterMs ?? cfg.breaker.cooldownMs),
          { embedMs, t0, lf },
        );
      }
      return this.probe(acq, requestId, text, vector, { embedMs, t0, lf });
    }

    // 4) Normal path: dynamic batching.
    try {
      const r = await this.batcher.enqueue({ id: requestId, text });
      if (vector) this.d.cache.insert(text, vector, r.output.result);
      const totalMs = performance.now() - t0;
      metrics.totalLatency.record(totalMs);
      metrics.queueWait.record(r.queueWaitMs);
      metrics.servedByModel.inc();
      Object.assign(lf, {
        source: 'model',
        batch_id: r.batchId,
        batch_size: r.batchSize,
        batch_queue_wait_ms: r.queueWaitMs,
        backend_inference_ms: r.output.backend.inference_ms,
        backend_queue_wait_ms: r.output.backend.queue_wait_ms,
        total_ms: Math.round(totalMs),
        status: 200,
      });
      return {
        status: 200,
        body: {
          id: requestId,
          result: r.output.result,
          meta: {
            source: 'model',
            batch: { id: r.batchId, size: r.batchSize, queue_wait_ms: r.queueWaitMs },
            backend: r.output.backend,
            cache: { hit: false, best_similarity: lf.cache_best_similarity ?? null },
            breaker_state: 'closed',
            latency: { total_ms: Math.round(totalMs), embed_ms: embedMs },
          },
        },
        logFields: lf,
      };
    } catch (err) {
      if (err instanceof QueueFullError) {
        const totalMs = performance.now() - t0;
        metrics.rejected429.inc();
        metrics.totalLatency.record(totalMs);
        Object.assign(lf, { status: 429, error: 'queue_full', total_ms: Math.round(totalMs) });
        return {
          status: 429,
          body: {
            error: 'queue_full',
            message: err.message,
            request_id: requestId,
          },
          headers: { 'retry-after': '1' },
          logFields: lf,
        };
      }
      return this.degrade(requestId, text, vector, err, { embedMs, t0, lf });
    }
  }

  /** Half-open probe: a single-request batch sent directly, bypassing the batcher. */
  private async probe(
    acq: ReturnType<CircuitBreaker['tryAcquire']>,
    requestId: string,
    text: string,
    vector: Float32Array | null,
    ctx: { embedMs: number; t0: number; lf: Record<string, unknown> },
  ): Promise<HandleResult> {
    const { metrics } = this.d;
    const batchId = `probe-${requestId}`;
    const t = performance.now();
    let resp;
    try {
      resp = await this.d.modelClient.inferBatch([{ id: requestId, text }], batchId);
    } catch (err) {
      this.d.breaker.record(acq, { ok: false, latencyMs: performance.now() - t });
      metrics.backendFailures.inc();
      ctx.lf.probe = true;
      return this.degrade(requestId, text, vector, err, ctx);
    }
    const callMs = performance.now() - t;
    this.d.breaker.record(acq, { ok: true, latencyMs: callMs });
    metrics.backendCalls.inc();
    metrics.modelCallLatency.record(callMs);
    metrics.backendInferenceLatency.record(resp.inference_ms);
    metrics.batchSizes.record(1);

    const result = resp.outputs[0]?.result;
    if (!result) {
      return this.degrade(requestId, text, vector, new Error('probe response missing output'), ctx);
    }
    if (vector) this.d.cache.insert(text, vector, result);
    const totalMs = performance.now() - ctx.t0;
    metrics.totalLatency.record(totalMs);
    metrics.servedByModel.inc();
    Object.assign(ctx.lf, {
      source: 'model',
      probe: true,
      batch_id: batchId,
      batch_size: 1,
      backend_inference_ms: resp.inference_ms,
      total_ms: Math.round(totalMs),
      status: 200,
    });
    return {
      status: 200,
      body: {
        id: requestId,
        result,
        meta: {
          source: 'model',
          probe: true,
          batch: { id: batchId, size: 1, queue_wait_ms: 0 },
          backend: { inference_ms: resp.inference_ms, queue_wait_ms: resp.queue_wait_ms, model: resp.model },
          cache: { hit: false, best_similarity: ctx.lf.cache_best_similarity ?? null },
          breaker_state: 'half_open',
          latency: { total_ms: Math.round(totalMs), embed_ms: ctx.embedMs },
        },
      },
      logFields: ctx.lf,
    };
  }

  /**
   * Graceful degradation: when the model is unavailable (circuit open or the
   * batch failed), try the cache once more with the relaxed degraded
   * threshold (optionally accepting stale entries) before returning an error.
   */
  private degrade(
    requestId: string,
    text: string,
    vector: Float32Array | null,
    err: unknown,
    ctx: { embedMs: number; t0: number; lf: Record<string, unknown> },
  ): HandleResult {
    const { cfg, metrics } = this.d;
    const circuitOpen =
      err instanceof CircuitOpenError ||
      (err instanceof BatchError && err.causeErr instanceof CircuitOpenError);
    const retryAfterMs = circuitOpen
      ? err instanceof CircuitOpenError
        ? err.retryAfterMs
        : (err as BatchError & { causeErr: CircuitOpenError }).causeErr.retryAfterMs
      : null;

    if (vector && cfg.cache.enabled) {
      const outcome = this.d.cache.lookup(text, vector, 'degraded');
      if (outcome.hit) {
        metrics.servedDegraded.inc();
        ctx.lf.degraded_reason = circuitOpen ? 'circuit_open' : 'backend_error';
        return this.respondCached(requestId, outcome.hit, {
          degraded: true,
          embedMs: ctx.embedMs,
          t0: ctx.t0,
          lf: ctx.lf,
        });
      }
    }

    const totalMs = performance.now() - ctx.t0;
    metrics.totalLatency.record(totalMs);

    if (circuitOpen) {
      metrics.rejected503.inc();
      Object.assign(ctx.lf, {
        status: 503,
        error: 'circuit_open',
        retry_after_ms: retryAfterMs,
        total_ms: Math.round(totalMs),
      });
      return {
        status: 503,
        body: {
          error: 'circuit_open',
          message:
            'model backend circuit is open (backend failing or too slow) and no sufficiently similar cached response was available',
          retry_after_ms: retryAfterMs,
          request_id: requestId,
        },
        headers: { 'retry-after': String(Math.max(1, Math.ceil((retryAfterMs ?? 1000) / 1000))) },
        logFields: ctx.lf,
      };
    }

    metrics.errors502.inc();
    const detail =
      err instanceof BatchError
        ? err.causeErr instanceof Error
          ? err.causeErr.message
          : String(err.causeErr)
        : err instanceof Error
          ? err.message
          : String(err);
    Object.assign(ctx.lf, {
      status: 502,
      error: 'model_backend_error',
      error_detail: detail,
      batch_id: err instanceof BatchError ? err.batchId : undefined,
      total_ms: Math.round(totalMs),
    });
    return {
      status: 502,
      body: {
        error: 'model_backend_error',
        message: 'model backend call failed and no sufficiently similar cached response was available',
        request_id: requestId,
      },
      logFields: ctx.lf,
    };
  }

  private respondCached(
    requestId: string,
    hit: CacheHit,
    ctx: { degraded: boolean; embedMs: number; t0: number; lf: Record<string, unknown> },
  ): HandleResult {
    const { metrics } = this.d;
    const totalMs = performance.now() - ctx.t0;
    metrics.totalLatency.record(totalMs);
    metrics.servedByCache.inc();
    const source = ctx.degraded ? 'cache_degraded' : 'cache';
    Object.assign(ctx.lf, {
      source,
      cache_hit: true,
      similarity: hit.similarity,
      cache_age_ms: hit.ageMs,
      cache_stale: hit.stale,
      total_ms: Math.round(totalMs),
      status: 200,
    });
    const body: Record<string, unknown> = {
      id: requestId,
      result: hit.response,
      meta: {
        source,
        cache: {
          hit: true,
          similarity: hit.similarity,
          matched_key: hit.matchedKey,
          age_ms: hit.ageMs,
          stale: hit.stale,
        },
        breaker_state: this.d.breaker.currentState(),
        latency: { total_ms: Math.round(totalMs), embed_ms: ctx.embedMs },
      },
    };
    if (ctx.degraded) {
      (body.meta as Record<string, unknown>).warning =
        'degraded response: model backend unavailable, served from semantic cache';
      (body.meta as Record<string, unknown>).degraded_reason = ctx.lf.degraded_reason;
    }
    return { status: 200, body, logFields: ctx.lf };
  }
}
