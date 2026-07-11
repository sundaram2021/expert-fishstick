import Fastify, { type FastifyInstance } from 'fastify';
import { CircuitBreaker } from './breaker/circuitBreaker.js';
import { SemanticCache } from './cache/semanticCache.js';
import {
  applyRuntimeConfig,
  loadConfig,
  mergeConfig,
  type ConfigOverrides,
  type GatewayConfig,
} from './config.js';
import { createEmbeddingService, EmbeddingService } from './embedding/embeddingService.js';
import { createLogger, type Logger } from './logger.js';
import { GatewayMetrics } from './metrics/registry.js';
import { ModelClient } from './modelClient.js';
import { InferencePipeline } from './pipeline.js';

export interface BuiltGateway {
  app: FastifyInstance;
  cfg: GatewayConfig;
  log: Logger;
  metrics: GatewayMetrics;
  cache: SemanticCache;
  breaker: CircuitBreaker;
  pipeline: InferencePipeline;
  embeddings: EmbeddingService;
}

function genRequestId(): string {
  return `req_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export async function buildGateway(
  overrides: ConfigOverrides = {},
  injected: { embeddings?: EmbeddingService } = {},
): Promise<BuiltGateway> {
  const cfg = mergeConfig(loadConfig(), overrides);
  const log = createLogger(cfg.logLevel, 'gateway');
  const metrics = new GatewayMetrics();

  const embeddings = injected.embeddings ?? (await createEmbeddingService(cfg, log));
  const cache = new SemanticCache(() => cfg.cache);
  const breaker = new CircuitBreaker(() => cfg.breaker, {
    onTransition: (t) =>
      log.warn(
        { from: t.from, to: t.to, reason: t.reason },
        'breaker.transition',
      ),
  });
  const modelClient = new ModelClient(() => ({
    baseUrl: cfg.modelBackendUrl,
    timeoutMs: cfg.modelTimeoutMs,
  }));
  const pipeline = new InferencePipeline({ cfg, log, embeddings, cache, breaker, modelClient, metrics });

  const app = Fastify({ logger: false, bodyLimit: 1024 * 1024 });

  app.post(
    '/infer',
    {
      schema: {
        body: {
          type: 'object',
          required: ['text'],
          additionalProperties: false,
          properties: {
            text: { type: 'string', minLength: 1, maxLength: 8192 },
          },
        },
      },
    },
    async (req, reply) => {
      const { text } = req.body as { text: string };
      const headerId = req.headers['x-request-id'];
      const requestId =
        typeof headerId === 'string' && headerId.length > 0 ? headerId.slice(0, 64) : genRequestId();
      metrics.inFlight++;
      try {
        const res = await pipeline.handle(text, requestId);
        log.info(res.logFields, 'request.completed');
        reply.header('x-request-id', requestId);
        if (res.headers) {
          for (const [k, v] of Object.entries(res.headers)) reply.header(k, v);
        }
        reply.code(res.status);
        if (!cfg.exposeMeta && res.body && typeof res.body === 'object') {
          const { meta: _meta, ...rest } = res.body as Record<string, unknown>;
          return rest;
        }
        return res.body;
      } finally {
        metrics.inFlight--;
      }
    },
  );

  app.get('/metrics', async () => {
    const now = Date.now();
    return {
      service: 'inference-gateway',
      now: new Date(now).toISOString(),
      uptime_s: Math.round((now - metrics.startedAt) / 1000),
      requests: {
        total: metrics.requests.value,
        in_flight: metrics.inFlight,
        rate_10s_per_s: metrics.rate.ratePerSec(10, now),
        rate_60s_per_s: metrics.rate.ratePerSec(60, now),
        served_by: {
          model: metrics.servedByModel.value,
          cache: metrics.servedByCache.value,
          cache_degraded: metrics.servedDegraded.value,
        },
        errors: {
          '429_queue_full': metrics.rejected429.value,
          '502_backend_error': metrics.errors502.value,
          '503_circuit_open': metrics.rejected503.value,
        },
      },
      latency_ms: {
        total_request: metrics.totalLatency.snapshot(),
        model_backend_call: metrics.modelCallLatency.snapshot(),
        model_backend_reported_inference: metrics.backendInferenceLatency.snapshot(),
        embedding: embeddings.latency.snapshot(),
        batch_queue_wait: metrics.queueWait.snapshot(),
      },
      batching: {
        ...metrics.batchSizes.snapshot(),
        live: pipeline.batcher.stats(),
        rejected_queue_full: metrics.queueRejections.value,
        config: {
          window_ms: cfg.batch.windowMs,
          max_batch_size: cfg.batch.maxSize,
          max_concurrent_batches: cfg.batch.maxConcurrentBatches,
          max_queue_depth: cfg.batch.maxQueueDepth,
        },
      },
      semantic_cache: cache.snapshot(),
      circuit_breaker: breaker.snapshot(),
      model_backend: {
        url: cfg.modelBackendUrl,
        calls: metrics.backendCalls.value,
        failures: metrics.backendFailures.value,
        timeout_ms: cfg.modelTimeoutMs,
      },
      embedder: { name: embeddings.name, dims: embeddings.dims, ready: embeddings.ready },
    };
  });

  app.get('/healthz', async () => ({
    status: 'ok',
    embedder_ready: embeddings.ready,
    breaker_state: breaker.currentState(),
  }));

  if (cfg.adminEnabled) {
    app.get('/admin/config', async () => cfg);

    app.post('/admin/config', async (req, reply) => {
      const patch = (req.body ?? {}) as Record<string, unknown>;
      const { applied, rejected } = applyRuntimeConfig(cfg, patch);
      if (Object.keys(applied).length > 0) log.warn({ applied }, 'admin.config_changed');
      if (Object.keys(rejected).length > 0 && Object.keys(applied).length === 0) {
        return reply.code(400).send({ applied, rejected });
      }
      return { applied, rejected };
    });

    app.post('/admin/metrics/reset', async () => {
      metrics.resetAll();
      cache.resetStats();
      breaker.resetCounters();
      log.warn('admin.metrics_reset');
      return { ok: true };
    });

    app.post('/admin/cache/clear', async () => {
      cache.clear();
      log.warn('admin.cache_cleared');
      return { ok: true };
    });
  }

  const sweepTimer = setInterval(() => {
    const removed = cache.sweep();
    if (removed > 0) log.debug({ removed }, 'cache.sweep');
  }, cfg.cache.sweepIntervalMs);
  sweepTimer.unref();
  app.addHook('onClose', async () => clearInterval(sweepTimer));

  return { app, cfg, log, metrics, cache, breaker, pipeline, embeddings };
}
