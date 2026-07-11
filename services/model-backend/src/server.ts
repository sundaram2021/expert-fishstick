import Fastify, { type FastifyInstance } from 'fastify';
import { loadConfig, type BackendConfig } from './config.js';
import { createLogger, type Logger } from './logger.js';
import { FaultInjector, type FaultMode } from './faults.js';
import { StubModel, type InferenceInput } from './stubModel.js';
import { RealModel } from './realModel.js';

export interface BuiltBackend {
  app: FastifyInstance;
  cfg: BackendConfig;
  log: Logger;
  faults: FaultInjector;
  modelName: string;
}

interface InferBody {
  batch_id?: string;
  inputs: InferenceInput[];
}

export async function buildBackend(overrides: Partial<BackendConfig> = {}): Promise<BuiltBackend> {
  const base = loadConfig();
  const cfg: BackendConfig = {
    ...base,
    ...overrides,
    stub: { ...base.stub, ...(overrides.stub ?? {}) },
    real: { ...base.real, ...(overrides.real ?? {}) },
  };
  const log = createLogger(cfg.logLevel, 'model-backend');
  const faults = new FaultInjector();

  let model: StubModel | RealModel;
  if (cfg.mode === 'real') {
    const real = new RealModel(cfg.real.modelId, faults);
    log.info({ model: cfg.real.modelId }, 'model.loading');
    await real.init();
    log.info({ model: cfg.real.modelId }, 'model.ready');
    model = real;
  } else {
    model = new StubModel(cfg.stub, faults);
    log.info({ model: model.name, ...cfg.stub }, 'model.ready (stub)');
  }

  const app = Fastify({ logger: false, bodyLimit: 8 * 1024 * 1024 });

  const stats = { batches: 0, items: 0, failures: 0 };

  app.post(
    '/infer',
    {
      schema: {
        body: {
          type: 'object',
          required: ['inputs'],
          additionalProperties: false,
          properties: {
            batch_id: { type: 'string', maxLength: 128 },
            inputs: {
              type: 'array',
              minItems: 1,
              maxItems: cfg.maxBatchSize,
              items: {
                type: 'object',
                required: ['id', 'text'],
                additionalProperties: false,
                properties: {
                  id: { type: 'string', minLength: 1, maxLength: 128 },
                  text: { type: 'string', minLength: 1, maxLength: cfg.maxTextLen },
                },
              },
            },
          },
        },
      },
    },
    async (req, reply) => {
      const body = req.body as InferBody;
      const t0 = performance.now();
      try {
        const out = await model.inferBatch(body.inputs);
        stats.batches++;
        stats.items += body.inputs.length;
        log.info(
          {
            batch_id: body.batch_id ?? null,
            batch_size: body.inputs.length,
            queue_wait_ms: out.queueWaitMs,
            inference_ms: out.inferenceMs,
            total_ms: Math.round(performance.now() - t0),
          },
          'batch.served',
        );
        return {
          batch_id: body.batch_id ?? null,
          model: model.name,
          batch_size: body.inputs.length,
          inference_ms: out.inferenceMs,
          queue_wait_ms: out.queueWaitMs,
          outputs: out.outputs,
        };
      } catch (err) {
        stats.failures++;
        const message = err instanceof Error ? err.message : String(err);
        log.error(
          { batch_id: body.batch_id ?? null, batch_size: body.inputs.length, err: message },
          'batch.failed',
        );
        return reply.code(500).send({ error: 'inference_failed', message });
      }
    },
  );

  app.get('/healthz', async () => ({
    status: 'ok',
    model: model.name,
    mode: cfg.mode,
    fault: faults.current(),
    stats,
  }));

  // ---- Fault-injection admin API (used by the load test / demos) ----
  app.post('/admin/fault', async (req, reply) => {
    const body = (req.body ?? {}) as {
      mode?: FaultMode;
      error_rate?: number;
      extra_latency_ms?: number;
      duration_ms?: number;
    };
    const mode = body.mode ?? 'none';
    if (!['none', 'error', 'slow'].includes(mode)) {
      return reply.code(400).send({ error: 'bad_request', message: `unknown fault mode: ${mode}` });
    }
    const state = faults.set(mode, {
      errorRate: body.error_rate,
      extraLatencyMs: body.extra_latency_ms,
      durationMs: body.duration_ms,
    });
    log.warn({ fault: state }, 'fault.set');
    return state;
  });

  app.get('/admin/fault', async () => faults.current());

  return { app, cfg, log, faults, modelName: model.name };
}
