/**
 * Environment-driven configuration with validation.
 * Every tunable of the backend is an env var so the whole system can be
 * reconfigured from docker-compose without touching code.
 */

function intEnv(name: string, def: number, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return def;
  const v = Number(raw);
  if (!Number.isFinite(v) || v < min || v > max) {
    throw new Error(`Invalid env ${name}=${raw} (expected number in [${min}, ${max}])`);
  }
  return v;
}

function enumEnv<T extends string>(name: string, allowed: readonly T[], def: T): T {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return def;
  if (!allowed.includes(raw as T)) {
    throw new Error(`Invalid env ${name}=${raw} (expected one of: ${allowed.join(', ')})`);
  }
  return raw as T;
}

export interface BackendConfig {
  port: number;
  host: string;
  logLevel: string;
  /** 'stub' simulates a GPU with configurable latency; 'real' runs an ONNX DistilBERT classifier. */
  mode: 'stub' | 'real';
  stub: {
    /** Fixed per-batch cost (kernel launch / memory movement equivalent). */
    baseMs: number;
    /** Marginal cost of each additional item in a batch. */
    perItemMs: number;
    /** Std-dev of gaussian jitter added to each batch, simulating real variance. */
    jitterStdMs: number;
    /** How many batches the simulated device can execute at once (a single GPU = 1). */
    concurrency: number;
  };
  real: {
    modelId: string;
  };
  maxBatchSize: number;
  maxTextLen: number;
}

export function loadConfig(): BackendConfig {
  return {
    port: intEnv('PORT', 8081, 1, 65535),
    host: process.env.HOST ?? '0.0.0.0',
    logLevel: process.env.LOG_LEVEL ?? 'info',
    mode: enumEnv('MODEL_MODE', ['stub', 'real'] as const, 'stub'),
    stub: {
      baseMs: intEnv('STUB_BASE_MS', 80, 0, 60_000),
      perItemMs: intEnv('STUB_PER_ITEM_MS', 6, 0, 10_000),
      jitterStdMs: intEnv('STUB_JITTER_STD_MS', 12, 0, 10_000),
      concurrency: intEnv('STUB_CONCURRENCY', 1, 1, 64),
    },
    real: {
      modelId: process.env.REAL_MODEL_ID ?? 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
    },
    maxBatchSize: intEnv('MAX_BATCH_SIZE', 256, 1, 4096),
    maxTextLen: intEnv('MAX_TEXT_LEN', 8192, 1, 1_000_000),
  };
}
