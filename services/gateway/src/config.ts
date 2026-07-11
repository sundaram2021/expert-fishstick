/**
 * Environment-driven configuration with validation, plus a small runtime
 * override mechanism (used by the /admin/config endpoint so demos and load
 * tests can flip batching/caching/breaker parameters without a restart).
 *
 * Components read configuration through closures over this object, so
 * runtime changes take effect immediately.
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

function floatEnv(name: string, def: number, min = 0, max = 1): number {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return def;
  const v = Number(raw);
  if (!Number.isFinite(v) || v < min || v > max) {
    throw new Error(`Invalid env ${name}=${raw} (expected number in [${min}, ${max}])`);
  }
  return v;
}

function boolEnv(name: string, def: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return def;
  if (raw === 'true' || raw === '1') return true;
  if (raw === 'false' || raw === '0') return false;
  throw new Error(`Invalid env ${name}=${raw} (expected true/false)`);
}

function enumEnv<T extends string>(name: string, allowed: readonly T[], def: T): T {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return def;
  if (!allowed.includes(raw as T)) {
    throw new Error(`Invalid env ${name}=${raw} (expected one of: ${allowed.join(', ')})`);
  }
  return raw as T;
}

export interface GatewayConfig {
  port: number;
  host: string;
  logLevel: string;
  /** Include the observability `meta` block in responses (demo/debug aid). */
  exposeMeta: boolean;
  adminEnabled: boolean;
  modelBackendUrl: string;
  modelTimeoutMs: number;
  batch: {
    /** How long the first request in a batch waits for company. */
    windowMs: number;
    /** Dispatch immediately when a forming batch reaches this size. */
    maxSize: number;
    /** How many batches may be in flight to the backend at once. */
    maxConcurrentBatches: number;
    /** Back-pressure: max requests waiting (forming + sealed); beyond this -> 429. */
    maxQueueDepth: number;
  };
  embedding: {
    provider: 'transformers' | 'hash';
    modelId: string;
    /** Micro-batching window for the embedding forward pass. */
    windowMs: number;
    maxBatchSize: number;
    cacheDir?: string;
    /** If the real embedder fails to load, fall back to the lexical hash embedder instead of crashing. */
    allowFallback: boolean;
  };
  cache: {
    enabled: boolean;
    /** Cosine similarity required to serve a cached response in normal operation. */
    similarityThreshold: number;
    ttlMs: number;
    maxSize: number;
    /** Relaxed threshold used while the circuit is open (degraded mode). */
    degradedThreshold: number;
    /** While the circuit is open, allow serving entries past their TTL (bounded by staleGraceFactor). */
    serveStaleWhenOpen: boolean;
    /** Expired entries are physically removed after ttl * staleGraceFactor. */
    staleGraceFactor: number;
    sweepIntervalMs: number;
  };
  breaker: {
    /** Consecutive backend failures (or latency breaches) that trip the circuit. */
    failureThreshold: number;
    /** A successful call slower than this counts as a failure for tripping purposes. */
    latencyThresholdMs: number;
    /** How long the circuit stays open before probing. */
    cooldownMs: number;
    /** Fraction of requests admitted as probes while half-open. */
    halfOpenRatio: number;
    /** Max concurrent probes while half-open. */
    halfOpenMaxProbes: number;
    /** Consecutive probe successes required to close. */
    halfOpenSuccessesToClose: number;
  };
}

export function loadConfig(): GatewayConfig {
  return {
    port: intEnv('PORT', 8080, 1, 65535),
    host: process.env.HOST ?? '0.0.0.0',
    logLevel: process.env.LOG_LEVEL ?? 'info',
    exposeMeta: boolEnv('EXPOSE_META', true),
    adminEnabled: boolEnv('ADMIN_ENABLED', true),
    modelBackendUrl: process.env.MODEL_BACKEND_URL ?? 'http://localhost:8081',
    modelTimeoutMs: intEnv('MODEL_TIMEOUT_MS', 10_000, 100, 300_000),
    batch: {
      windowMs: intEnv('BATCH_WINDOW_MS', 50, 0, 5_000),
      maxSize: intEnv('BATCH_MAX_SIZE', 32, 1, 256),
      maxConcurrentBatches: intEnv('BATCH_MAX_CONCURRENT', 4, 1, 64),
      maxQueueDepth: intEnv('BATCH_MAX_QUEUE', 500, 1, 100_000),
    },
    embedding: {
      provider: enumEnv('EMBEDDER', ['transformers', 'hash'] as const, 'transformers'),
      modelId: process.env.EMBED_MODEL_ID ?? 'Xenova/all-MiniLM-L6-v2',
      windowMs: intEnv('EMBED_BATCH_WINDOW_MS', 8, 0, 1_000),
      maxBatchSize: intEnv('EMBED_BATCH_MAX_SIZE', 32, 1, 256),
      cacheDir: process.env.TRANSFORMERS_CACHE_DIR,
      allowFallback: boolEnv('EMBED_ALLOW_FALLBACK', false),
    },
    cache: {
      enabled: boolEnv('CACHE_ENABLED', true),
      similarityThreshold: floatEnv('CACHE_SIMILARITY_THRESHOLD', 0.9, 0, 1),
      ttlMs: intEnv('CACHE_TTL_MS', 300_000, 1_000, 86_400_000),
      maxSize: intEnv('CACHE_MAX_SIZE', 1_000, 1, 1_000_000),
      degradedThreshold: floatEnv('CACHE_DEGRADED_THRESHOLD', 0.83, 0, 1),
      serveStaleWhenOpen: boolEnv('CACHE_SERVE_STALE_WHEN_OPEN', true),
      staleGraceFactor: floatEnv('CACHE_STALE_GRACE_FACTOR', 3, 1, 100),
      sweepIntervalMs: intEnv('CACHE_SWEEP_INTERVAL_MS', 30_000, 1_000, 3_600_000),
    },
    breaker: {
      failureThreshold: intEnv('BREAKER_FAILURE_THRESHOLD', 3, 1, 1_000),
      latencyThresholdMs: intEnv('BREAKER_LATENCY_THRESHOLD_MS', 2_000, 1, 300_000),
      cooldownMs: intEnv('BREAKER_COOLDOWN_MS', 10_000, 10, 3_600_000),
      halfOpenRatio: floatEnv('BREAKER_HALF_OPEN_RATIO', 0.25, 0, 1),
      halfOpenMaxProbes: intEnv('BREAKER_HALF_OPEN_MAX_PROBES', 2, 1, 100),
      halfOpenSuccessesToClose: intEnv('BREAKER_HALF_OPEN_SUCCESSES_TO_CLOSE', 2, 1, 100),
    },
  };
}

/** Deep-merge partial overrides into a config (used by tests and buildGateway). */
export type ConfigOverrides = {
  [K in keyof GatewayConfig]?: GatewayConfig[K] extends object ? Partial<GatewayConfig[K]> : GatewayConfig[K];
};

export function mergeConfig(base: GatewayConfig, over: ConfigOverrides): GatewayConfig {
  return {
    ...base,
    ...over,
    batch: { ...base.batch, ...(over.batch ?? {}) },
    embedding: { ...base.embedding, ...(over.embedding ?? {}) },
    cache: { ...base.cache, ...(over.cache ?? {}) },
    breaker: { ...base.breaker, ...(over.breaker ?? {}) },
  } as GatewayConfig;
}

/**
 * Runtime-mutable settings exposed via /admin/config.
 * Maps a flat wire key -> [path, validator]. Returns applied changes.
 */
const RUNTIME_KEYS: Record<
  string,
  { apply: (cfg: GatewayConfig, v: unknown) => void; validate: (v: unknown) => boolean }
> = {
  batch_window_ms: {
    validate: (v) => typeof v === 'number' && v >= 0 && v <= 5_000,
    apply: (c, v) => (c.batch.windowMs = v as number),
  },
  batch_max_size: {
    validate: (v) => typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= 256,
    apply: (c, v) => (c.batch.maxSize = v as number),
  },
  batch_max_concurrent: {
    validate: (v) => typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= 64,
    apply: (c, v) => (c.batch.maxConcurrentBatches = v as number),
  },
  batch_max_queue: {
    validate: (v) => typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= 100_000,
    apply: (c, v) => (c.batch.maxQueueDepth = v as number),
  },
  cache_enabled: {
    validate: (v) => typeof v === 'boolean',
    apply: (c, v) => (c.cache.enabled = v as boolean),
  },
  cache_similarity_threshold: {
    validate: (v) => typeof v === 'number' && v >= 0 && v <= 1,
    apply: (c, v) => (c.cache.similarityThreshold = v as number),
  },
  cache_degraded_threshold: {
    validate: (v) => typeof v === 'number' && v >= 0 && v <= 1,
    apply: (c, v) => (c.cache.degradedThreshold = v as number),
  },
  cache_ttl_ms: {
    validate: (v) => typeof v === 'number' && v >= 1_000 && v <= 86_400_000,
    apply: (c, v) => (c.cache.ttlMs = v as number),
  },
  breaker_failure_threshold: {
    validate: (v) => typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= 1_000,
    apply: (c, v) => (c.breaker.failureThreshold = v as number),
  },
  breaker_latency_threshold_ms: {
    validate: (v) => typeof v === 'number' && v >= 1 && v <= 300_000,
    apply: (c, v) => (c.breaker.latencyThresholdMs = v as number),
  },
  breaker_cooldown_ms: {
    validate: (v) => typeof v === 'number' && v >= 10 && v <= 3_600_000,
    apply: (c, v) => (c.breaker.cooldownMs = v as number),
  },
  model_timeout_ms: {
    validate: (v) => typeof v === 'number' && v >= 100 && v <= 300_000,
    apply: (c, v) => (c.modelTimeoutMs = v as number),
  },
};

export function applyRuntimeConfig(
  cfg: GatewayConfig,
  patch: Record<string, unknown>,
): { applied: Record<string, unknown>; rejected: Record<string, string> } {
  const applied: Record<string, unknown> = {};
  const rejected: Record<string, string> = {};
  for (const [key, value] of Object.entries(patch)) {
    const spec = RUNTIME_KEYS[key];
    if (!spec) {
      rejected[key] = 'unknown_key';
      continue;
    }
    if (!spec.validate(value)) {
      rejected[key] = 'invalid_value';
      continue;
    }
    spec.apply(cfg, value);
    applied[key] = value;
  }
  return { applied, rejected };
}
