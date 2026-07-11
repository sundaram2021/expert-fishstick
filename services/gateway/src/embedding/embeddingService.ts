/**
 * EmbeddingService wraps an Embedder behind the same DynamicBatcher used for
 * model calls (smaller window): concurrent requests are embedded in one
 * forward pass. Without this, per-request embedding becomes the serving
 * bottleneck under load — batching applies at *every* inference stage, not
 * just the primary model.
 */
import { DynamicBatcher } from '../batching/batcher.js';
import { LatencyTracker } from '../metrics/registry.js';
import type { GatewayConfig } from '../config.js';
import type { Logger } from '../logger.js';
import { HashEmbedder, type Embedder } from './embedder.js';
import { TransformersEmbedder } from './transformersEmbedder.js';

export class EmbeddingService {
  readonly latency = new LatencyTracker(4096);
  private batcher: DynamicBatcher<string, Float32Array>;
  private _ready = false;

  constructor(
    private readonly embedder: Embedder,
    opts: () => { windowMs: number; maxBatchSize: number },
  ) {
    this.batcher = new DynamicBatcher<string, Float32Array>(
      (texts) => this.embedder.embed(texts),
      () => ({
        windowMs: opts().windowMs,
        maxBatchSize: opts().maxBatchSize,
        maxConcurrentBatches: 2,
        maxQueueDepth: 10_000,
      }),
    );
  }

  get ready(): boolean {
    return this._ready;
  }

  get name(): string {
    return this.embedder.name;
  }

  get dims(): number {
    return this.embedder.dims;
  }

  async init(): Promise<void> {
    await this.embedder.init();
    this._ready = true;
  }

  async embedOne(text: string): Promise<{ vector: Float32Array; ms: number }> {
    const t0 = performance.now();
    const r = await this.batcher.enqueue(text);
    const ms = performance.now() - t0;
    this.latency.record(ms);
    return { vector: r.output, ms: Math.round(ms) };
  }
}

export async function createEmbeddingService(
  cfg: GatewayConfig,
  log: Logger,
): Promise<EmbeddingService> {
  const mk = (e: Embedder) => new EmbeddingService(e, () => cfg.embedding);

  if (cfg.embedding.provider === 'hash') {
    const svc = mk(new HashEmbedder());
    await svc.init();
    log.warn({ embedder: svc.name }, 'embedder.ready (lexical hash fallback — cache is NOT semantic)');
    return svc;
  }

  const svc = mk(new TransformersEmbedder(cfg.embedding.modelId, cfg.embedding.cacheDir));
  try {
    const t0 = Date.now();
    await svc.init();
    log.info({ embedder: svc.name, load_ms: Date.now() - t0 }, 'embedder.ready');
    return svc;
  } catch (err) {
    if (!cfg.embedding.allowFallback) throw err;
    log.error(
      { err: err instanceof Error ? err.message : String(err) },
      'embedder.init_failed — falling back to lexical hash embedder (semantic quality degraded)',
    );
    const fb = mk(new HashEmbedder());
    await fb.init();
    return fb;
  }
}
