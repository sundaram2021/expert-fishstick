/**
 * A GPU-realistic inference stub.
 *
 * Latency model: cost(batch) = base + perItem * batchSize + gaussianJitter
 *
 * This is the cost shape of real accelerator inference: a large fixed cost
 * per forward pass (kernel launches, memory movement, attention setup) plus a
 * comparatively small marginal cost per item in the batch. It is exactly this
 * shape that makes dynamic batching profitable — 16 requests in one pass cost
 * far less than 16 passes.
 *
 * Execution is serialized through a semaphore (default concurrency 1) to
 * mimic a single device: concurrent calls queue, as they would on real
 * hardware.
 *
 * Outputs are a deterministic lexicon-based sentiment classification so that
 * repeated/paraphrased requests are stable, which makes semantic-cache
 * verification meaningful.
 */
import { Semaphore } from './semaphore.js';
import type { FaultInjector } from './faults.js';

export interface InferenceInput {
  id: string;
  text: string;
}

export interface ModelResult {
  label: 'positive' | 'negative' | 'neutral';
  score: number;
  model: string;
  tokens: number;
}

export interface BatchOutput {
  outputs: Array<{ id: string; result: ModelResult }>;
  inferenceMs: number;
  queueWaitMs: number;
}

const POSITIVE = new Set([
  'good', 'great', 'love', 'excellent', 'amazing', 'happy', 'wonderful', 'fantastic',
  'perfect', 'best', 'awesome', 'nice', 'thanks', 'thank', 'helpful', 'fast', 'easy', 'works',
]);
const NEGATIVE = new Set([
  'bad', 'terrible', 'hate', 'awful', 'horrible', 'angry', 'broken', 'damaged', 'slow',
  'worst', 'refund', 'cancel', 'failed', 'error', 'crash', 'crashes', 'frustrated',
  'disappointed', 'useless', 'never', 'late',
]);

export function fnv1a(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Box–Muller transform: standard normal sample from a uniform RNG. */
export function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export interface StubOptions {
  baseMs: number;
  perItemMs: number;
  jitterStdMs: number;
  concurrency: number;
}

export class StubModel {
  readonly name = 'stub-sentiment-v1';
  private device: Semaphore;

  constructor(
    private cfg: StubOptions,
    private faults: FaultInjector,
    private rng: () => number = Math.random,
  ) {
    this.device = new Semaphore(cfg.concurrency);
  }

  get queueDepth(): number {
    return this.device.queueDepth;
  }

  computeMs(batchSize: number): number {
    const jitter = gaussian(this.rng) * this.cfg.jitterStdMs;
    return Math.max(2, Math.round(this.cfg.baseMs + this.cfg.perItemMs * batchSize + jitter));
  }

  classify(text: string): ModelResult {
    const words = text.toLowerCase().split(/[^a-z']+/).filter(Boolean);
    let pos = 0;
    let neg = 0;
    for (const w of words) {
      if (POSITIVE.has(w)) pos++;
      if (NEGATIVE.has(w)) neg++;
    }
    // Deterministic per-text texture so distinct texts get distinct scores.
    const jitter = ((fnv1a(text) % 1000) / 1000 - 0.5) * 0.08;
    let label: ModelResult['label'];
    let score: number;
    if (pos === 0 && neg === 0) {
      label = 'neutral';
      score = 0.5 + jitter;
    } else if (pos >= neg) {
      label = 'positive';
      score = 0.6 + 0.35 * (pos / (pos + neg)) + jitter;
    } else {
      label = 'negative';
      score = 0.6 + 0.35 * (neg / (pos + neg)) + jitter;
    }
    return {
      label,
      score: Math.round(clamp(score, 0.05, 0.99) * 1000) / 1000,
      model: this.name,
      tokens: words.length,
    };
  }

  async inferBatch(inputs: InferenceInput[]): Promise<BatchOutput> {
    const queuedAt = performance.now();
    await this.device.acquire();
    const queueWaitMs = Math.round(performance.now() - queuedAt);
    try {
      this.faults.maybeFail(this.rng);
      const inferenceMs = this.computeMs(inputs.length) + this.faults.extraLatencyMsNow();
      await sleep(inferenceMs);
      return {
        outputs: inputs.map((i) => ({ id: i.id, result: this.classify(i.text) })),
        inferenceMs,
        queueWaitMs,
      };
    } finally {
      this.device.release();
    }
  }
}
