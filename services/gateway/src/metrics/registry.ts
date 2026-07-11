/**
 * Hand-rolled in-process metrics: counters, latency percentile trackers,
 * batch-size distributions and sliding-window rates.
 *
 * Percentiles use a fixed-capacity ring buffer of recent samples (a simple
 * reservoir over the last N observations). Sorting ~4k floats on scrape costs
 * microseconds and keeps the write path allocation-free — the right tradeoff
 * for a single-process gateway. (At bigger scale you would move to t-digest /
 * HDR histograms and a Prometheus scrape; see README.)
 */

const round = (v: number, digits: number): number => {
  const f = 10 ** digits;
  return Math.round(v * f) / f;
};

export class Counter {
  private v = 0;
  inc(n = 1): void {
    this.v += n;
  }
  get value(): number {
    return this.v;
  }
  reset(): void {
    this.v = 0;
  }
}

export interface LatencySnapshot {
  count: number;
  avg: number | null;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  max: number | null;
}

export class LatencyTracker {
  private buf: Float64Array;
  private idx = 0;
  private filled = 0;
  private _count = 0;
  private _sum = 0;
  private _max = 0;

  constructor(capacity = 4096) {
    this.buf = new Float64Array(capacity);
  }

  record(ms: number): void {
    this.buf[this.idx] = ms;
    this.idx = (this.idx + 1) % this.buf.length;
    if (this.filled < this.buf.length) this.filled++;
    this._count++;
    this._sum += ms;
    if (ms > this._max) this._max = ms;
  }

  snapshot(): LatencySnapshot {
    if (this._count === 0) {
      return { count: 0, avg: null, p50: null, p95: null, p99: null, max: null };
    }
    const arr = Array.from(this.buf.subarray(0, this.filled)).sort((a, b) => a - b);
    const q = (p: number): number =>
      arr[Math.min(arr.length - 1, Math.max(0, Math.ceil(p * arr.length) - 1))] as number;
    return {
      count: this._count,
      avg: round(this._sum / this._count, 1),
      p50: round(q(0.5), 1),
      p95: round(q(0.95), 1),
      p99: round(q(0.99), 1),
      max: round(this._max, 1),
    };
  }

  reset(): void {
    this.idx = 0;
    this.filled = 0;
    this._count = 0;
    this._sum = 0;
    this._max = 0;
  }
}

/** Exact-count distribution over small integer values (batch sizes). */
export class SizeDistribution {
  private counts = new Map<number, number>();
  private _sum = 0;
  private _n = 0;

  record(size: number): void {
    this.counts.set(size, (this.counts.get(size) ?? 0) + 1);
    this._sum += size;
    this._n++;
  }

  snapshot(): { batches: number; avg_batch_size: number | null; distribution: Record<string, number> } {
    const distribution: Record<string, number> = {};
    for (const [size, count] of [...this.counts.entries()].sort((a, b) => a[0] - b[0])) {
      distribution[String(size)] = count;
    }
    return {
      batches: this._n,
      avg_batch_size: this._n ? round(this._sum / this._n, 2) : null,
      distribution,
    };
  }

  reset(): void {
    this.counts.clear();
    this._sum = 0;
    this._n = 0;
  }
}

/** Sliding-window request rate using per-second buckets. */
export class RateWindow {
  private buckets = new Float64Array(120);
  private stamps = new Int32Array(120);

  record(n = 1, now = Date.now()): void {
    const sec = Math.floor(now / 1000);
    const i = sec % this.buckets.length;
    if (this.stamps[i] !== sec) {
      this.stamps[i] = sec;
      this.buckets[i] = 0;
    }
    this.buckets[i] = (this.buckets[i] as number) + n;
  }

  /** Average events/sec over the trailing `windowSec` full seconds. */
  ratePerSec(windowSec: number, now = Date.now()): number {
    const sec = Math.floor(now / 1000);
    let total = 0;
    for (let k = 1; k <= windowSec; k++) {
      const s = sec - k;
      const i = ((s % this.buckets.length) + this.buckets.length) % this.buckets.length;
      if (this.stamps[i] === s) total += this.buckets[i] as number;
    }
    return round(total / windowSec, 2);
  }

  reset(): void {
    this.buckets.fill(0);
    this.stamps.fill(0);
  }
}

/** All gateway metrics in one place, snapshotted by the /metrics endpoint. */
export class GatewayMetrics {
  readonly startedAt = Date.now();
  inFlight = 0;

  readonly requests = new Counter();
  readonly rate = new RateWindow();

  readonly servedByModel = new Counter();
  readonly servedByCache = new Counter();
  readonly servedDegraded = new Counter();

  readonly rejected429 = new Counter();
  readonly rejected503 = new Counter();
  readonly errors502 = new Counter();

  readonly totalLatency = new LatencyTracker();
  readonly modelCallLatency = new LatencyTracker();
  readonly backendInferenceLatency = new LatencyTracker();
  readonly queueWait = new LatencyTracker();

  readonly batchSizes = new SizeDistribution();
  readonly queueRejections = new Counter();

  readonly backendCalls = new Counter();
  readonly backendFailures = new Counter();

  resetAll(): void {
    this.requests.reset();
    this.rate.reset();
    this.servedByModel.reset();
    this.servedByCache.reset();
    this.servedDegraded.reset();
    this.rejected429.reset();
    this.rejected503.reset();
    this.errors502.reset();
    this.totalLatency.reset();
    this.modelCallLatency.reset();
    this.backendInferenceLatency.reset();
    this.queueWait.reset();
    this.batchSizes.reset();
    this.queueRejections.reset();
    this.backendCalls.reset();
    this.backendFailures.reset();
  }
}
