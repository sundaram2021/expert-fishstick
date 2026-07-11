/**
 * Dynamic micro-batcher.
 *
 * Semantics:
 *  - The first request entering an empty queue opens a batching window
 *    (windowMs). When the window closes, everything queued is sealed into a
 *    batch and dispatched.
 *  - If the forming batch reaches maxBatchSize before the window closes, it
 *    is sealed and dispatched immediately (the window does not delay a full
 *    batch).
 *  - Sealed batches wait in a FIFO if maxConcurrentBatches are already in
 *    flight — new requests keep forming the *next* batch meanwhile. A request
 *    that arrives while a batch is mid-flight simply joins the next forming
 *    batch; it never joins an in-flight one.
 *  - Back-pressure: when the number of waiting requests (forming + sealed)
 *    reaches maxQueueDepth, new requests are rejected with QueueFullError
 *    (surfaced as HTTP 429) instead of growing an unbounded queue.
 *
 * Every member of a batch receives its own resolution: its individual output
 * on success, or a BatchError on failure. Callers never observe other
 * members' data.
 *
 * The same class powers both model-call batching and embedding micro-batching.
 */
import { BatchError, QueueFullError } from '../errors.js';

export interface BatcherOptions {
  windowMs: number;
  maxBatchSize: number;
  maxConcurrentBatches: number;
  maxQueueDepth: number;
}

export interface BatchItemResult<TOut> {
  output: TOut;
  batchId: string;
  batchSize: number;
  /** Time this item spent waiting between enqueue and dispatch. */
  queueWaitMs: number;
}

export interface BatcherHooks {
  onDispatch?: (batchId: string, size: number) => void;
  onSettle?: (batchId: string, size: number, ok: boolean, ms: number) => void;
  onReject?: () => void;
}

interface Pending<TIn, TOut> {
  input: TIn;
  enqueuedAt: number;
  resolve: (r: BatchItemResult<TOut>) => void;
  reject: (e: unknown) => void;
}

export type BatchHandler<TIn, TOut> = (inputs: TIn[], batchId: string) => Promise<TOut[]>;

export class DynamicBatcher<TIn, TOut> {
  private queue: Array<Pending<TIn, TOut>> = [];
  private ready: Array<Array<Pending<TIn, TOut>>> = [];
  private timer: NodeJS.Timeout | null = null;
  private inFlight = 0;
  private seq = 0;

  constructor(
    private readonly handler: BatchHandler<TIn, TOut>,
    private readonly opts: () => BatcherOptions,
    private readonly hooks: BatcherHooks = {},
  ) {}

  stats(): { queueDepth: number; readyBatches: number; inFlight: number } {
    return {
      queueDepth: this.queue.length + this.ready.reduce((a, b) => a + b.length, 0),
      readyBatches: this.ready.length,
      inFlight: this.inFlight,
    };
  }

  enqueue(input: TIn): Promise<BatchItemResult<TOut>> {
    const o = this.opts();
    const waiting = this.queue.length + this.ready.reduce((a, b) => a + b.length, 0);
    if (waiting >= o.maxQueueDepth) {
      this.hooks.onReject?.();
      return Promise.reject(
        new QueueFullError(`request queue is full (${waiting}/${o.maxQueueDepth} waiting)`),
      );
    }
    return new Promise<BatchItemResult<TOut>>((resolve, reject) => {
      this.queue.push({ input, enqueuedAt: Date.now(), resolve, reject });
      if (this.queue.length >= o.maxBatchSize) {
        // Full batch: dispatch immediately, don't wait out the window.
        this.seal();
      } else if (!this.timer) {
        // First request of a new batch opens the window.
        this.timer = setTimeout(() => this.seal(), o.windowMs);
      }
    });
  }

  /** Seal everything currently queued into ready batches (chunked to maxBatchSize). */
  private seal(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const o = this.opts();
    while (this.queue.length > 0) {
      this.ready.push(this.queue.splice(0, Math.max(1, o.maxBatchSize)));
    }
    this.pump();
  }

  /** Dispatch ready batches while concurrency slots are available. */
  private pump(): void {
    const o = this.opts();
    while (this.inFlight < o.maxConcurrentBatches && this.ready.length > 0) {
      const batch = this.ready.shift();
      if (!batch || batch.length === 0) continue;
      this.inFlight++;
      void this.dispatch(batch).finally(() => {
        this.inFlight--;
        this.pump();
      });
    }
  }

  private async dispatch(batch: Array<Pending<TIn, TOut>>): Promise<void> {
    const batchId = `b${(++this.seq).toString(36)}-${Date.now().toString(36)}`;
    const size = batch.length;
    const dispatchedAt = Date.now();
    this.hooks.onDispatch?.(batchId, size);
    try {
      const outputs = await this.handler(
        batch.map((p) => p.input),
        batchId,
      );
      if (!Array.isArray(outputs) || outputs.length !== size) {
        throw new Error(
          `batch handler returned ${Array.isArray(outputs) ? outputs.length : typeof outputs} outputs for a batch of ${size}`,
        );
      }
      for (let i = 0; i < size; i++) {
        const p = batch[i] as Pending<TIn, TOut>;
        p.resolve({
          output: outputs[i] as TOut,
          batchId,
          batchSize: size,
          queueWaitMs: dispatchedAt - p.enqueuedAt,
        });
      }
      this.hooks.onSettle?.(batchId, size, true, Date.now() - dispatchedAt);
    } catch (err) {
      for (const p of batch) {
        p.reject(new BatchError(`batch ${batchId} (size ${size}) failed`, batchId, size, err));
      }
      this.hooks.onSettle?.(batchId, size, false, Date.now() - dispatchedAt);
    }
  }
}
