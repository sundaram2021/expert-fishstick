import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DynamicBatcher, type BatcherOptions } from '../src/batching/batcher.js';
import { BatchError, QueueFullError } from '../src/errors.js';

const opts = (over: Partial<BatcherOptions> = {}): (() => BatcherOptions) => {
  const o: BatcherOptions = {
    windowMs: 50,
    maxBatchSize: 4,
    maxConcurrentBatches: 4,
    maxQueueDepth: 100,
    ...over,
  };
  return () => o;
};

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('DynamicBatcher', () => {
  it('batches requests arriving within the window into one dispatch', async () => {
    const calls: string[][] = [];
    const b = new DynamicBatcher<string, string>(async (inputs) => {
      calls.push(inputs);
      return inputs.map((i) => `out:${i}`);
    }, opts({ windowMs: 50, maxBatchSize: 10 }));

    const p1 = b.enqueue('a');
    const p2 = b.enqueue('b');
    const p3 = b.enqueue('c');
    await vi.advanceTimersByTimeAsync(49);
    expect(calls).toHaveLength(0); // window still open
    await vi.advanceTimersByTimeAsync(2);
    const [r1, r2, r3] = await Promise.all([p1, p2, p3]);

    expect(calls).toEqual([['a', 'b', 'c']]);
    expect(r1.output).toBe('out:a');
    expect(r2.output).toBe('out:b');
    expect(r3.output).toBe('out:c');
    expect(new Set([r1.batchId, r2.batchId, r3.batchId]).size).toBe(1);
    expect(r1.batchSize).toBe(3);
  });

  it('dispatches immediately when the batch is full, without waiting out the window', async () => {
    const calls: string[][] = [];
    const b = new DynamicBatcher<string, string>(async (inputs) => {
      calls.push(inputs);
      return inputs.map((i) => `out:${i}`);
    }, opts({ windowMs: 10_000, maxBatchSize: 2 }));

    const p1 = b.enqueue('a');
    const p2 = b.enqueue('b');
    // no timer advance at all — full batch must not wait for the window
    await vi.advanceTimersByTimeAsync(0);
    await Promise.all([p1, p2]);
    expect(calls).toEqual([['a', 'b']]);
  });

  it('a request arriving after a batch seals joins the NEXT batch (mid-flight isolation)', async () => {
    const resolvers: Array<(v: string[]) => void> = [];
    const calls: string[][] = [];
    const b = new DynamicBatcher<string, string>(
      (inputs) =>
        new Promise((resolve) => {
          calls.push(inputs);
          resolvers.push((outs) => resolve(outs));
        }),
      opts({ windowMs: 30, maxBatchSize: 2 }),
    );

    const p1 = b.enqueue('a');
    const p2 = b.enqueue('b'); // seals batch 1 (full), dispatches immediately
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toEqual([['a', 'b']]);

    const p3 = b.enqueue('c'); // batch 1 is mid-flight: c starts batch 2's window
    await vi.advanceTimersByTimeAsync(31);
    expect(calls).toEqual([['a', 'b'], ['c']]);

    resolvers[0]?.(['out:a', 'out:b']);
    resolvers[1]?.(['out:c']);
    const [r1, r2, r3] = await Promise.all([p1, p2, p3]);
    expect(r3.batchId).not.toBe(r1.batchId);
    expect(r2.output).toBe('out:b');
    expect(r3.output).toBe('out:c');
  });

  it('respects maxConcurrentBatches: sealed batches wait for a slot', async () => {
    const resolvers: Array<(v: string[]) => void> = [];
    const calls: string[][] = [];
    const b = new DynamicBatcher<string, string>(
      (inputs) =>
        new Promise((resolve) => {
          calls.push(inputs);
          resolvers.push((outs) => resolve(outs));
        }),
      opts({ windowMs: 10, maxBatchSize: 2, maxConcurrentBatches: 1 }),
    );

    const p1 = b.enqueue('a');
    const p2 = b.enqueue('b'); // batch 1 dispatches (slot taken)
    const p3 = b.enqueue('c');
    const p4 = b.enqueue('d'); // batch 2 sealed (full) but must wait
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toHaveLength(1);
    expect(b.stats().readyBatches).toBe(1);

    resolvers[0]?.(['out:a', 'out:b']); // free the slot
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toHaveLength(2);
    resolvers[1]?.(['out:c', 'out:d']);
    const rs = await Promise.all([p1, p2, p3, p4]);
    expect(rs.map((r) => r.output)).toEqual(['out:a', 'out:b', 'out:c', 'out:d']);
  });

  it('applies back-pressure: rejects with QueueFullError beyond maxQueueDepth', async () => {
    let rejections = 0;
    const b = new DynamicBatcher<string, string>(
      () => new Promise(() => {}), // never resolves; batches pile up
      opts({ windowMs: 5, maxBatchSize: 2, maxConcurrentBatches: 1, maxQueueDepth: 3 }),
    );
    // hooks via a second batcher would complicate; count rejections from promise
    const ps: Array<Promise<unknown>> = [];
    for (let i = 0; i < 6; i++) {
      ps.push(
        b.enqueue(`t${i}`).catch((e) => {
          if (e instanceof QueueFullError) rejections++;
          return null;
        }),
      );
    }
    await vi.advanceTimersByTimeAsync(10);
    // depth 3 allowed to wait (first 2 seal+dispatch immediately at maxBatchSize=2,
    // then in-flight; the rest queue until depth 3) — at least one must be rejected.
    expect(rejections).toBeGreaterThanOrEqual(1);
    expect(b.stats().queueDepth).toBeLessThanOrEqual(3);
  });

  it('propagates a batch failure to every member as BatchError', async () => {
    const b = new DynamicBatcher<string, string>(async () => {
      throw new Error('backend exploded');
    }, opts({ windowMs: 10, maxBatchSize: 10 }));

    // attach handlers immediately so the rejection is always observed
    const e1 = b.enqueue('a').then(
      () => null,
      (e) => e,
    );
    const e2 = b.enqueue('b').then(
      () => null,
      (e) => e,
    );
    await vi.advanceTimersByTimeAsync(11);
    const err1 = (await e1) as BatchError;
    const err2 = (await e2) as BatchError;
    expect(err1).toBeInstanceOf(BatchError);
    expect(err2).toBeInstanceOf(BatchError);
    expect(err1.batchSize).toBe(2);
    expect((err1.causeErr as Error).message).toBe('backend exploded');
  });

  it('rejects the batch if the handler returns a mismatched output count', async () => {
    const b = new DynamicBatcher<string, string>(async () => ['only-one'], opts({ windowMs: 5, maxBatchSize: 10 }));
    const e1 = b.enqueue('a').then(
      () => null,
      (e) => e,
    );
    const e2 = b.enqueue('b').then(
      () => null,
      (e) => e,
    );
    await vi.advanceTimersByTimeAsync(6);
    expect(await e1).toBeInstanceOf(BatchError);
    expect(await e2).toBeInstanceOf(BatchError);
  });

  it('reports queue wait time per item', async () => {
    vi.useRealTimers();
    const b = new DynamicBatcher<string, string>(
      async (inputs) => inputs.map((i) => `out:${i}`),
      opts({ windowMs: 40, maxBatchSize: 10 }),
    );
    const r = await b.enqueue('a');
    expect(r.queueWaitMs).toBeGreaterThanOrEqual(35);
    expect(r.queueWaitMs).toBeLessThan(500);
  });
});
