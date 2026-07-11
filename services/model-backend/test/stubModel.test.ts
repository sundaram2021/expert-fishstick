import { describe, expect, it } from 'vitest';
import { FaultInjector, InjectedFaultError } from '../src/faults.js';
import { StubModel, gaussian } from '../src/stubModel.js';

const mkModel = (over: Partial<ConstructorParameters<typeof StubModel>[0]> = {}, rng?: () => number) =>
  new StubModel(
    { baseMs: 20, perItemMs: 2, jitterStdMs: 0, concurrency: 1, ...over },
    new FaultInjector(),
    rng,
  );

describe('StubModel', () => {
  it('classifies deterministically from the lexicon', () => {
    const m = mkModel();
    expect(m.classify('I love this, it is great').label).toBe('positive');
    expect(m.classify('this is terrible and broken').label).toBe('negative');
    expect(m.classify('the sky is above the ground').label).toBe('neutral');
    // determinism: same text, same score
    expect(m.classify('hello world')).toEqual(m.classify('hello world'));
  });

  it('latency grows with batch size (base + perItem * n)', () => {
    const m = mkModel({ baseMs: 100, perItemMs: 10, jitterStdMs: 0 });
    expect(m.computeMs(1)).toBe(110);
    expect(m.computeMs(16)).toBe(260);
    // 16 items in one batch cost far less than 16 single batches
    expect(m.computeMs(16)).toBeLessThan(16 * m.computeMs(1));
  });

  it('serializes concurrent batches like a single device', async () => {
    const m = mkModel({ baseMs: 60, perItemMs: 0 });
    const [a, b] = await Promise.all([
      m.inferBatch([{ id: '1', text: 'x' }]),
      m.inferBatch([{ id: '2', text: 'y' }]),
    ]);
    // one of the two must have queued behind the other
    const waits = [a.queueWaitMs, b.queueWaitMs].sort((x, y) => x - y);
    expect(waits[0]).toBeLessThan(15);
    expect(waits[1]).toBeGreaterThanOrEqual(40);
  });

  it('responds per-input with matching ids', async () => {
    const m = mkModel({ baseMs: 1, perItemMs: 0 });
    const out = await m.inferBatch([
      { id: 'a', text: 'great stuff' },
      { id: 'b', text: 'awful stuff' },
    ]);
    expect(out.outputs.map((o) => o.id)).toEqual(['a', 'b']);
    expect(out.outputs[0]?.result.label).toBe('positive');
    expect(out.outputs[1]?.result.label).toBe('negative');
  });
});

describe('FaultInjector', () => {
  it('error fault fails calls, then expires on its own', async () => {
    let t = 1_000_000;
    const faults = new FaultInjector(() => t);
    const m = new StubModel({ baseMs: 1, perItemMs: 0, jitterStdMs: 0, concurrency: 1 }, faults);
    faults.set('error', { errorRate: 1, durationMs: 500 });
    await expect(m.inferBatch([{ id: '1', text: 'x' }])).rejects.toBeInstanceOf(InjectedFaultError);
    t += 501; // fault expired
    await expect(m.inferBatch([{ id: '1', text: 'x' }])).resolves.toBeTruthy();
  });

  it('slow fault adds latency', async () => {
    let t = 0;
    const faults = new FaultInjector(() => t);
    faults.set('slow', { extraLatencyMs: 123, durationMs: 1000 });
    expect(faults.extraLatencyMsNow()).toBe(123);
    t = 1001;
    expect(faults.extraLatencyMsNow()).toBe(0);
  });
});

describe('gaussian', () => {
  it('produces roughly zero-mean samples', () => {
    let seed = 42;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2 ** 31;
      return seed / 2 ** 31;
    };
    const n = 5000;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += gaussian(rng);
    expect(Math.abs(sum / n)).toBeLessThan(0.1);
  });
});
