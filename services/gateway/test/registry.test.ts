import { describe, expect, it } from 'vitest';
import { LatencyTracker, RateWindow, SizeDistribution } from '../src/metrics/registry.js';

describe('LatencyTracker', () => {
  it('computes correct percentiles (nearest-rank)', () => {
    const t = new LatencyTracker(1024);
    for (let i = 1; i <= 100; i++) t.record(i);
    const s = t.snapshot();
    expect(s.count).toBe(100);
    expect(s.p50).toBe(50);
    expect(s.p95).toBe(95);
    expect(s.p99).toBe(99);
    expect(s.max).toBe(100);
    expect(s.avg).toBe(50.5);
  });

  it('survives ring buffer wrap-around', () => {
    const t = new LatencyTracker(10);
    for (let i = 0; i < 1000; i++) t.record(5);
    for (let i = 0; i < 10; i++) t.record(9); // last 10 samples are all 9
    const s = t.snapshot();
    expect(s.p50).toBe(9);
    expect(s.count).toBe(1010);
  });

  it('handles the empty case', () => {
    expect(new LatencyTracker().snapshot().p99).toBeNull();
  });
});

describe('SizeDistribution', () => {
  it('tracks exact batch-size counts and average', () => {
    const d = new SizeDistribution();
    [1, 1, 4, 8, 8, 8].forEach((s) => d.record(s));
    const s = d.snapshot();
    expect(s.batches).toBe(6);
    expect(s.distribution).toEqual({ '1': 2, '4': 1, '8': 3 });
    expect(s.avg_batch_size).toBe(5);
  });
});

describe('RateWindow', () => {
  it('computes trailing-window rates from full seconds', () => {
    const r = new RateWindow();
    const base = 1_700_000_000_000;
    // 10 events/sec for the 10 seconds before "now"
    for (let s = 1; s <= 10; s++) {
      for (let i = 0; i < 10; i++) r.record(1, base - s * 1000);
    }
    expect(r.ratePerSec(10, base)).toBe(10);
    expect(r.ratePerSec(60, base)).toBeCloseTo(100 / 60, 1);
  });
});
