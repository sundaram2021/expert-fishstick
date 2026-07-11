import { describe, expect, it } from 'vitest';
import { SemanticCache, type CacheOptions } from '../src/cache/semanticCache.js';
import { l2normalize } from '../src/embedding/embedder.js';

const defaults: CacheOptions = {
  enabled: true,
  similarityThreshold: 0.9,
  ttlMs: 1_000,
  maxSize: 3,
  degradedThreshold: 0.8,
  serveStaleWhenOpen: true,
  staleGraceFactor: 3,
};

/** Build a unit vector at a chosen cosine similarity to the base e1 vector. */
const vecAt = (cos: number): Float32Array => {
  const v = new Float32Array(4);
  v[0] = cos;
  v[1] = Math.sqrt(Math.max(0, 1 - cos * cos));
  return l2normalize(v);
};
const e1 = vecAt(1);

const mk = (over: Partial<CacheOptions> = {}) => {
  const o = { ...defaults, ...over };
  let t = 1_000_000;
  const cache = new SemanticCache(
    () => o,
    () => t,
  );
  return { cache, o, tick: (ms: number) => (t += ms) };
};

describe('SemanticCache', () => {
  it('returns semantically similar entries above the threshold, not just exact matches', () => {
    const { cache } = mk({ similarityThreshold: 0.9 });
    cache.insert('what is the refund policy', e1, { answer: 42 });

    const near = cache.lookup('refund policy question, reworded', vecAt(0.95));
    expect(near.hit).not.toBeNull();
    expect(near.hit?.similarity).toBeCloseTo(0.95, 2);
    expect(near.hit?.response).toEqual({ answer: 42 });

    const far = cache.lookup('completely unrelated', vecAt(0.5));
    expect(far.hit).toBeNull();
    expect(far.bestSimilarity).toBeCloseTo(0.5, 2);
  });

  it('misses just below the threshold and hits just above it', () => {
    const { cache } = mk({ similarityThreshold: 0.9 });
    cache.insert('seed', e1, 'resp');
    // margins sized above float32 rounding error (~1e-7 at this scale)
    expect(cache.lookup('q', vecAt(0.8995)).hit).toBeNull();
    expect(cache.lookup('q', vecAt(0.9005)).hit).not.toBeNull();
  });

  it('exact-text fast path hits with similarity 1.0 regardless of scan', () => {
    const { cache } = mk();
    cache.insert('Hello   World', e1, 'resp');
    const out = cache.lookup('  hello world ', e1); // normalization: trim/case/whitespace
    expect(out.hit?.similarity).toBe(1);
  });

  it('expires entries after TTL in normal mode', () => {
    const { cache, tick } = mk({ ttlMs: 1_000 });
    cache.insert('seed', e1, 'resp');
    tick(1_001);
    expect(cache.lookup('seed', e1, 'normal').hit).toBeNull();
  });

  it('degraded mode serves stale entries within the grace window and flags them', () => {
    const { cache, tick } = mk({ ttlMs: 1_000, staleGraceFactor: 3, degradedThreshold: 0.8 });
    cache.insert('seed', e1, 'resp');
    tick(1_500); // expired, within grace (3s)
    const out = cache.lookup('seed', e1, 'degraded');
    expect(out.hit).not.toBeNull();
    expect(out.hit?.stale).toBe(true);

    tick(2_000); // now past grace (3.5s > 3s) — gone even for degraded
    expect(cache.lookup('seed', e1, 'degraded').hit).toBeNull();
  });

  it('degraded mode uses the relaxed threshold', () => {
    const { cache } = mk({ similarityThreshold: 0.9, degradedThreshold: 0.8 });
    cache.insert('seed', e1, 'resp');
    const v = vecAt(0.85);
    expect(cache.lookup('q', v, 'normal').hit).toBeNull();
    expect(cache.lookup('q', v, 'degraded').hit).not.toBeNull();
  });

  it('evicts least-recently-used entries at maxSize', () => {
    const { cache } = mk({ maxSize: 3, similarityThreshold: 0.99 });
    const va = l2normalize(new Float32Array([1, 0, 0, 0]));
    const vb = l2normalize(new Float32Array([0, 1, 0, 0]));
    const vc = l2normalize(new Float32Array([0, 0, 1, 0]));
    const vd = l2normalize(new Float32Array([0, 0, 0, 1]));
    cache.insert('a', va, 'A');
    cache.insert('b', vb, 'B');
    cache.insert('c', vc, 'C');
    // touch 'a' so 'b' becomes LRU
    expect(cache.lookup('a', va).hit).not.toBeNull();
    cache.insert('d', vd, 'D'); // evicts 'b'
    expect(cache.size).toBe(3);
    expect(cache.lookup('b', vb).hit).toBeNull();
    expect(cache.lookup('a', va).hit).not.toBeNull();
    expect(cache.lookup('d', vd).hit).not.toBeNull();
    expect((cache.snapshot() as { evictions_lru: number }).evictions_lru).toBe(1);
  });

  it('re-inserting the same key refreshes it instead of duplicating', () => {
    const { cache, tick } = mk({ ttlMs: 1_000 });
    cache.insert('seed', e1, 'old');
    tick(900);
    cache.insert('seed', e1, 'new');
    tick(900); // 1800 since first insert, but only 900 since refresh
    const out = cache.lookup('seed', e1);
    expect(out.hit?.response).toBe('new');
    expect(cache.size).toBe(1);
  });

  it('tracks hit-rate, similarity averages and calls saved', () => {
    const { cache } = mk({ similarityThreshold: 0.9 });
    cache.insert('seed', e1, 'resp');
    cache.lookup('q1', vecAt(0.95)); // hit
    cache.lookup('q2', vecAt(0.92)); // hit
    cache.lookup('q3', vecAt(0.5)); // miss
    const s = cache.snapshot() as Record<string, number>;
    expect(s.hits).toBe(2);
    expect(s.misses).toBe(1);
    expect(s.hit_rate).toBeCloseTo(2 / 3, 3);
    expect(s.avg_hit_similarity).toBeGreaterThan(0.9);
    expect(s.estimated_model_calls_saved).toBe(2);
    expect(s.avg_best_similarity_on_miss).toBeCloseTo(0.5, 2);
  });

  it('sweep removes only grace-expired entries', () => {
    const { cache, tick } = mk({ ttlMs: 1_000, staleGraceFactor: 2 });
    cache.insert('old', e1, 'resp');
    tick(1_500);
    cache.insert('fresh', vecAt(0), 'resp2');
    expect(cache.sweep()).toBe(0); // 'old' expired but within grace (2s)
    tick(600); // 'old' now 2.1s old — past grace
    expect(cache.sweep()).toBe(1);
    expect(cache.size).toBe(1);
  });

  it('disabled cache never hits or stores', () => {
    const { cache } = mk({ enabled: false });
    cache.insert('seed', e1, 'resp');
    expect(cache.size).toBe(0);
    expect(cache.lookup('seed', e1).hit).toBeNull();
  });
});
