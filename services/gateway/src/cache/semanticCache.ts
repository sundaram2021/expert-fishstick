/**
 * Semantic cache.
 *
 * Entries are keyed by normalized text and carry an L2-normalized embedding.
 * Lookups do a linear cosine scan (dot product of normalized vectors) over
 * the cache and return the best entry above the similarity threshold. An
 * exact-key fast path skips the scan for identical requests.
 *
 * Eviction & freshness:
 *  - LRU: the backing Map keeps insertion order; hits re-insert the entry,
 *    so the first key is always least-recently-used. Inserts beyond maxSize
 *    evict from the front. O(1).
 *  - TTL: entries older than ttlMs stop being served in normal mode. They are
 *    retained (up to ttl * staleGraceFactor) so that *degraded* mode — while
 *    the circuit breaker is open — can still serve a stale-but-relevant
 *    answer instead of an error. Grace-expired entries are removed lazily on
 *    scan and by a periodic sweep.
 *
 * Degraded mode also uses a lower similarity threshold: when the alternative
 * is an error, a "pretty close" cached answer is better than nothing. Both
 * thresholds are configurable; the README discusses the staleness/cost
 * tradeoff.
 *
 * Sizing: a linear scan over ≤ maxSize (default 1000) 384-dim float32 vectors
 * is ~0.4M multiply-adds — microseconds on any modern CPU, and orders of
 * magnitude cheaper than the inference it saves. Past ~50k entries you would
 * swap the scan for an ANN index (HNSW) or an external vector store; the
 * interface would not change.
 */
import { dot } from '../embedding/embedder.js';

export interface CacheOptions {
  enabled: boolean;
  similarityThreshold: number;
  ttlMs: number;
  maxSize: number;
  degradedThreshold: number;
  serveStaleWhenOpen: boolean;
  staleGraceFactor: number;
}

export interface CacheHit {
  response: unknown;
  similarity: number;
  matchedKey: string;
  ageMs: number;
  stale: boolean;
  entryHits: number;
}

export interface LookupOutcome {
  hit: CacheHit | null;
  /** Best similarity seen during the scan, hit or not — used for threshold tuning. */
  bestSimilarity: number | null;
}

interface Entry {
  key: string;
  vector: Float32Array;
  response: unknown;
  createdAt: number;
  lastAccessAt: number;
  hits: number;
}

const round4 = (v: number): number => Math.round(v * 10_000) / 10_000;

export class SemanticCache {
  private entries = new Map<string, Entry>();
  private stats = {
    hits: 0,
    misses: 0,
    staleServed: 0,
    evictionsLru: 0,
    expiredRemoved: 0,
    insertions: 0,
    hitSimilaritySum: 0,
    missBestSimilaritySum: 0,
    missesWithCandidates: 0,
  };

  constructor(
    private readonly opts: () => CacheOptions,
    private readonly now: () => number = Date.now,
  ) {}

  static normalizeKey(text: string): string {
    return text.trim().toLowerCase().replace(/\s+/g, ' ');
  }

  get size(): number {
    return this.entries.size;
  }

  lookup(text: string, vector: Float32Array, mode: 'normal' | 'degraded' = 'normal'): LookupOutcome {
    const o = this.opts();
    if (!o.enabled) return { hit: null, bestSimilarity: null };

    const threshold = mode === 'degraded' ? o.degradedThreshold : o.similarityThreshold;
    const allowStale = mode === 'degraded' && o.serveStaleWhenOpen;
    const t = this.now();
    const key = SemanticCache.normalizeKey(text);

    let best: Entry | null = null;
    let bestSim = -1;

    const exact = this.entries.get(key);
    if (exact && !this.pastGrace(exact, t) && (allowStale || !this.isExpired(exact, t))) {
      best = exact;
      bestSim = 1;
    } else {
      for (const e of [...this.entries.values()]) {
        const expired = this.isExpired(e, t);
        if (expired && this.pastGrace(e, t)) {
          this.entries.delete(e.key);
          this.stats.expiredRemoved++;
          continue;
        }
        if (expired && !allowStale) continue;
        const sim = dot(vector, e.vector);
        if (sim > bestSim) {
          bestSim = sim;
          best = e;
        }
      }
    }

    if (best && bestSim >= threshold) {
      const stale = this.isExpired(best, t);
      if (stale) this.stats.staleServed++;
      best.hits++;
      best.lastAccessAt = t;
      // LRU touch: move to the back of the Map's insertion order.
      this.entries.delete(best.key);
      this.entries.set(best.key, best);
      this.stats.hits++;
      this.stats.hitSimilaritySum += bestSim;
      return {
        hit: {
          response: best.response,
          similarity: round4(bestSim),
          matchedKey: best.key,
          ageMs: t - best.createdAt,
          stale,
          entryHits: best.hits,
        },
        bestSimilarity: round4(bestSim),
      };
    }

    this.stats.misses++;
    if (best) {
      this.stats.missBestSimilaritySum += Math.max(0, bestSim);
      this.stats.missesWithCandidates++;
    }
    return { hit: null, bestSimilarity: best ? round4(bestSim) : null };
  }

  insert(text: string, vector: Float32Array, response: unknown): void {
    const o = this.opts();
    if (!o.enabled) return;
    const key = SemanticCache.normalizeKey(text);
    const t = this.now();
    const existing = this.entries.get(key);
    if (existing) {
      existing.response = response;
      existing.vector = vector;
      existing.createdAt = t;
      existing.lastAccessAt = t;
      this.entries.delete(key);
      this.entries.set(key, existing);
      return;
    }
    while (this.entries.size >= o.maxSize) {
      const lruKey = this.entries.keys().next().value as string;
      this.entries.delete(lruKey);
      this.stats.evictionsLru++;
    }
    this.entries.set(key, {
      key,
      vector,
      response,
      createdAt: t,
      lastAccessAt: t,
      hits: 0,
    });
    this.stats.insertions++;
  }

  /** Periodic cleanup of grace-expired entries. Returns number removed. */
  sweep(): number {
    const t = this.now();
    let removed = 0;
    for (const e of [...this.entries.values()]) {
      if (this.pastGrace(e, t)) {
        this.entries.delete(e.key);
        this.stats.expiredRemoved++;
        removed++;
      }
    }
    return removed;
  }

  clear(): void {
    this.entries.clear();
  }

  resetStats(): void {
    this.stats = {
      hits: 0,
      misses: 0,
      staleServed: 0,
      evictionsLru: 0,
      expiredRemoved: 0,
      insertions: 0,
      hitSimilaritySum: 0,
      missBestSimilaritySum: 0,
      missesWithCandidates: 0,
    };
  }

  snapshot(): Record<string, unknown> {
    const s = this.stats;
    const o = this.opts();
    const lookups = s.hits + s.misses;
    return {
      enabled: o.enabled,
      size: this.entries.size,
      max_size: o.maxSize,
      ttl_ms: o.ttlMs,
      similarity_threshold: o.similarityThreshold,
      degraded_threshold: o.degradedThreshold,
      lookups,
      hits: s.hits,
      misses: s.misses,
      hit_rate: lookups > 0 ? round4(s.hits / lookups) : null,
      avg_hit_similarity: s.hits > 0 ? round4(s.hitSimilaritySum / s.hits) : null,
      avg_best_similarity_on_miss:
        s.missesWithCandidates > 0 ? round4(s.missBestSimilaritySum / s.missesWithCandidates) : null,
      estimated_model_calls_saved: s.hits,
      stale_served_degraded: s.staleServed,
      insertions: s.insertions,
      evictions_lru: s.evictionsLru,
      expired_removed: s.expiredRemoved,
    };
  }

  private isExpired(e: Entry, t: number): boolean {
    return t - e.createdAt > this.opts().ttlMs;
  }

  private pastGrace(e: Entry, t: number): boolean {
    const o = this.opts();
    return t - e.createdAt > o.ttlMs * Math.max(1, o.staleGraceFactor);
  }
}
