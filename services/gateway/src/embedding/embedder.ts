/** Embedding provider interface + a deterministic lexical fallback. */

export interface Embedder {
  readonly name: string;
  readonly dims: number;
  init(): Promise<void>;
  /** Embed a batch of texts into L2-normalized vectors (cosine == dot). */
  embed(texts: string[]): Promise<Float32Array[]>;
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

export function l2normalize(v: Float32Array): Float32Array {
  let ss = 0;
  for (let i = 0; i < v.length; i++) ss += (v[i] as number) * (v[i] as number);
  const n = Math.sqrt(ss) || 1;
  for (let i = 0; i < v.length; i++) v[i] = (v[i] as number) / n;
  return v;
}

export function fnv1a(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Deterministic char-trigram hashing embedder.
 *
 * NOT semantic — cosine similarity of these vectors measures lexical overlap
 * only. It exists for two purposes:
 *  1. Fast, dependency-free unit/integration tests.
 *  2. An optional degraded fallback (EMBED_ALLOW_FALLBACK=true) so the
 *     gateway can still de-duplicate near-identical requests if the real
 *     embedding model cannot be loaded.
 * Production semantic caching should always run the transformers provider.
 */
export class HashEmbedder implements Embedder {
  readonly name = 'hash-trigram-384';
  readonly dims = 384;

  async init(): Promise<void> {
    /* nothing to load */
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    return texts.map((t) => this.one(t));
  }

  private one(text: string): Float32Array {
    const v = new Float32Array(this.dims);
    const s = `  ${text.toLowerCase().trim().replace(/\s+/g, ' ')}  `;
    for (let i = 0; i < s.length - 2; i++) {
      const h = fnv1a(s.slice(i, i + 3));
      const idx = h % this.dims;
      const sign = (h >>> 24) & 1 ? 1 : -1;
      v[idx] = (v[idx] as number) + sign;
    }
    return l2normalize(v);
  }
}
