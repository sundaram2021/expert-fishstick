/**
 * Real semantic embedder: sentence-transformers/all-MiniLM-L6-v2 running as
 * quantized ONNX via transformers.js — in-process, CPU, no Python, no
 * external service. 384-dim vectors, mean pooling, L2-normalized.
 *
 * This is what makes the cache *semantic*: paraphrases land near each other
 * in embedding space even with zero token overlap. The Docker build bakes the
 * model weights into the image (see scripts/warmup-model.ts) so the container
 * has no runtime dependency on huggingface.co.
 */
import { type Embedder, l2normalize } from './embedder.js';

export class TransformersEmbedder implements Embedder {
  readonly name: string;
  readonly dims = 384;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private extractor: any = null;

  constructor(
    private readonly modelId = 'Xenova/all-MiniLM-L6-v2',
    private readonly cacheDir?: string,
  ) {
    this.name = modelId;
  }

  async init(): Promise<void> {
    const { pipeline, env } = await import('@huggingface/transformers');
    if (this.cacheDir) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (env as any).cacheDir = this.cacheDir;
    }
    this.extractor = await pipeline('feature-extraction', this.modelId, { dtype: 'q8' });
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (!this.extractor) throw new Error('transformers embedder not initialized');
    const out = await this.extractor(texts, { pooling: 'mean', normalize: true });
    const data = out.data as Float32Array;
    const dims = out.dims as number[];
    const d = dims[dims.length - 1] as number;
    const n = texts.length;
    const vectors: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
      // Copy out of the shared tensor buffer; re-normalize defensively.
      vectors.push(l2normalize(new Float32Array(data.subarray(i * d, (i + 1) * d))));
    }
    return vectors;
  }
}
