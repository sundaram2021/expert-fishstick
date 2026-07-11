/**
 * Optional real-model mode: a genuine DistilBERT sentiment classifier running
 * on CPU via ONNX (transformers.js). Enable with MODEL_MODE=real.
 *
 * It honors the exact same serving contract as the stub — same request/response
 * shape, same serialized-device semantics, same fault injection — so the
 * gateway cannot tell the difference. That symmetry is the point: the serving
 * layer is model-agnostic.
 */
import { Semaphore } from './semaphore.js';
import type { FaultInjector } from './faults.js';
import type { BatchOutput, InferenceInput, ModelResult } from './stubModel.js';

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export class RealModel {
  readonly name: string;
  private device = new Semaphore(1);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private pipe: any = null;

  constructor(
    private modelId: string,
    private faults: FaultInjector,
  ) {
    this.name = modelId;
  }

  async init(): Promise<void> {
    const { pipeline, env } = await import('@huggingface/transformers');
    if (process.env.TRANSFORMERS_CACHE_DIR) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (env as any).cacheDir = process.env.TRANSFORMERS_CACHE_DIR;
    }
    this.pipe = await pipeline('text-classification', this.modelId, { dtype: 'q8' });
  }

  async inferBatch(inputs: InferenceInput[]): Promise<BatchOutput> {
    if (!this.pipe) throw new Error('real model not initialized');
    const queuedAt = performance.now();
    await this.device.acquire();
    const queueWaitMs = Math.round(performance.now() - queuedAt);
    try {
      this.faults.maybeFail();
      const extra = this.faults.extraLatencyMsNow();
      if (extra > 0) await sleep(extra);
      const t0 = performance.now();
      const raw = await this.pipe(
        inputs.map((i) => i.text),
        { top_k: 1 },
      );
      const inferenceMs = Math.round(performance.now() - t0) + extra;
      const outputs = inputs.map((inp, idx) => {
        const row = raw[idx];
        const first = Array.isArray(row) ? row[0] : row;
        const label: ModelResult['label'] =
          String(first.label).toLowerCase() === 'positive' ? 'positive' : 'negative';
        const result: ModelResult = {
          label,
          score: Math.round(Number(first.score) * 1000) / 1000,
          model: this.name,
          tokens: inp.text.split(/\s+/).filter(Boolean).length,
        };
        return { id: inp.id, result };
      });
      return { outputs, inferenceMs, queueWaitMs };
    } finally {
      this.device.release();
    }
  }
}
