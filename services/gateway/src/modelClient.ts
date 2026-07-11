/** Thin HTTP client for the model backend, with per-call timeout. */
import { BackendHttpError } from './errors.js';
import type { BackendBatchResponse, InferenceInput } from './types.js';

export class ModelClient {
  constructor(private readonly opts: () => { baseUrl: string; timeoutMs: number }) {}

  async inferBatch(inputs: InferenceInput[], batchId: string): Promise<BackendBatchResponse> {
    const o = this.opts();
    let res: Response;
    try {
      res = await fetch(`${o.baseUrl}/infer`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ batch_id: batchId, inputs }),
        signal: AbortSignal.timeout(o.timeoutMs),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new BackendHttpError(0, `model backend unreachable: ${msg}`);
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new BackendHttpError(res.status, `model backend returned ${res.status}: ${text.slice(0, 300)}`);
    }
    return (await res.json()) as BackendBatchResponse;
  }

  async health(): Promise<boolean> {
    try {
      const res = await fetch(`${this.opts().baseUrl}/healthz`, {
        signal: AbortSignal.timeout(2_000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }
}
