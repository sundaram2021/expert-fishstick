/** Shared request/response shapes between gateway and model backend. */

export interface InferenceInput {
  id: string;
  text: string;
}

export interface ModelResult {
  label: string;
  score: number;
  model: string;
  tokens: number;
}

export interface BackendBatchResponse {
  batch_id: string | null;
  model: string;
  batch_size: number;
  /** Pure inference time reported by the backend (excludes its own queueing). */
  inference_ms: number;
  /** Time the batch waited for the (simulated) device inside the backend. */
  queue_wait_ms: number;
  outputs: Array<{ id: string; result: ModelResult }>;
}
