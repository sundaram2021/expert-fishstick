import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { BuiltBackend } from '../src/server.js';
import { buildBackend } from '../src/server.js';

let backend: BuiltBackend;

beforeAll(async () => {
  backend = await buildBackend({
    mode: 'stub',
    logLevel: 'silent',
    stub: { baseMs: 5, perItemMs: 1, jitterStdMs: 0, concurrency: 1 },
  });
});

afterAll(async () => {
  await backend.app.close();
});

describe('model-backend HTTP API', () => {
  it('serves a batch with per-input outputs', async () => {
    const res = await backend.app.inject({
      method: 'POST',
      url: '/infer',
      payload: {
        batch_id: 'b1',
        inputs: [
          { id: 'r1', text: 'I love it' },
          { id: 'r2', text: 'this is broken and terrible' },
        ],
      },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.batch_id).toBe('b1');
    expect(body.batch_size).toBe(2);
    expect(body.inference_ms).toBeGreaterThan(0);
    expect(body.outputs).toHaveLength(2);
    expect(body.outputs[0].id).toBe('r1');
    expect(body.outputs[0].result.label).toBe('positive');
    expect(body.outputs[1].result.label).toBe('negative');
  });

  it('rejects malformed bodies', async () => {
    const res = await backend.app.inject({ method: 'POST', url: '/infer', payload: { inputs: [] } });
    expect(res.statusCode).toBe(400);
  });

  it('fault injection turns the backend unhealthy and recovers', async () => {
    const set = await backend.app.inject({
      method: 'POST',
      url: '/admin/fault',
      payload: { mode: 'error', error_rate: 1, duration_ms: 60_000 },
    });
    expect(set.statusCode).toBe(200);

    const res = await backend.app.inject({
      method: 'POST',
      url: '/infer',
      payload: { inputs: [{ id: 'r1', text: 'hello' }] },
    });
    expect(res.statusCode).toBe(500);
    expect(res.json().error).toBe('inference_failed');

    const clear = await backend.app.inject({ method: 'POST', url: '/admin/fault', payload: { mode: 'none' } });
    expect(clear.statusCode).toBe(200);

    const ok = await backend.app.inject({
      method: 'POST',
      url: '/infer',
      payload: { inputs: [{ id: 'r1', text: 'hello' }] },
    });
    expect(ok.statusCode).toBe(200);
  });

  it('healthz reports model and fault state', async () => {
    const res = await backend.app.inject({ method: 'GET', url: '/healthz' });
    expect(res.statusCode).toBe(200);
    expect(res.json().model).toBe('stub-sentiment-v1');
  });
});
