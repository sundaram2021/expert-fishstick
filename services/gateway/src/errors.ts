/** Typed errors used to route failures to the right HTTP status + fallback path. */

export class QueueFullError extends Error {
  readonly statusCode = 429;
  constructor(message = 'request queue is full') {
    super(message);
    this.name = 'QueueFullError';
  }
}

export class CircuitOpenError extends Error {
  readonly statusCode = 503;
  constructor(
    public readonly retryAfterMs: number,
    message = 'circuit breaker is open',
  ) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

/** A whole batch failed; every member request receives this. */
export class BatchError extends Error {
  constructor(
    message: string,
    public readonly batchId: string,
    public readonly batchSize: number,
    public readonly causeErr: unknown,
  ) {
    super(message);
    this.name = 'BatchError';
  }
}

export class BackendHttpError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'BackendHttpError';
  }
}
