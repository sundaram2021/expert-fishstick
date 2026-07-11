/**
 * Runtime fault injection for demonstrating (and load-testing) the gateway's
 * circuit breaker. Faults are set through the admin API and expire on their
 * own, so a demo can show trip -> open -> half-open -> recovery end to end.
 */

export type FaultMode = 'none' | 'error' | 'slow';

export interface FaultState {
  mode: FaultMode;
  /** Probability [0..1] that an inference call fails while an 'error' fault is active. */
  errorRate: number;
  /** Extra latency added to every batch while a 'slow' fault is active. */
  extraLatencyMs: number;
  /** Epoch ms when the fault expires. */
  until: number;
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

export class InjectedFaultError extends Error {
  readonly statusCode = 500;
  constructor() {
    super('injected_fault: model backend is failing (fault injection active)');
    this.name = 'InjectedFaultError';
  }
}

export class FaultInjector {
  private state: FaultState = { mode: 'none', errorRate: 1, extraLatencyMs: 0, until: 0 };

  constructor(private now: () => number = Date.now) {}

  set(
    mode: FaultMode,
    opts: { errorRate?: number; extraLatencyMs?: number; durationMs?: number } = {},
  ): FaultState & { active: boolean; remainingMs: number } {
    this.state = {
      mode,
      errorRate: clamp(opts.errorRate ?? 1, 0, 1),
      extraLatencyMs: Math.max(0, opts.extraLatencyMs ?? 0),
      until: mode === 'none' ? 0 : this.now() + (opts.durationMs ?? 30_000),
    };
    return this.current();
  }

  current(): FaultState & { active: boolean; remainingMs: number } {
    const active = this.state.mode !== 'none' && this.now() < this.state.until;
    return { ...this.state, active, remainingMs: active ? this.state.until - this.now() : 0 };
  }

  /** Throws InjectedFaultError if an 'error' fault is active and the dice say so. */
  maybeFail(rng: () => number = Math.random): void {
    const c = this.current();
    if (c.active && c.mode === 'error' && rng() < c.errorRate) {
      throw new InjectedFaultError();
    }
  }

  /** Extra latency to add right now (0 unless a 'slow' fault is active). */
  extraLatencyMsNow(): number {
    const c = this.current();
    return c.active && c.mode === 'slow' ? c.extraLatencyMs : 0;
  }
}
