/**
 * Circuit breaker — hand-rolled state machine (no libraries, per assignment).
 *
 *                 N consecutive failures / latency breaches
 *   ┌────────┐ ─────────────────────────────────────────────► ┌────────┐
 *   │ CLOSED │                                                │  OPEN  │
 *   └────────┘ ◄──────────────┐                               └────────┘
 *        ▲                    │ successesToClose                  │
 *        │                    │ consecutive probe successes       │ cooldown
 *        │               ┌───────────┐                            │ elapsed
 *        └───            │ HALF_OPEN │ ◄──────────────────────────┘
 *   any probe failure ──►│  (probes) │
 *   re-opens (trip++)    └───────────┘
 *
 * Failure definition: a backend call that errors OR completes slower than
 * latencyThresholdMs (a "latency breach"). Both push the same consecutive-
 * failure counter, per the assignment ("fails or exceeds a latency threshold
 * on N consecutive requests").
 *
 * Half-open admission: a configurable *ratio* of requests is admitted as
 * probes (assignment: "allow a small percentage of requests through"),
 * additionally capped by halfOpenMaxProbes concurrent probes so a traffic
 * spike can't stampede a recovering backend.
 *
 * Concurrency-correctness notes (single-threaded event loop, but async
 * interleavings still matter):
 *  - Acquisitions carry the half-open *generation* they were issued in. If
 *    the circuit re-opens and later re-enters half-open, results from stale
 *    probes are ignored — an old probe success can never close the new
 *    circuit.
 *  - Results from calls acquired while CLOSED that land after a trip are
 *    counted in stats but do not mutate the state machine.
 *  - The open -> half_open transition happens lazily on the next state read
 *    after the cooldown elapses (no timers to leak or race).
 */

export type BreakerState = 'closed' | 'open' | 'half_open';

export interface BreakerOptions {
  failureThreshold: number;
  latencyThresholdMs: number;
  cooldownMs: number;
  halfOpenRatio: number;
  halfOpenMaxProbes: number;
  halfOpenSuccessesToClose: number;
}

export interface Acquisition {
  allowed: boolean;
  probe: boolean;
  generation: number;
  state: BreakerState;
  retryAfterMs?: number;
}

export interface BreakerTransition {
  from: BreakerState;
  to: BreakerState;
  reason: string;
  at: number;
}

export class CircuitBreaker {
  private state: BreakerState = 'closed';
  private consecutiveFailures = 0;
  private openedAt = 0;
  private stateSince: number;
  private tripCount = 0;
  private generation = 0;
  private probesInFlight = 0;
  private probeSuccesses = 0;
  private lastTransition: BreakerTransition | null = null;

  private counters = {
    successes: 0,
    failures: 0,
    latencyBreaches: 0,
    rejectedWhileOpen: 0,
    probesAdmitted: 0,
    probeFailures: 0,
    staleResultsIgnored: 0,
  };

  constructor(
    private readonly opts: () => BreakerOptions,
    private readonly hooks: { onTransition?: (t: BreakerTransition) => void } = {},
    private readonly rng: () => number = Math.random,
    private readonly now: () => number = Date.now,
  ) {
    this.stateSince = this.now();
  }

  /** Current state, applying the lazy open -> half_open transition. */
  currentState(): BreakerState {
    this.maybeEnterHalfOpen();
    return this.state;
  }

  retryAfterMs(): number {
    if (this.state !== 'open') return 0;
    return Math.max(0, this.opts().cooldownMs - (this.now() - this.openedAt));
  }

  /**
   * Ask permission to make a backend call. Returns an Acquisition that MUST
   * be passed back to record() exactly once if allowed=true.
   */
  tryAcquire(): Acquisition {
    const o = this.opts();
    this.maybeEnterHalfOpen();

    if (this.state === 'closed') {
      return { allowed: true, probe: false, generation: this.generation, state: 'closed' };
    }

    if (this.state === 'open') {
      this.counters.rejectedWhileOpen++;
      return {
        allowed: false,
        probe: false,
        generation: this.generation,
        state: 'open',
        retryAfterMs: this.retryAfterMs(),
      };
    }

    // half_open: admit a bounded percentage of traffic as probes.
    if (this.probesInFlight < o.halfOpenMaxProbes && this.rng() < o.halfOpenRatio) {
      this.probesInFlight++;
      this.counters.probesAdmitted++;
      return { allowed: true, probe: true, generation: this.generation, state: 'half_open' };
    }
    return {
      allowed: false,
      probe: false,
      generation: this.generation,
      state: 'half_open',
      retryAfterMs: o.cooldownMs,
    };
  }

  /** Report the outcome of a call previously admitted by tryAcquire(). */
  record(acq: Acquisition, outcome: { ok: boolean; latencyMs: number }): void {
    const o = this.opts();
    const breach = outcome.ok && outcome.latencyMs > o.latencyThresholdMs;
    const failure = !outcome.ok || breach;
    if (breach) this.counters.latencyBreaches++;
    if (failure) this.counters.failures++;
    else this.counters.successes++;

    if (acq.probe) {
      // Stale probe from a previous half-open generation, or the circuit
      // already moved on: never let it mutate the current state machine.
      if (acq.generation !== this.generation || this.state !== 'half_open') {
        this.counters.staleResultsIgnored++;
        return;
      }
      this.probesInFlight = Math.max(0, this.probesInFlight - 1);
      if (failure) {
        this.counters.probeFailures++;
        this.trip(breach ? 'probe_latency_breach' : 'probe_failed');
      } else {
        this.probeSuccesses++;
        if (this.probeSuccesses >= o.halfOpenSuccessesToClose) {
          this.transition('closed', 'probes_succeeded');
        }
      }
      return;
    }

    // Non-probe result landing after the state moved on: stats only.
    if (this.state !== 'closed') {
      this.counters.staleResultsIgnored++;
      return;
    }

    if (failure) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= o.failureThreshold) {
        this.trip(breach ? 'latency_threshold_exceeded' : 'consecutive_failures');
      }
    } else {
      this.consecutiveFailures = 0;
    }
  }

  snapshot(): Record<string, unknown> {
    this.maybeEnterHalfOpen();
    const o = this.opts();
    return {
      state: this.state,
      trip_count: this.tripCount,
      consecutive_failures: this.consecutiveFailures,
      time_in_state_ms: this.now() - this.stateSince,
      opened_at: this.openedAt > 0 ? new Date(this.openedAt).toISOString() : null,
      retry_after_ms: this.state === 'open' ? this.retryAfterMs() : null,
      half_open: {
        generation: this.generation,
        probes_in_flight: this.probesInFlight,
        probe_successes_current: this.probeSuccesses,
      },
      counters: { ...this.counters },
      last_transition: this.lastTransition,
      config: {
        failure_threshold: o.failureThreshold,
        latency_threshold_ms: o.latencyThresholdMs,
        cooldown_ms: o.cooldownMs,
        half_open_ratio: o.halfOpenRatio,
        half_open_max_probes: o.halfOpenMaxProbes,
        half_open_successes_to_close: o.halfOpenSuccessesToClose,
      },
    };
  }

  resetCounters(): void {
    this.counters = {
      successes: 0,
      failures: 0,
      latencyBreaches: 0,
      rejectedWhileOpen: 0,
      probesAdmitted: 0,
      probeFailures: 0,
      staleResultsIgnored: 0,
    };
    this.tripCount = 0;
  }

  private maybeEnterHalfOpen(): void {
    if (this.state === 'open' && this.now() - this.openedAt >= this.opts().cooldownMs) {
      this.transition('half_open', 'cooldown_elapsed');
    }
  }

  private trip(reason: string): void {
    this.tripCount++;
    this.transition('open', reason);
  }

  private transition(to: BreakerState, reason: string): void {
    const from = this.state;
    if (from === to) return;
    this.state = to;
    this.stateSince = this.now();
    if (to === 'open') {
      this.openedAt = this.now();
      this.probesInFlight = 0;
      this.probeSuccesses = 0;
    } else if (to === 'half_open') {
      this.generation++;
      this.probesInFlight = 0;
      this.probeSuccesses = 0;
    } else {
      this.consecutiveFailures = 0;
      this.probesInFlight = 0;
      this.probeSuccesses = 0;
    }
    this.lastTransition = { from, to, reason, at: this.now() };
    this.hooks.onTransition?.(this.lastTransition);
  }
}
