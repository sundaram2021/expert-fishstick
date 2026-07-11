import { describe, expect, it } from 'vitest';
import { CircuitBreaker, type BreakerOptions } from '../src/breaker/circuitBreaker.js';

const defaults: BreakerOptions = {
  failureThreshold: 3,
  latencyThresholdMs: 1_000,
  cooldownMs: 5_000,
  halfOpenRatio: 1, // deterministic admission unless a test overrides rng
  halfOpenMaxProbes: 2,
  halfOpenSuccessesToClose: 2,
};

const mk = (over: Partial<BreakerOptions> = {}, rng: () => number = () => 0) => {
  const o = { ...defaults, ...over };
  let t = 1_000_000;
  const transitions: Array<{ from: string; to: string; reason: string }> = [];
  const cb = new CircuitBreaker(
    () => o,
    { onTransition: (tr) => transitions.push({ from: tr.from, to: tr.to, reason: tr.reason }) },
    rng,
    () => t,
  );
  return { cb, transitions, tick: (ms: number) => (t += ms) };
};

const fail = { ok: false, latencyMs: 50 };
const okFast = { ok: true, latencyMs: 50 };
const okSlow = { ok: true, latencyMs: 5_000 }; // latency breach

describe('CircuitBreaker state machine', () => {
  it('stays closed below the failure threshold and resets on success', () => {
    const { cb } = mk({ failureThreshold: 3 });
    for (let i = 0; i < 2; i++) cb.record(cb.tryAcquire(), fail);
    expect(cb.currentState()).toBe('closed');
    cb.record(cb.tryAcquire(), okFast); // success resets the streak
    for (let i = 0; i < 2; i++) cb.record(cb.tryAcquire(), fail);
    expect(cb.currentState()).toBe('closed');
  });

  it('opens after N consecutive failures', () => {
    const { cb, transitions } = mk({ failureThreshold: 3 });
    for (let i = 0; i < 3; i++) cb.record(cb.tryAcquire(), fail);
    expect(cb.currentState()).toBe('open');
    expect(transitions).toEqual([{ from: 'closed', to: 'open', reason: 'consecutive_failures' }]);
    expect((cb.snapshot() as { trip_count: number }).trip_count).toBe(1);
  });

  it('latency breaches count as failures even when the call succeeds', () => {
    const { cb } = mk({ failureThreshold: 2, latencyThresholdMs: 1_000 });
    cb.record(cb.tryAcquire(), okSlow);
    cb.record(cb.tryAcquire(), okSlow);
    expect(cb.currentState()).toBe('open');
    const snap = cb.snapshot() as { counters: { latencyBreaches: number } };
    expect(snap.counters.latencyBreaches).toBe(2);
  });

  it('rejects immediately while open, with a decreasing retry-after', () => {
    const { cb, tick } = mk({ failureThreshold: 1, cooldownMs: 5_000 });
    cb.record(cb.tryAcquire(), fail);
    const a1 = cb.tryAcquire();
    expect(a1.allowed).toBe(false);
    expect(a1.retryAfterMs).toBe(5_000);
    tick(2_000);
    const a2 = cb.tryAcquire();
    expect(a2.allowed).toBe(false);
    expect(a2.retryAfterMs).toBe(3_000);
  });

  it('transitions to half-open after the cooldown and admits a bounded probe', () => {
    const { cb, tick, transitions } = mk({ failureThreshold: 1, cooldownMs: 5_000, halfOpenMaxProbes: 1 });
    cb.record(cb.tryAcquire(), fail);
    tick(5_001);
    const probe = cb.tryAcquire();
    expect(probe.allowed).toBe(true);
    expect(probe.probe).toBe(true);
    // concurrent second request: probe slot taken
    const denied = cb.tryAcquire();
    expect(denied.allowed).toBe(false);
    expect(denied.state).toBe('half_open');
    expect(transitions.map((t) => t.to)).toEqual(['open', 'half_open']);
  });

  it('half-open admission respects the percentage gate', () => {
    let roll = 0.99;
    const { cb, tick } = mk({ failureThreshold: 1, halfOpenRatio: 0.25 }, () => roll);
    cb.record(cb.tryAcquire(), fail);
    tick(5_001);
    expect(cb.tryAcquire().allowed).toBe(false); // 0.99 >= 0.25 → not admitted
    roll = 0.1;
    expect(cb.tryAcquire().allowed).toBe(true); // 0.1 < 0.25 → probe
  });

  it('closes after the configured number of consecutive probe successes', () => {
    const { cb, tick, transitions } = mk({ failureThreshold: 1, halfOpenSuccessesToClose: 2 });
    cb.record(cb.tryAcquire(), fail);
    tick(5_001);
    const p1 = cb.tryAcquire();
    cb.record(p1, okFast);
    expect(cb.currentState()).toBe('half_open'); // 1 of 2
    const p2 = cb.tryAcquire();
    cb.record(p2, okFast);
    expect(cb.currentState()).toBe('closed');
    expect(transitions.map((t) => t.to)).toEqual(['open', 'half_open', 'closed']);
  });

  it('any probe failure re-opens the circuit and counts a new trip', () => {
    const { cb, tick } = mk({ failureThreshold: 1 });
    cb.record(cb.tryAcquire(), fail); // trip 1
    tick(5_001);
    const probe = cb.tryAcquire();
    cb.record(probe, fail); // probe fails → trip 2
    expect(cb.currentState()).toBe('open');
    expect((cb.snapshot() as { trip_count: number }).trip_count).toBe(2);
    // full recovery afterwards
    tick(5_001);
    const p1 = cb.tryAcquire();
    cb.record(p1, okFast);
    const p2 = cb.tryAcquire();
    cb.record(p2, okFast);
    expect(cb.currentState()).toBe('closed');
  });

  it('ignores stale probe results from a previous half-open generation', () => {
    const { cb, tick } = mk({ failureThreshold: 1, halfOpenMaxProbes: 2, halfOpenSuccessesToClose: 1 });
    cb.record(cb.tryAcquire(), fail);
    tick(5_001);
    const slowProbe = cb.tryAcquire(); // gen 1, will come back late
    const fastProbe = cb.tryAcquire(); // gen 1
    cb.record(fastProbe, fail); // re-opens; gen 1 is dead
    expect(cb.currentState()).toBe('open');
    tick(5_001);
    expect(cb.currentState()).toBe('half_open'); // gen 2
    cb.record(slowProbe, okFast); // stale gen-1 success arrives late
    expect(cb.currentState()).toBe('half_open'); // must NOT close from a stale probe
    const snap = cb.snapshot() as { counters: { staleResultsIgnored: number } };
    expect(snap.counters.staleResultsIgnored).toBeGreaterThanOrEqual(1);
  });

  it('ignores non-probe results that land after the circuit tripped', () => {
    const { cb } = mk({ failureThreshold: 1 });
    const inFlight = cb.tryAcquire(); // acquired while closed
    cb.record(cb.tryAcquire(), fail); // another call trips the breaker
    expect(cb.currentState()).toBe('open');
    cb.record(inFlight, okFast); // late success from the closed era
    expect(cb.currentState()).toBe('open'); // state untouched
  });

  it('full lifecycle: closed → open → half_open → closed', () => {
    const { cb, tick, transitions } = mk({ failureThreshold: 2, halfOpenSuccessesToClose: 1 });
    cb.record(cb.tryAcquire(), fail);
    cb.record(cb.tryAcquire(), okSlow); // breach completes the streak
    expect(cb.currentState()).toBe('open');
    tick(5_001);
    const probe = cb.tryAcquire();
    expect(probe.probe).toBe(true);
    cb.record(probe, okFast);
    expect(cb.currentState()).toBe('closed');
    expect(transitions.map((t) => `${t.from}->${t.to}`)).toEqual([
      'closed->open',
      'open->half_open',
      'half_open->closed',
    ]);
  });
});
