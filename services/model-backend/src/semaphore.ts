/**
 * Counting semaphore used to reproduce the execution model of a real
 * accelerator: a GPU executes one kernel (one batch) at a time. Work that
 * arrives while the device is busy queues up behind it.
 *
 * This is the detail that makes the batching benchmark honest — without it, a
 * naive stub would happily run 64 "inferences" in parallel and batching would
 * show no benefit, which is not how GPUs behave.
 */
export class Semaphore {
  private free: number;
  private waiters: Array<() => void> = [];

  constructor(permits: number) {
    this.free = permits;
  }

  /** Number of acquirers currently waiting for a permit. */
  get queueDepth(): number {
    return this.waiters.length;
  }

  async acquire(): Promise<void> {
    if (this.free > 0) {
      this.free--;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
  }

  release(): void {
    const next = this.waiters.shift();
    if (next) next();
    else this.free++;
  }
}
