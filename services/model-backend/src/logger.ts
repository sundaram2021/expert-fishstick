import { pino } from 'pino';

export function createLogger(level: string, service: string) {
  return pino({
    level,
    base: { service },
    timestamp: pino.stdTimeFunctions.isoTime,
  });
}

export type Logger = ReturnType<typeof createLogger>;
