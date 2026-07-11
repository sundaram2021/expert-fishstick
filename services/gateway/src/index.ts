import { buildGateway } from './server.js';

async function main(): Promise<void> {
  const { app, cfg, log } = await buildGateway();
  await app.listen({ port: cfg.port, host: cfg.host });
  log.info(
    {
      port: cfg.port,
      model_backend: cfg.modelBackendUrl,
      batch: cfg.batch,
      cache: {
        enabled: cfg.cache.enabled,
        threshold: cfg.cache.similarityThreshold,
        ttl_ms: cfg.cache.ttlMs,
        max_size: cfg.cache.maxSize,
      },
      breaker: cfg.breaker,
    },
    'gateway.listening',
  );

  const shutdown = async (signal: string) => {
    log.info({ signal }, 'gateway.shutting_down');
    try {
      await app.close();
      process.exit(0);
    } catch {
      process.exit(1);
    }
  };
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
  process.on('SIGINT', () => void shutdown('SIGINT'));
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
