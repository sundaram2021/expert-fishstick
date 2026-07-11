import { buildBackend } from './server.js';

async function main(): Promise<void> {
  const { app, cfg, log } = await buildBackend();
  await app.listen({ port: cfg.port, host: cfg.host });
  log.info({ port: cfg.port, mode: cfg.mode }, 'model-backend.listening');

  const shutdown = async (signal: string) => {
    log.info({ signal }, 'model-backend.shutting_down');
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
