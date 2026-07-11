/**
 * Downloads the embedding model into TRANSFORMERS_CACHE_DIR and runs one
 * forward pass. Executed at Docker build time so the model weights are baked
 * into the image — containers start with zero runtime dependency on
 * huggingface.co.
 */
import { TransformersEmbedder } from '../embedding/transformersEmbedder.js';

const modelId = process.env.EMBED_MODEL_ID ?? 'Xenova/all-MiniLM-L6-v2';
const cacheDir = process.env.TRANSFORMERS_CACHE_DIR;

const t0 = Date.now();
const embedder = new TransformersEmbedder(modelId, cacheDir);
await embedder.init();
const [vec] = await embedder.embed(['warmup: the quick brown fox jumps over the lazy dog']);
console.log(
  JSON.stringify({
    model: modelId,
    cache_dir: cacheDir ?? '(default)',
    dims: vec?.length,
    load_ms: Date.now() - t0,
  }),
);
