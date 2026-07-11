/**
 * Threshold calibration: measures real MiniLM cosine similarities for
 * paraphrase pairs (should HIT the cache) and for trap pairs — entity swaps
 * and intent flips (must MISS, or the cache serves wrong answers).
 *
 * The output of this script is what justifies CACHE_SIMILARITY_THRESHOLD.
 * Run: npm run build && npm run calibrate
 */
import { dot } from '../embedding/embedder.js';
import { TransformersEmbedder } from '../embedding/transformersEmbedder.js';

const PARAPHRASES: Array<[string, string]> = [
  ['What is your refund policy for damaged items?', 'How do refunds work if my item arrived damaged?'],
  ['Reset my account password', 'I need to reset the password on my account'],
  ["What's the weather like in Paris today?", 'How is the weather in Paris right now?'],
  ['Cancel my subscription immediately', 'I want to cancel my subscription right away'],
  ['How long does shipping take to Canada?', 'What is the delivery time for orders to Canada?'],
  ['My payment failed but I was still charged', 'I got charged even though the payment failed'],
  ['The app crashes when I open the settings page', 'Opening the settings page makes the app crash'],
  ['Where can I download my invoice?', 'How do I get a copy of my invoice?'],
];

const TRAPS: Array<[string, string, string]> = [
  ["What's the weather like in Paris today?", "What's the weather like in London today?", 'entity swap (city)'],
  ['Cancel my subscription immediately', 'Upgrade my subscription immediately', 'intent flip'],
  ['How do I delete my account?', 'How do I create an account?', 'intent flip'],
  ['How long does shipping take to Canada?', 'How long does shipping take to Japan?', 'entity swap (country)'],
  ['My order #4521 arrived damaged', 'My order #9930 arrived damaged', 'entity swap (order id)'],
  ['Track my order', 'Tell me a joke about cats', 'unrelated'],
];

const embedder = new TransformersEmbedder(
  process.env.EMBED_MODEL_ID ?? 'Xenova/all-MiniLM-L6-v2',
  process.env.TRANSFORMERS_CACHE_DIR,
);
await embedder.init();

async function sim(a: string, b: string): Promise<number> {
  const [va, vb] = await embedder.embed([a, b]);
  return Math.round(dot(va as Float32Array, vb as Float32Array) * 10_000) / 10_000;
}

console.log('\n=== PARAPHRASE PAIRS (cache should HIT) ===');
const paraSims: number[] = [];
for (const [a, b] of PARAPHRASES) {
  const s = await sim(a, b);
  paraSims.push(s);
  console.log(`${s.toFixed(4)}  "${a}"  <->  "${b}"`);
}

console.log('\n=== TRAP PAIRS (cache must MISS) ===');
const trapSims: number[] = [];
for (const [a, b, kind] of TRAPS) {
  const s = await sim(a, b);
  trapSims.push(s);
  console.log(`${s.toFixed(4)}  [${kind}]  "${a}"  <->  "${b}"`);
}

const minPara = Math.min(...paraSims);
const maxTrap = Math.max(...trapSims);
console.log('\n=== SUMMARY ===');
console.log(`paraphrases: min=${minPara.toFixed(4)} avg=${(paraSims.reduce((a, b) => a + b) / paraSims.length).toFixed(4)}`);
console.log(`traps:       max=${maxTrap.toFixed(4)} avg=${(trapSims.reduce((a, b) => a + b) / trapSims.length).toFixed(4)}`);
console.log(
  maxTrap < minPara
    ? `clean margin: any threshold in (${maxTrap.toFixed(3)}, ${minPara.toFixed(3)}) separates all pairs`
    : 'WARNING: trap/paraphrase distributions overlap — threshold must favor precision (sit above the traps)',
);
