/** Workload data for the load test phases. */

const TEMPLATES = [
  (i) => `Support ticket ${i}: my package arrived two days late and the box was damaged.`,
  (i) => `Order ${i}: the checkout page crashes when I apply a discount code.`,
  (i) => `Ticket ${i}: I love the new dashboard but the export button is broken.`,
  (i) => `Case ${i}: requesting a refund for the duplicate charge on invoice ${i * 7}.`,
  (i) => `Feedback ${i}: the mobile app is fast and easy to use, great work team.`,
  (i) => `Issue ${i}: the password reset email never arrives for my account.`,
  (i) => `Review ${i}: shipping was fast but the packaging felt cheap and flimsy.`,
  (i) => `Report ${i}: search results are wrong when I filter by date range.`,
  (i) => `Note ${i}: billing portal shows an error after updating my payment card.`,
  (i) => `Message ${i}: the onboarding flow is excellent, signup took two minutes.`,
];

/** Unique text per index — guarantees zero accidental cache hits. */
export const uniqueText = (i) => TEMPLATES[i % TEMPLATES.length](i);

/** Paraphrase pairs: seed is cached first, then the query SHOULD hit. */
export const PARAPHRASE_PAIRS = [
  ['What is your refund policy for damaged items?', 'How do refunds work if my item arrived damaged?'],
  ['Reset my account password', 'I need to reset the password on my account'],
  ["What's the weather like in Paris today?", 'How is the weather in Paris right now?'],
  ['Cancel my subscription immediately', 'I want to cancel my subscription right away'],
  ['How long does shipping take to Canada?', 'What is the delivery time for orders to Canada?'],
  ['My payment failed but I was still charged', 'I got charged even though the payment failed'],
  ['The app crashes when I open the settings page', 'Opening the settings page makes the app crash'],
  ['Where can I download my invoice?', 'How do I get a copy of my invoice?'],
];

/** Trap pairs: seed is cached first, then the query MUST miss (wrong answer otherwise). */
export const TRAP_PAIRS = [
  ["What's the weather like in Paris today?", "What's the weather like in London today?", 'entity swap (city)'],
  ['Cancel my subscription immediately', 'Upgrade my subscription immediately', 'intent flip'],
  ['How do I delete my account?', 'How do I create an account?', 'intent flip'],
  ['How long does shipping take to Canada?', 'How long does shipping take to Japan?', 'entity swap (country)'],
  ['Track my order', 'Tell me a joke about cats', 'unrelated'],
];

/** Texts seeded before the breaker demo so degraded mode has something to serve. */
export const BREAKER_SEEDS = [
  'What is the status of the payments service?',
  'Summarize the latest deployment incident',
  'Is the checkout API healthy right now?',
];

/** Queries sent while the circuit is open — paraphrases of the seeds. */
export const BREAKER_DEGRADED_QUERIES = [
  'What is the current status of the payments service?',
  'Give me a summary of the latest deployment incident',
  'Is the checkout API currently healthy?',
];
