/**
 * Gnarl agent memory for the Vercel AI SDK.
 *
 * - `gnarlMemoryTools` — `rememberFact` / `recallMemories` tools for a model to call.
 * - `gnarlMemoryMiddleware` — recalls memories for the latest user message and
 *   adds them to the system prompt, via `wrapLanguageModel`.
 *
 * Thin wrappers over the published `gnarl-client`.
 */
export {
  formatMemories,
  type GnarlMemoryMiddlewareOptions,
  gnarlMemoryMiddleware,
} from "./middleware.js";
export { type GnarlMemoryScope, PARTIAL_NOTE, type Recalled } from "./recall.js";
export {
  type GnarlMemoryTools,
  type GnarlMemoryToolsOptions,
  gnarlMemoryTools,
  type RecalledMemory,
  type RecallMemoriesOutput,
  type RememberFactOutput,
} from "./tools.js";
