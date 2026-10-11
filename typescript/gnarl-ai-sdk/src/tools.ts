import { type Tool, tool } from "ai";
import { z } from "zod";
import { clientOf, type GnarlMemoryScope, recall, scopeFields } from "./recall.js";

export interface GnarlMemoryToolsOptions extends GnarlMemoryScope {
  /** Most memories `recallMemories` may return per call. Default 10; the node caps at 100. */
  maxK?: number;
  /** `agent` recorded on stored memories. Default `ai-sdk`. */
  agent?: string;
}

const rememberInput = z.object({
  fact: z.string().min(1).describe("One self-contained fact worth remembering, in plain language."),
  factType: z.string().optional().describe("Optional kind of fact, e.g. 'preference', 'profile', 'event'."),
  pinned: z.boolean().optional().describe("Pin facts that should always be kept."),
});

export interface RememberFactOutput {
  id: string;
  stored: true;
}

export interface RecalledMemory {
  id: string;
  content: string;
  score: number;
  factType: string | null;
  created: string | null;
  session: string | null;
}

export interface RecallMemoriesOutput {
  memories: RecalledMemory[];
  /** True when part of the memory space did not answer; a missing memory may still exist. */
  partial: boolean;
}

// A type alias, not an interface: only an alias is assignable to the AI SDK's
// index-signature `ToolSet`, so `tools: gnarlMemoryTools()` typechecks.
export type GnarlMemoryTools = {
  rememberFact: Tool<z.infer<typeof rememberInput>, RememberFactOutput>;
  recallMemories: Tool<{ query: string; k?: number | undefined }, RecallMemoriesOutput>;
};

/**
 * Tools that let a model save and look up long-term memories in Gnarl.
 *
 * The scope (namespace, space, user, session) is fixed here by the developer;
 * the model chooses only what to remember and what to look for, so it cannot
 * read or write another user's memories.
 *
 * ```ts
 * const result = await generateText({ model, tools: gnarlMemoryTools({ user: "alice" }), prompt });
 * ```
 */
export function gnarlMemoryTools(options: GnarlMemoryToolsOptions = {}): GnarlMemoryTools {
  const client = clientOf(options);
  const maxK = Math.min(Math.max(options.maxK ?? 10, 1), 100);
  const agent = options.agent ?? "ai-sdk";

  const recallInput = z.object({
    query: z.string().min(1).describe("What to look for, as a natural-language question or phrase."),
    k: z.number().int().min(1).max(maxK).optional().describe(`How many memories to return (1-${maxK}, default 5).`),
  });

  return {
    rememberFact: tool({
      description: "Save a durable fact about the user or the task to long-term memory, so it can be recalled in later conversations.",
      inputSchema: rememberInput,
      execute: async ({ fact, factType, pinned }, { abortSignal }) => {
        const res = await client.memory.remember(
          {
            content: fact,
            agent,
            ...scopeFields(options),
            ...(options.session !== undefined ? { session: options.session } : {}),
            ...(factType !== undefined ? { fact_type: factType } : {}),
            ...(pinned !== undefined ? { pinned } : {}),
          },
          abortSignal ? { signal: abortSignal } : undefined,
        );
        return { id: res.id, stored: true as const };
      },
    }),
    recallMemories: tool({
      description: "Search long-term memory for facts relevant to a question. Returns the best matches first.",
      inputSchema: recallInput,
      execute: async ({ query, k }, { abortSignal }) => {
        const res = await recall(client, options, query, Math.min(k ?? 5, maxK), abortSignal);
        return {
          memories: res.memories.map((m) => ({
            id: m.id,
            content: m.content,
            score: m.score,
            factType: m.fact_type ?? null,
            created: m.created ?? null,
            session: m.session ?? null,
          })),
          partial: res.partial,
        };
      },
    }),
  };
}
