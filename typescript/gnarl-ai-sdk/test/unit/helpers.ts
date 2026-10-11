import type { GnarlClient, Memory, RecallRequest, RememberRequest } from "gnarl-client";
import { vi } from "vitest";

/**
 * A stand-in GnarlClient: `memory.remember` stores in a list, `memory.recall`
 * scores by shared words and honours the user/session scope the node enforces.
 */
export function fakeClient(opts: { partial?: boolean } = {}) {
  const stored: (RememberRequest & { id: string })[] = [];
  const remember = vi.fn(async (req: RememberRequest) => {
    const id = `m${stored.length + 1}`;
    stored.push({ ...req, id });
    return { id, namespace: req.namespace ?? "default", user: req.user ?? "default", embedder: "fake" };
  });
  const recall = vi.fn(async (req: RecallRequest) => {
    const words = new Set(req.query.toLowerCase().match(/[a-z0-9]+/g) ?? []);
    const memories: Memory[] = stored
      .filter((m) => (req.user ? m.user === req.user : true))
      .filter((m) => (req.session ? m.session === req.session : true))
      .map((m) => {
        const hits = (m.content.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => words.has(w)).length;
        return {
          id: m.id,
          content: m.content,
          score: hits / (words.size || 1),
          fact_type: m.fact_type ?? null,
          session: m.session ?? null,
        };
      })
      .filter((m) => m.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, req.k ?? 5);
    return {
      namespace: req.namespace ?? "default",
      count: memories.length,
      embedder: "fake",
      memories,
      partial: opts.partial === true,
      coverage: { expected_claims: 4, served_claims: opts.partial ? 3 : 4, skipped_claims: [] },
    };
  });
  const client = { memory: { remember, recall } } as unknown as GnarlClient;
  return { client, stored, remember, recall };
}

export const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};
