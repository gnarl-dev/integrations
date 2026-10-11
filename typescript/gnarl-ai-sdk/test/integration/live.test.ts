/**
 * Against a real node at $GNARL_URL (skipped without one): real embeddings,
 * real recall, real per-user isolation. The language model is the AI SDK's
 * MockLanguageModelV4 — no LLM is called — so what is verified is what the
 * middleware and tools send to Gnarl and to the model, not a model's answer.
 */
import { generateText, stepCountIs, wrapLanguageModel } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { GnarlClient } from "gnarl-client";
import { beforeAll, describe, expect, it } from "vitest";
import { gnarlMemoryMiddleware, gnarlMemoryTools } from "../../src/index.js";

const url = process.env.GNARL_URL;
let reachable = false;
if (url) {
  try {
    await new GnarlClient({ url }).ping();
    reachable = true;
  } catch {
    reachable = false;
  }
}

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};
const FACTS = [
  "Alice is vegetarian and her favourite dish is falafel",
  "Alice lives in Lisbon, near the river",
  "Alice has a grey cat called Miso",
];
const opts = { toolCallId: "c", messages: [], context: {} } as never;

describe.skipIf(!reachable)("against a live node", () => {
  const run = Math.random().toString(36).slice(2, 10);
  const namespace = `it-ai-${run}`;
  const client = new GnarlClient({ url });
  const alice = gnarlMemoryTools({ client, namespace, user: "alice" });
  const bob = gnarlMemoryTools({ client, namespace, user: "bob" });

  beforeAll(async () => {
    for (const fact of FACTS) await alice.rememberFact.execute?.({ fact }, opts);
    await bob.rememberFact.execute?.({ fact: "Bob plays the cello" }, opts);
  });

  it("recallMemories finds each fact by meaning, and only Alice's", async () => {
    const ask = async (query: string) => {
      const out = await alice.recallMemories.execute?.({ query, k: 1 }, opts);
      if (!out || Symbol.asyncIterator in out) throw new Error("unexpected tool output");
      expect(out.partial).toBe(false);
      return out.memories[0]?.content;
    };
    expect(await ask("what food should I cook for her?")).toBe(FACTS[0]);
    expect(await ask("which city is her home in?")).toBe(FACTS[1]);
    expect(await ask("does she own an animal?")).toBe(FACTS[2]);

    const bobs = await bob.recallMemories.execute?.({ query: "falafel lisbon cat", k: 10 }, opts);
    if (!bobs || Symbol.asyncIterator in bobs) throw new Error("unexpected tool output");
    expect(bobs.memories.map((m) => m.content)).toEqual(["Bob plays the cello"]);
  });

  it("the middleware puts Alice's relevant memories, and no one else's, in the prompt", async () => {
    const base = new MockLanguageModelV4({
      doGenerate: async () => ({
        content: [{ type: "text", text: "ok" }],
        finishReason: { unified: "stop", raw: "stop" },
        usage,
        warnings: [],
      }),
    });
    const model = wrapLanguageModel({ model: base, middleware: gnarlMemoryMiddleware({ client, namespace, user: "alice", k: 1 }) });
    await generateText({ model, system: "You are helpful.", prompt: "Suggest a dinner she would enjoy." });
    const system = base.doGenerateCalls[0]?.prompt[0];
    expect(system?.role).toBe("system");
    const text = system?.role === "system" ? system.content : "";
    expect(text).toContain(`- ${FACTS[0]}`);
    expect(text).not.toContain("cello");

    const bobBase = new MockLanguageModelV4({
      doGenerate: async () => ({
        content: [{ type: "text", text: "ok" }],
        finishReason: { unified: "stop", raw: "stop" },
        usage,
        warnings: [],
      }),
    });
    const bobModel = wrapLanguageModel({ model: bobBase, middleware: gnarlMemoryMiddleware({ client, namespace, user: "bob", k: 5 }) });
    await generateText({ model: bobModel, prompt: "what does alice like to eat?" });
    const bobSystem = bobBase.doGenerateCalls[0]?.prompt[0];
    const bobText = bobSystem?.role === "system" ? bobSystem.content : "";
    expect(bobText).not.toMatch(/falafel|Lisbon|Miso/);
  });

  it("a model's rememberFact call is stored and recalled on the next turn", async () => {
    const user = `carol-${run}`;
    let step = 0;
    const base = new MockLanguageModelV4({
      doGenerate: async () => {
        step++;
        if (step === 1) {
          return {
            content: [
              {
                type: "tool-call",
                toolCallId: "c1",
                toolName: "rememberFact",
                input: JSON.stringify({ fact: "Carol is allergic to peanuts" }),
              },
            ],
            finishReason: { unified: "tool-calls", raw: "tool_calls" },
            usage,
            warnings: [],
          };
        }
        return { content: [{ type: "text", text: "Noted." }], finishReason: { unified: "stop", raw: "stop" }, usage, warnings: [] };
      },
    });
    const model = wrapLanguageModel({ model: base, middleware: gnarlMemoryMiddleware({ client, namespace, user }) });
    await generateText({
      model,
      prompt: "Please remember I'm allergic to peanuts",
      tools: gnarlMemoryTools({ client, namespace, user }),
      stopWhen: stepCountIs(3),
    });
    await generateText({ model, prompt: "Can I eat satay sauce?" });
    const last = base.doGenerateCalls.at(-1)?.prompt[0];
    expect(last?.role === "system" ? last.content : "").toContain("Carol is allergic to peanuts");
  });

  it("remember: user-and-assistant stores the turn", async () => {
    const user = `dave-${run}`;
    const base = new MockLanguageModelV4({
      doGenerate: async () => ({
        content: [{ type: "text", text: "Oslo has long winters." }],
        finishReason: { unified: "stop", raw: "stop" },
        usage,
        warnings: [],
      }),
    });
    const model = wrapLanguageModel({
      model: base,
      middleware: gnarlMemoryMiddleware({ client, namespace, user, remember: "user-and-assistant" }),
    });
    await generateText({ model, prompt: "I am moving to Oslo next month" });
    const res = await client.memory.recall({ query: "where is he moving?", namespace, user, k: 5 });
    expect(res.memories.map((m) => m.content).sort()).toEqual(["I am moving to Oslo next month", "Oslo has long winters."]);
  });
});
