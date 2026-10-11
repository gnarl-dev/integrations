import { generateText, stepCountIs } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, it } from "vitest";
import { gnarlMemoryTools } from "../../src/index.js";
import { fakeClient, usage } from "./helpers.js";

const opts = { toolCallId: "c", messages: [], context: {} } as never;

describe("gnarlMemoryTools", () => {
  it("rememberFact stores in the developer's scope", async () => {
    const fake = fakeClient();
    const tools = gnarlMemoryTools({ client: fake.client, namespace: "agent", user: "alice", session: "t1" });
    const out = await tools.rememberFact.execute?.({ fact: "Alice is vegetarian", factType: "preference" }, opts);
    expect(out).toEqual({ id: "m1", stored: true });
    expect(fake.remember).toHaveBeenCalledWith(
      { content: "Alice is vegetarian", agent: "ai-sdk", namespace: "agent", user: "alice", session: "t1", fact_type: "preference" },
      undefined,
    );
  });

  it("recallMemories returns scored memories and partial", async () => {
    const fake = fakeClient({ partial: true });
    const tools = gnarlMemoryTools({ client: fake.client, user: "alice" });
    await tools.rememberFact.execute?.({ fact: "alice lives in lisbon" }, opts);
    const out = await tools.recallMemories.execute?.({ query: "where does alice live" }, opts);
    expect(out).toMatchObject({ partial: true, memories: [{ content: "alice lives in lisbon", factType: null }] });
    expect(fake.recall).toHaveBeenCalledWith({ query: "where does alice live", k: 5, user: "alice" }, undefined);
  });

  it("caps k, and never lets the model pick the user", async () => {
    const tools = gnarlMemoryTools({ client: fakeClient().client, user: "alice", maxK: 3 });
    const schema = tools.recallMemories.inputSchema as { safeParse: (v: unknown) => { success: boolean } };
    expect(schema.safeParse({ query: "x", k: 3 }).success).toBe(true);
    expect(schema.safeParse({ query: "x", k: 4 }).success).toBe(false);
    expect(schema.safeParse({ query: "" }).success).toBe(false);
    const rememberSchema = tools.rememberFact.inputSchema as unknown as { shape: Record<string, unknown> };
    expect(Object.keys(rememberSchema.shape).sort()).toEqual(["fact", "factType", "pinned"]);
  });

  it("works as tools in a generateText loop", async () => {
    const fake = fakeClient();
    let step = 0;
    const model = new MockLanguageModelV4({
      doGenerate: async () => {
        step++;
        if (step === 1) {
          return {
            content: [{ type: "tool-call", toolCallId: "c1", toolName: "rememberFact", input: JSON.stringify({ fact: "user likes tea" }) }],
            finishReason: { unified: "tool-calls", raw: "tool_calls" },
            usage,
            warnings: [],
          };
        }
        return { content: [{ type: "text", text: "Noted." }], finishReason: { unified: "stop", raw: "stop" }, usage, warnings: [] };
      },
    });
    const res = await generateText({
      model,
      prompt: "I like tea",
      tools: gnarlMemoryTools({ client: fake.client, user: "u" }),
      stopWhen: stepCountIs(3),
    });
    expect(res.text).toBe("Noted.");
    expect(fake.stored.map((m) => [m.content, m.user])).toEqual([["user likes tea", "u"]]);
  });
});
