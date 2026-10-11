import { generateText, stepCountIs, streamText, tool, wrapLanguageModel } from "ai";
import { convertArrayToReadableStream, MockLanguageModelV4 } from "ai/test";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { formatMemories, gnarlMemoryMiddleware, PARTIAL_NOTE } from "../../src/index.js";
import { fakeClient, usage } from "./helpers.js";

function textModel(reply = "ok") {
  return new MockLanguageModelV4({
    doGenerate: async () => ({
      content: [{ type: "text", text: reply }],
      finishReason: { unified: "stop", raw: "stop" },
      usage,
      warnings: [],
    }),
    doStream: async () => ({
      stream: convertArrayToReadableStream([
        { type: "text-start", id: "t" },
        { type: "text-delta", id: "t", delta: reply.slice(0, 3) },
        { type: "text-delta", id: "t", delta: reply.slice(3) },
        { type: "text-end", id: "t" },
        { type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage },
      ]),
    }),
  });
}

function systemOf(model: MockLanguageModelV4, call = 0): string {
  const prompt = (model.doGenerateCalls[call] ?? model.doStreamCalls[call])?.prompt ?? [];
  return prompt.filter((m) => m.role === "system").map((m) => (m.role === "system" ? m.content : ""))[0] ?? "";
}

describe("gnarlMemoryMiddleware", () => {
  it("recalls for the latest user message and injects into the system prompt", async () => {
    const fake = fakeClient();
    await fake.client.memory.remember({ content: "alice is vegetarian", user: "alice" });
    await fake.client.memory.remember({ content: "bob is vegetarian too", user: "bob" });
    const base = textModel();
    const model = wrapLanguageModel({ model: base, middleware: gnarlMemoryMiddleware({ client: fake.client, user: "alice", k: 3 }) });

    await generateText({ model, system: "You are a cook.", prompt: "what should a vegetarian like alice eat?" });

    expect(fake.recall).toHaveBeenCalledWith({ query: "what should a vegetarian like alice eat?", k: 3, user: "alice" }, undefined);
    const system = systemOf(base);
    expect(system.startsWith("You are a cook.\n\nRelevant long-term memories")).toBe(true);
    expect(system).toContain("- alice is vegetarian");
    expect(system).not.toContain("bob");
    // One system message: providers often accept system content only first.
    expect(base.doGenerateCalls[0]?.prompt.filter((m) => m.role === "system")).toHaveLength(1);
  });

  it("adds a system message when there is none, and leaves the prompt alone with nothing recalled", async () => {
    const fake = fakeClient();
    await fake.client.memory.remember({ content: "the boat is at pier four", user: "u" });
    const base = textModel();
    const model = wrapLanguageModel({ model: base, middleware: gnarlMemoryMiddleware({ client: fake.client, user: "u" }) });
    await generateText({ model, prompt: "where is the boat?" });
    expect(base.doGenerateCalls[0]?.prompt[0]).toMatchObject({ role: "system" });
    await generateText({ model, prompt: "unrelated question" });
    expect(base.doGenerateCalls[1]?.prompt.map((m) => m.role)).toEqual(["user"]);
  });

  it("announces a partial recall and honours minScore and session", async () => {
    const fake = fakeClient({ partial: true });
    await fake.client.memory.remember({ content: "cats are great", user: "u", session: "s1" });
    await fake.client.memory.remember({ content: "cats elsewhere", user: "u", session: "s2" });
    const base = textModel();
    const model = wrapLanguageModel({
      model: base,
      middleware: gnarlMemoryMiddleware({ client: fake.client, user: "u", session: "s1", minScore: 0.5 }),
    });
    await generateText({ model, prompt: "cats great" });
    const system = systemOf(base);
    expect(system).toContain("- cats are great");
    expect(system).not.toContain("elsewhere");
    expect(system.endsWith(PARTIAL_NOTE)).toBe(true);
  });

  it("does not store anything by default", async () => {
    const fake = fakeClient();
    const model = wrapLanguageModel({ model: textModel(), middleware: gnarlMemoryMiddleware({ client: fake.client, user: "u" }) });
    await generateText({ model, prompt: "remember that I like tea" });
    expect(fake.remember).not.toHaveBeenCalled();
  });

  it("stores the turn once per turn, not once per tool step", async () => {
    const fake = fakeClient();
    let step = 0;
    const base = new MockLanguageModelV4({
      doGenerate: async () => {
        step++;
        if (step === 1) {
          return {
            content: [{ type: "tool-call", toolCallId: "c1", toolName: "clock", input: "{}" }],
            finishReason: { unified: "tool-calls", raw: "tool_calls" },
            usage,
            warnings: [],
          };
        }
        return { content: [{ type: "text", text: "It is noon." }], finishReason: { unified: "stop", raw: "stop" }, usage, warnings: [] };
      },
    });
    const model = wrapLanguageModel({
      model: base,
      middleware: gnarlMemoryMiddleware({ client: fake.client, user: "u", remember: "user-and-assistant" }),
    });
    await generateText({
      model,
      prompt: "what time is it?",
      tools: { clock: tool({ inputSchema: z.object({}), execute: async () => "12:00" }) },
      stopWhen: stepCountIs(3),
    });
    expect(base.doGenerateCalls).toHaveLength(2);
    // Both steps recall (the injected context is not carried between steps)...
    expect(fake.recall).toHaveBeenCalledTimes(2);
    // ...but only the first step, which ends with the user's message, stores.
    expect(fake.stored.map((m) => [m.content, m.fact_type, m.user])).toEqual([["what time is it?", "chat.user", "u"]]);
  });

  it("stores the user message and the streamed reply", async () => {
    const fake = fakeClient();
    const model = wrapLanguageModel({
      model: textModel("Lisbon is lovely"),
      middleware: gnarlMemoryMiddleware({ client: fake.client, user: "u", session: "t1", remember: "user-and-assistant" }),
    });
    const result = streamText({ model, prompt: "I moved to Lisbon" });
    expect(await result.text).toBe("Lisbon is lovely");
    expect(fake.stored.map((m) => [m.content, m.fact_type, m.session])).toEqual([
      ["I moved to Lisbon", "chat.user", "t1"],
      ["Lisbon is lovely", "chat.assistant", "t1"],
    ]);
  });

  it("stores only the user message when asked to", async () => {
    const fake = fakeClient();
    const model = wrapLanguageModel({
      model: textModel("noted"),
      middleware: gnarlMemoryMiddleware({ client: fake.client, user: "u", remember: "user" }),
    });
    await generateText({ model, prompt: "I like tea" });
    expect(fake.stored.map((m) => m.content)).toEqual(["I like tea"]);
  });

  it("fails the call when Gnarl is down, unless told to continue", async () => {
    const fake = fakeClient();
    fake.recall.mockRejectedValue(new Error("connection refused"));
    const strict = wrapLanguageModel({ model: textModel(), middleware: gnarlMemoryMiddleware({ client: fake.client }) });
    await expect(generateText({ model: strict, prompt: "hi" })).rejects.toThrow("connection refused");

    const seen = vi.fn();
    const base = textModel("fine");
    const lenient = wrapLanguageModel({ model: base, middleware: gnarlMemoryMiddleware({ client: fake.client, onError: seen }) });
    const res = await generateText({ model: lenient, prompt: "hi" });
    expect(res.text).toBe("fine");
    expect(seen).toHaveBeenCalledOnce();
  });

  it("formats memories", () => {
    expect(formatMemories([], false)).toBe("");
    expect(formatMemories([], true)).toContain(PARTIAL_NOTE);
    expect(formatMemories([{ id: "1", content: "a", score: 1 }], false).split("\n")).toHaveLength(2);
  });
});
