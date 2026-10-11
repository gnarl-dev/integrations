import type { LanguageModelMiddleware } from "ai";
import type { Memory } from "gnarl-client";
import { clientOf, type GnarlMemoryScope, PARTIAL_NOTE, recall, scopeFields } from "./recall.js";

type TransformParams = NonNullable<LanguageModelMiddleware["transformParams"]>;
type CallOptions = Parameters<TransformParams>[0]["params"];
type Prompt = CallOptions["prompt"];
type WrapGenerate = NonNullable<LanguageModelMiddleware["wrapGenerate"]>;
type WrapStream = NonNullable<LanguageModelMiddleware["wrapStream"]>;
type StreamResult = Awaited<ReturnType<Parameters<WrapStream>[0]["doStream"]>>;
type StreamPart = StreamResult["stream"] extends ReadableStream<infer P> ? P : never;

export interface GnarlMemoryMiddlewareOptions extends GnarlMemoryScope {
  /** How many memories to recall for each call. Default 5. */
  k?: number;
  /** Drop recalled memories scoring below this. */
  minScore?: number;
  /**
   * What to store after a successful call:
   * - `"none"` (default): recall only; store through `gnarlMemoryTools` or the client.
   * - `"user"`: the user message that started the turn.
   * - `"user-and-assistant"`: that, and the model's text reply.
   *
   * Only the first model call of a turn (the one whose prompt ends with the
   * user's message) stores anything, so a multi-step tool loop does not store
   * the same message once per step.
   */
  remember?: "none" | "user" | "user-and-assistant";
  /**
   * Builds the text injected into the system prompt. Default: a short header
   * and one `- ` line per memory.
   */
  format?: (memories: Memory[], partial: boolean) => string;
  /**
   * `"throw"` (default) fails the model call when Gnarl cannot be reached.
   * `"continue"` calls the model without memories and reports the error here.
   */
  onError?: "throw" | "continue" | ((error: unknown) => void);
  /** `agent` recorded on stored memories. Default `ai-sdk`. */
  agent?: string;
}

/** The default injected text. */
export function formatMemories(memories: Memory[], partial: boolean): string {
  if (memories.length === 0 && !partial) return "";
  const lines = ["Relevant long-term memories (recalled from Gnarl; may be outdated):"];
  for (const m of memories) lines.push(`- ${m.content}`);
  if (partial) lines.push(PARTIAL_NOTE);
  return lines.join("\n");
}

function textOf(message: Prompt[number]): string {
  if (message.role === "system") return message.content;
  const parts: string[] = [];
  for (const part of message.content) {
    if (part.type === "text") parts.push(part.text);
  }
  return parts.join(" ").trim();
}

function lastUserText(prompt: Prompt): string {
  for (let i = prompt.length - 1; i >= 0; i--) {
    const m = prompt[i];
    if (m?.role === "user") return textOf(m);
  }
  return "";
}

/** True when the prompt ends with the user's message: the first call of a turn. */
function startsTurn(prompt: Prompt): boolean {
  return prompt[prompt.length - 1]?.role === "user";
}

function inject(prompt: Prompt, text: string): Prompt {
  const first = prompt[0];
  // Folded into a leading system message rather than added as a second one:
  // several providers accept system content only at the start of a prompt.
  if (first?.role === "system") {
    return [{ ...first, content: `${first.content}\n\n${text}` }, ...prompt.slice(1)];
  }
  return [{ role: "system", content: text }, ...prompt];
}

/**
 * Language-model middleware that recalls Gnarl memories relevant to the
 * latest user message and adds them to the system prompt; optionally it
 * stores the turn afterwards.
 *
 * ```ts
 * const model = wrapLanguageModel({
 *   model: openai("gpt-4.1"),
 *   middleware: gnarlMemoryMiddleware({ user: "alice" }),
 * });
 * ```
 */
export function gnarlMemoryMiddleware(options: GnarlMemoryMiddlewareOptions = {}): LanguageModelMiddleware {
  const client = clientOf(options);
  const k = options.k ?? 5;
  const format = options.format ?? formatMemories;
  const mode = options.remember ?? "none";
  const agent = options.agent ?? "ai-sdk";

  const fail = (error: unknown): void => {
    const onError = options.onError ?? "throw";
    if (onError === "throw") throw error;
    if (typeof onError === "function") onError(error);
  };

  const store = async (content: string, role: "user" | "assistant", signal?: AbortSignal): Promise<void> => {
    if (!content.trim()) return;
    try {
      await client.memory.remember(
        {
          content,
          agent,
          fact_type: `chat.${role}`,
          ...scopeFields(options),
          ...(options.session !== undefined ? { session: options.session } : {}),
        },
        signal ? { signal } : undefined,
      );
    } catch (error) {
      fail(error);
    }
  };

  const transformParams: TransformParams = async ({ params }) => {
    const query = lastUserText(params.prompt);
    if (!query) return params;
    let text = "";
    try {
      const res = await recall(client, options, query, k, params.abortSignal);
      const min = options.minScore;
      const kept = min === undefined ? res.memories : res.memories.filter((m) => m.score >= min);
      text = format(kept, res.partial);
    } catch (error) {
      fail(error);
      return params;
    }
    return text ? { ...params, prompt: inject(params.prompt, text) } : params;
  };

  const wrapGenerate: WrapGenerate = async ({ doGenerate, params }) => {
    const result = await doGenerate();
    if (mode !== "none" && startsTurn(params.prompt)) {
      await store(lastUserText(params.prompt), "user", params.abortSignal);
      if (mode === "user-and-assistant") {
        const reply = result.content
          .map((c) => (c.type === "text" ? c.text : ""))
          .join("")
          .trim();
        await store(reply, "assistant", params.abortSignal);
      }
    }
    return result;
  };

  const wrapStream: WrapStream = async ({ doStream, params }) => {
    const result = await doStream();
    if (mode === "none" || !startsTurn(params.prompt)) return result;
    await store(lastUserText(params.prompt), "user", params.abortSignal);
    if (mode !== "user-and-assistant") return result;
    let reply = "";
    const capture = new TransformStream<StreamPart, StreamPart>({
      transform(part, controller) {
        if (part.type === "text-delta") reply += part.delta;
        controller.enqueue(part);
      },
      async flush() {
        // Stored when the stream completes, before it reports done, so a
        // caller that awaits the full response also awaits the write.
        await store(reply, "assistant", params.abortSignal);
      },
    });
    return { ...result, stream: result.stream.pipeThrough(capture) };
  };

  return { specificationVersion: "v4", transformParams, wrapGenerate, wrapStream };
}
