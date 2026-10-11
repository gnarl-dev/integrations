# gnarl-ai-sdk

[Gnarl](https://gnarl.dev) agent memory for the [Vercel AI SDK](https://ai-sdk.dev)
(`ai` 7): memory tools a model can call, and a middleware that recalls relevant
memories into every prompt. A thin layer over the published
[`gnarl-client`](https://www.npmjs.com/package/gnarl-client); the node embeds,
so there is no embedding model to configure.

```bash
npm install gnarl-ai-sdk gnarl-client ai zod
```

With no address configured, the client finds the node itself: `$GNARL_URL`,
else the endpoint the node on this machine recorded, else the Gnarly app's
`http://127.0.0.1:43300`. Pass a client to point anywhere else.

```ts
import { generateText, stepCountIs, wrapLanguageModel } from "ai";
import { GnarlClient } from "gnarl-client";
import { gnarlMemoryMiddleware, gnarlMemoryTools } from "gnarl-ai-sdk";

const client = new GnarlClient();                     // finds the local node
const model = wrapLanguageModel({ model: yourModel, middleware: gnarlMemoryMiddleware({ client, user: "alice" }) });
const { text } = await generateText({
  model,
  tools: gnarlMemoryTools({ client, user: "alice" }),  // rememberFact, recallMemories
  stopWhen: stepCountIs(5),
  prompt: "I'm vegetarian. What should I cook tonight?",
});
```

## `gnarlMemoryTools(options)`

Returns `{ rememberFact, recallMemories }`, made with the AI SDK's `tool()` and
zod input schemas.

| tool             | input                                      | output                                   |
| ---------------- | ------------------------------------------ | ---------------------------------------- |
| `rememberFact`   | `{ fact, factType?, pinned? }`             | `{ id, stored: true }`                   |
| `recallMemories` | `{ query, k? }` (`k` ≤ `maxK`, default 10) | `{ memories: [{ id, content, score, factType, created, session }], partial }` |

The scope — `namespace`, `space`, `user`, `session` — is set by you, not the
model, so a model cannot read or write another user's memories. Recall is
scoped to `user` by the node itself.

## `gnarlMemoryMiddleware(options)`

A `LanguageModelMiddleware` for `wrapLanguageModel`. Before each model call it
recalls the `k` (default 5) memories most relevant to the latest user message
and appends them to the system prompt (or adds one). Options:

- `client`, `namespace`, `space`, `user`, `session` — the scope, as above.
- `k`, `minScore` — how many memories, and a score floor.
- `remember` — `"none"` (default), `"user"`, or `"user-and-assistant"`: store
  the turn after a successful call. Only the first call of a turn stores, so a
  multi-step tool loop does not store the same message once per step. A
  streamed reply is stored when the stream completes.
- `format(memories, partial)` — replace the injected text.
- `onError` — `"throw"` (default) fails the call if Gnarl is unreachable;
  `"continue"` or a callback runs the model without memories.

If part of the memory space did not answer, the injected text says memories
may be missing (`partial`), rather than presenting a subset as everything.

Requires Node 22+ and ESM, as `ai` 7 does.

## Development

```bash
npm install
npm run lint && npm run typecheck && npm test && npm run build
GNARL_URL=http://127.0.0.1:<port> npm run test:integration   # a throwaway test node; skipped when unset
```

The tests use the AI SDK's `MockLanguageModelV4` (`ai/test`); no LLM is called.

Licensed under Apache-2.0.
