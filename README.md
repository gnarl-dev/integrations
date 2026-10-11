# Gnarl framework integrations

Use [Gnarl](https://gnarl.dev) as the memory layer of an agent built with
LangChain/LangGraph, LlamaIndex or the Vercel AI SDK. Each package is a thin
adapter over the published Gnarl client for its language — `gnarl-client` on
PyPI and on npm — and never speaks the HTTP API itself. The node embeds, so no
adapter needs an embedding model.

| Package | Directory | Framework | Provides |
| --- | --- | --- | --- |
| `langchain-gnarl` (PyPI) | [`python/langchain-gnarl`](python/langchain-gnarl) | LangGraph, LangChain | `GnarlStore` (`BaseStore`), `GnarlChatMessageHistory`, `GnarlRetriever` |
| `llama-index-memory-gnarl` (PyPI) | [`python/llama-index-memory-gnarl`](python/llama-index-memory-gnarl) | LlamaIndex | `GnarlMemoryBlock` (`Memory` block), `GnarlRetriever` |
| `gnarl-ai-sdk` (npm) | [`typescript/gnarl-ai-sdk`](typescript/gnarl-ai-sdk) | Vercel AI SDK 7 | `gnarlMemoryTools`, `gnarlMemoryMiddleware` |

Point any of them at a node with `GNARL_URL` (the Gnarly desktop node listens
on `http://127.0.0.1:43300`).

## Why one repository

Three packages, one place: they wrap the same memory API and should change
together when it does, and one CI file per language covers all of them. Each
package is still self-contained — its own manifest, version, README, LICENSE
and tests — so each publishes independently, and splitting one out later is a
`git subtree split`.

## Layout

```
python/langchain-gnarl/            pyproject (hatchling), src/langchain_gnarl, tests/{unit,integration}
python/llama-index-memory-gnarl/   pyproject (hatchling), src/llama_index/memory/gnarl, tests/{unit,integration}
typescript/gnarl-ai-sdk/           package.json (tsup, ESM), src, test/{unit,integration}
scripts/ci-node.sh                 starts a released node for CI integration tests
.github/workflows/                 python.yml, typescript.yml
```

## Tests

Unit tests use a stand-in client and need nothing running. Integration tests
run against a real node at `$GNARL_URL` and skip when it is unset or
unreachable; CI starts a released node and fails if they skipped.

```bash
python3 -m venv .venv
.venv/bin/pip install -e 'python/langchain-gnarl[dev]' -e 'python/llama-index-memory-gnarl[dev]'
(cd python/langchain-gnarl && ../../.venv/bin/pytest)
(cd python/llama-index-memory-gnarl && ../../.venv/bin/pytest)
(cd typescript/gnarl-ai-sdk && npm install && npm test && npm run test:integration)
```

Licensed under Apache-2.0.
