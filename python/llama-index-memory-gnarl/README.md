# llama-index-memory-gnarl

[Gnarl](https://gnarl.dev) agent memory for LlamaIndex: a memory block for the
`Memory` API and a retriever. Thin wrappers over the published
[`gnarl-client`](https://pypi.org/project/gnarl-client/); the node embeds, so
there is no embedding model to configure.

```bash
pip install llama-index-memory-gnarl
```

With no address configured, the client finds the node itself: `$GNARL_URL`,
else the endpoint the node on this machine recorded, else the Gnarly app's
`http://127.0.0.1:43300`. Pass a client to point anywhere else.

```python
from llama_index.core.memory import Memory
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.memory.gnarl import GnarlMemoryBlock

block = GnarlMemoryBlock(namespace="agent-memory", user="alice")
await block.aremember("Alice is vegetarian")                 # store a fact directly
memory = Memory.from_defaults(session_id="thread-42", memory_blocks=[block])
agent = FunctionAgent(llm=llm, tools=[])
await agent.run("Suggest a dinner for me", memory=memory)  # recalls "vegetarian"
```

## `GnarlMemoryBlock`

`GnarlMemoryBlock(client=None, namespace=None, space=None, user=None, session=None, k=5, min_score=None, retrieval_context_window=1, store_roles=[USER, ASSISTANT], require_complete=False, name="GnarlMemory", priority=0)`

- **get**: recalls the `k` memories most relevant to the latest
  `retrieval_context_window` messages and returns them as `- …` lines, which
  `Memory` places in its memory-blocks template.
- **put**: `Memory` hands a block the messages it flushes out of short-term
  history (when the token limit is reached). Each user/assistant message is
  stored as one memory with its `session`. To store every turn immediately,
  call `block.aput(messages)` yourself, or `aremember(text)` for a single fact.
- Recall is scoped to `user` by the node, and spans all sessions unless
  `session` is set.
- If part of the memory space did not answer, the text ends with a note saying
  memories may be missing; `require_complete=True` raises `IncompleteRecall`.

## `GnarlRetriever`

`GnarlRetriever(client=None, *, async_client=None, namespace=None, space=None, user=None, session=None, k=4, require_complete=False)`

A `BaseRetriever` returning `NodeWithScore`s whose metadata carries `score`,
`knn_score`, `fact_type`, `session`, `agent`, `created`, `tags`, `namespace`,
`partial` and `coverage`. Usable anywhere LlamaIndex takes a retriever, e.g.
`RetrieverQueryEngine.from_args(GnarlRetriever(user="alice"))`.

## Development

```bash
pip install -e '.[dev]'
ruff check . && mypy && pyright
pytest tests/unit
GNARL_URL=http://127.0.0.1:<port> pytest tests/integration   # a throwaway test node; skipped when unset
```

Licensed under Apache-2.0.
