# langchain-gnarl

LangChain and LangGraph integrations for [Gnarl](https://gnarl.dev) agent memory:
a LangGraph long-term memory store, a LangChain chat history and a LangChain
retriever. Thin wrappers over the published
[`gnarl-client`](https://pypi.org/project/gnarl-client/); the node does the
embedding, so there is no embedding model to configure.

```bash
pip install langchain-gnarl
export GNARL_URL=http://127.0.0.1:43300   # the Gnarly desktop node, or any node
```

```python
from langchain_gnarl import GnarlStore, GnarlChatMessageHistory, GnarlRetriever

store = GnarlStore()                                   # LangGraph BaseStore
store.put(("memories", "alice"), "diet", {"text": "Alice is vegetarian"})
hits = store.search(("memories", "alice"), query="what does alice eat?", limit=3)

history = GnarlChatMessageHistory("thread-42", user="alice")
history.add_user_message("I moved to Lisbon last spring")

docs = GnarlRetriever(namespace="langchain-chat", user="alice").invoke("where does alice live?")
```

Use the store in a graph like any other: `builder.compile(store=GnarlStore())`,
or `create_react_agent(model, tools, store=GnarlStore())`.

## `GnarlStore` (LangGraph `BaseStore`)

`GnarlStore(client=None, *, async_client=None, namespace="langgraph-store", index_fields=None)`

| LangGraph            | Gnarl memory                                                  |
| -------------------- | ------------------------------------------------------------- |
| the store            | one memory `namespace` (default `langgraph-store`)            |
| item namespace tuple | `user` — labels joined with `.` (LangGraph forbids `.` in labels) |
| item key             | `session`                                                     |
| item value           | stored verbatim (JSON); its text is embedded by the node     |
| `search(query=...)`  | `memory.recall`, scoped to `user` — semantic                 |

- Isolation between namespace tuples is enforced by the node: recall is scoped
  to `user` server-side. `("memories", "alice")` never recalls Bob's items.
- `search` with a query over a namespace that has sub-namespaces recalls each
  one and merges by score. Without a query it lists items, newest first.
- `filter=` supports equality and `$eq/$ne/$gt/$gte/$lt/$lte`, applied to the
  recalled candidates.
- Values are embedded by default from every string they contain;
  `index_fields=["text"]` (or a put's own `index=[...]`) narrows that.
  `index=False` is not supported: the node embeds every memory.
- TTL is not supported (`supports_ttl = False`).
- A `put` is visible as soon as it returns. A replaced or deleted item drops
  out of other processes' reads after the node's next refresh (about a second),
  and immediately in this process. `put` is not atomic; two concurrent puts of
  one key both land, and reads return the newer.
- `list_namespaces` and query-less `search` read one row per item.

## `GnarlChatMessageHistory` (LangChain `BaseChatMessageHistory`)

`GnarlChatMessageHistory(session_id, *, client=None, async_client=None, namespace="langchain-chat", user=None)`

One memory per turn (`session` = `session_id`, the full message kept
verbatim). Turns are therefore also recallable semantically across sessions.
`clear()` deletes the session's turns.

## `GnarlRetriever` (LangChain `BaseRetriever`)

`GnarlRetriever(client=None, async_client=None, namespace=None, space=None, user=None, session=None, k=4, require_complete=False)`

Returns `Document`s whose metadata carries `id`, `score`, `knn_score`,
`fact_type`, `session`, `agent`, `created`, `tags`, `namespace`, and the
recall's `partial` and `coverage`. When part of the memory space did not
answer, `partial` is `True` — the documents are a subset, and a missing memory
may still exist. `require_complete=True` raises `IncompleteRecall` instead.

## Clients

Every class takes an optional `gnarl.Client` and `gnarl.AsyncClient`. With
neither, they are built from `$GNARL_URL` (and `$GNARL_TOKEN`) on first use.
With only a sync client, async methods run it in a worker thread.

## Development

```bash
pip install -e '.[dev]'
ruff check . && mypy && pyright
pytest tests/unit
GNARL_URL=http://127.0.0.1:43300 pytest tests/integration   # skipped without a node
```

Licensed under Apache-2.0.
