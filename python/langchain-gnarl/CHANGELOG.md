# Changelog

All notable changes to `langchain-gnarl` are recorded here. Versions follow
[Semantic Versioning](https://semver.org); a release is the tag `langchain-gnarl-v<version>`.

## 0.1.0 — unreleased

First release.

- `GnarlStore`: a LangGraph `BaseStore` over Gnarl memory. Namespace tuples map
  to the memory `user`, so the node scopes recall per namespace. Supports
  get/put/delete, semantic `search` (with sub-namespace fan-out, `filter`,
  `offset`/`limit`) and `list_namespaces`, sync and async. No TTL.
- `GnarlChatMessageHistory`: LangChain chat history, one memory per turn, in
  order, with user and session isolation.
- `GnarlRetriever`: LangChain retriever over recall; documents carry score,
  `partial` and `coverage`, and `require_complete=True` raises `IncompleteRecall`.
- Requires `gnarl-client>=0.2.1`; with no address configured the client finds
  the node on this machine.
