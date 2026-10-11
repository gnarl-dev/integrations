# Changelog

All notable changes to `llama-index-memory-gnarl` are recorded here. Versions follow
[Semantic Versioning](https://semver.org); a release is the tag `llama-index-memory-gnarl-v<version>`.

## 0.1.0 — unreleased

First release.

- `GnarlMemoryBlock`: a memory block for LlamaIndex's `Memory`. Recalls the
  memories relevant to the latest messages; stores the messages `Memory`
  flushes to it; `aremember()` stores a fact directly. A partial recall is
  announced in the text, or raised with `require_complete=True`.
- `GnarlRetriever`: a LlamaIndex retriever over recall, returning scored nodes
  with `partial` and `coverage` metadata.
- Requires `gnarl-client>=0.2.1`; with no address configured the client finds
  the node on this machine.
