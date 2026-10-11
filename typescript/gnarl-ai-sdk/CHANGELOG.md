# Changelog

All notable changes to `gnarl-ai-sdk` are recorded here. Versions follow
[Semantic Versioning](https://semver.org); a release is the tag `gnarl-ai-sdk-v<version>`.

## 0.1.0 — unreleased

First release.

- `gnarlMemoryTools`: `rememberFact` and `recallMemories` tools (AI SDK `tool()`
  with zod schemas) in a developer-fixed scope.
- `gnarlMemoryMiddleware`: a `LanguageModelMiddleware` that recalls memories for
  the latest user message into the system prompt and can store the turn
  (`remember: "user" | "user-and-assistant"`), once per turn.
- ESM only; requires Node 22, `ai` ^7.0.137 and `gnarl-client` ^0.1.2. With no
  address configured the client finds the node on this machine.
