"""An in-process stand-in for ``gnarl.Client`` / ``gnarl.AsyncClient``.

It implements only the calls the adapters make, and answers them in the wire
shapes the published client returns (its own pydantic models), including the
node behaviours the adapters must absorb:

- a recall over a never-written namespace answers with no ``embedder``
  (what released nodes before rc30 send; gnarl-client 0.2.1 accepts it);
- with ``delete_lag=True``, a deleted document stays searchable (the node
  hides a delete only at its next refresh).
"""

from __future__ import annotations

import fnmatch
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from gnarl import AsyncClient, Client, Hit, NotFound, RecallResult, RememberResult, SearchResult
from gnarl._models import SearchCoverage, TotalHits


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class _Memory:
    def __init__(self, fake: FakeGnarl) -> None:
        self._f = fake

    def remember(self, content: str, **kw: Any) -> RememberResult:
        self._f.calls.append(("remember", content, kw))
        if not content.strip():
            raise ValueError("empty content")
        ns = kw.get("namespace") or "default"
        doc_id = str(uuid.uuid4())
        tags = dict(kw.get("tags") or {})
        tags["space"] = "personal"
        self._f.seq += 1
        doc = {
            "content": content,
            "user": kw.get("user") or "default-user",
            "session": kw.get("session"),
            "fact_type": kw.get("fact_type"),
            "agent": kw.get("agent") or "lucenia-edge",
            "tags": ",".join(f"{k}={v}" for k, v in sorted(tags.items())),
            "pinned": False,
            "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        self._f.indexes.setdefault(ns, {})[doc_id] = doc
        return RememberResult.model_validate(
            {"id": doc_id, "namespace": ns, "user": doc["user"], "embedder": "fake"}
        )

    def recall(self, query: str, **kw: Any) -> RecallResult:
        self._f.calls.append(("recall", query, kw))
        ns = kw.get("namespace") or "default"
        if ns not in self._f.indexes:
            # What a pre-rc30 node sends for a namespace with no index: no
            # `embedder`. The adapters must not trip on it.
            return RecallResult.model_validate({"namespace": ns, "count": 0, "memories": []})
        qt = _tokens(query)
        scored = []
        for doc_id, d in self._f.visible(ns):
            if kw.get("user") and d["user"] != kw["user"]:
                continue
            if kw.get("session") and d["session"] != kw["session"]:
                continue
            overlap = len(qt & _tokens(d["content"]))
            scored.append((overlap / (len(qt) or 1), doc_id, d))
        scored.sort(key=lambda s: s[0], reverse=True)
        mems = [
            {
                "id": doc_id, "score": score, "knn_score": score, "content": d["content"],
                "fact_type": d["fact_type"], "pinned": False, "created": d["created"],
                "session": d["session"], "agent": d["agent"], "tags": d["tags"],
            }
            for score, doc_id, d in scored[: kw.get("k") or 5]
        ]
        return RecallResult.model_validate(
            {
                "namespace": ns, "count": len(mems), "memories": mems, "embedder": "fake",
                "partial": self._f.partial,
                "coverage": {"expected_claims": 4, "served_claims": 3 if self._f.partial else 4,
                             "skipped_claims": []},
            }
        )


class FakeGnarl(Client):
    # A real Client subclass so framework type checks accept it; __init__ is
    # replaced, so no connection pool is ever opened.

    def __init__(self, *, delete_lag: bool = False, partial: bool = False) -> None:
        self.indexes: dict[str, dict[str, dict[str, Any]]] = {}
        self.deleted: set[str] = set()
        self.delete_lag = delete_lag
        self.partial = partial
        self.calls: list[tuple[str, Any, Any]] = []
        self.seq = 0
        self.memory = _Memory(self)  # type: ignore[assignment]

    def visible(self, ns: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (i, d) for i, d in self.indexes.get(ns, {}).items()
            if self.delete_lag or i not in self.deleted
        ]

    def search(self, index: str, query: Any = None, *, size: int | None = None,
               from_: int | None = None, **_: Any) -> SearchResult:
        self.calls.append(("search", index, query))
        if index not in self.indexes:
            raise NotFound(type="index_not_found", reason=f"index '{index}' not found")
        spec = query.model_dump(exclude_none=True, by_alias=True) if query is not None else {}
        rows = [(i, d) for i, d in self.visible(index) if _match(spec, d)]
        start = from_ or 0
        rows = rows[start : start + (size if size is not None else 10)]
        return SearchResult(
            hits=[Hit.model_validate({"_id": i, "_score": 0.0, "_source": d}) for i, d in rows],
            total=TotalHits.model_validate({"value": len(rows), "relation": "eq"}),
            partial=False,
            coverage=SearchCoverage.model_validate(
                {"expected_claims": 1, "served_claims": 1, "skipped_claims": []}
            ),
            took=0,
        )

    def iter_search(self, index: str, query: Any = None, *, page_size: int = 500,
                    **kw: Any) -> Any:
        start = 0
        while True:
            page = self.search(index, query, size=page_size, from_=start)
            yield from page.hits
            if len(page.hits) < page_size:
                return
            start += page_size

    def delete_document(self, index: str, doc_id: str) -> None:
        self.calls.append(("delete", index, doc_id))
        if index not in self.indexes:
            raise NotFound(type="index_not_found", reason="no index")
        self.deleted.add(doc_id)
        if not self.delete_lag:
            self.indexes[index].pop(doc_id, None)


def _match(spec: dict[str, Any], doc: dict[str, Any]) -> bool:
    if not spec or "match_all" in spec:
        return True
    if "term" in spec:
        ((field, value),) = spec["term"].items()
        return bool(doc.get(field) == value)
    if "wildcard" in spec:
        ((field, pattern),) = spec["wildcard"].items()
        v = doc.get(field)
        return isinstance(v, str) and fnmatch.fnmatchcase(v, pattern)
    if "bool" in spec:
        b = spec["bool"]
        required = b.get("must", []) + b.get("filter", [])
        if not all(_match(c, doc) for c in required):
            return False
        if any(_match(c, doc) for c in b.get("must_not", [])):
            return False
        should = b.get("should", [])
        # Elasticsearch semantics, which the node follows: `should` is
        # required only when there is nothing else to require.
        if should and not required:
            return any(_match(c, doc) for c in should)
        return True
    raise AssertionError(f"fake does not understand query {spec}")


class AsyncFakeGnarl(AsyncClient):
    """The same fake, awaited."""

    def __init__(self, sync: FakeGnarl) -> None:
        self.sync = sync
        self.memory = _AsyncMemory(sync)  # type: ignore[assignment]

    async def search(self, *a: Any, **kw: Any) -> SearchResult:
        return self.sync.search(*a, **kw)

    async def iter_search(self, *a: Any, **kw: Any) -> Any:
        for h in self.sync.iter_search(*a, **kw):
            yield h

    async def delete_document(self, *a: Any) -> None:
        self.sync.delete_document(*a)


class _AsyncMemory:
    def __init__(self, sync: FakeGnarl) -> None:
        self._s = sync

    async def remember(self, *a: Any, **kw: Any) -> RememberResult:
        return self._s.memory.remember(*a, **kw)

    async def recall(self, *a: Any, **kw: Any) -> RecallResult:
        return self._s.memory.recall(*a, **kw)
