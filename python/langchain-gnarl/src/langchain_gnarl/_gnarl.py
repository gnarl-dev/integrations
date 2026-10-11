"""What every adapter in this package needs from the Gnarl client.

Everything here goes through the published ``gnarl-client``; nothing speaks
HTTP itself. The helpers exist for the handful of places where a framework
needs something the memory API does not hand over directly — a lookup by key,
a delete, a listing — and for behaviours of the node an adapter has to absorb
rather than pass on to its caller (see :func:`wildcard_prefix` and
:class:`Tombstones`).
"""

from __future__ import annotations

import base64
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from gnarl import AsyncClient, Client, Hit, NotFound, Query
from gnarl import RecallResult as _GnarlRecall
from gnarl import query as q

#: One recalled memory, as the published client types it. Reached through the
#: public ``RecallResult`` model: the class itself lives in a private module.
RecalledMemory = Any

__all__ = [
    "Clients",
    "IncompleteRecall",
    "RecallResult",
    "Tombstones",
    "b64d",
    "b64e",
    "check_namespace",
    "parse_created",
    "parse_tags",
    "recall",
    "arecall",
    "wildcard_prefix",
]

# The node derives a memory namespace's INDEX name by lower-casing it and
# replacing anything outside [a-z0-9_-]. Re-deriving that rule here would let it
# drift; instead an adapter only accepts a name that the rule leaves alone, so
# the namespace and the index it lives in are the same string.
_NS_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def check_namespace(namespace: str) -> str:
    """Return ``namespace`` if it can be used as-is as the node's index name."""
    if not _NS_RE.match(namespace):
        raise ValueError(
            f"gnarl: namespace {namespace!r} must match {_NS_RE.pattern} "
            "(lower-case letters, digits, '_' and '-'; at most 64 characters)"
        )
    return namespace


class Clients:
    """A sync and an async client, either supplied or built on first use.

    A client built here finds the node itself: ``$GNARL_URL``, else the endpoint
    the node on this machine recorded, else the Gnarly app's default. Built
    lazily, so constructing an adapter never needs a reachable node and an
    adapter used only synchronously never opens an async connection pool.
    """

    def __init__(self, client: Client | None, async_client: AsyncClient | None) -> None:
        self._sync = client
        self._async = async_client
        self._lock = threading.Lock()

    @property
    def sync(self) -> Client:
        if self._sync is None:
            with self._lock:
                if self._sync is None:
                    self._sync = Client()
        return self._sync

    @property
    def aio(self) -> AsyncClient:
        if self._async is None:
            with self._lock:
                if self._async is None:
                    self._async = AsyncClient()
        return self._async


@dataclass
class RecallResult:
    """``memory.recall``'s answer, with completeness carried alongside it.

    ``partial`` is True when some part of the memory space did not answer. A
    memory that was never reached is otherwise indistinguishable from one that
    was never stored, so the adapters surface it instead of dropping it.
    """

    memories: list[RecalledMemory]
    partial: bool = False
    coverage: dict[str, Any] | None = None
    namespace: str | None = None


class IncompleteRecall(Exception):
    """A recall came back partial and the caller asked for completeness.

    Carries the partial :class:`RecallResult` as ``result``, so it can still be
    inspected or used deliberately.
    """

    def __init__(self, result: RecallResult) -> None:
        self.result = result
        super().__init__(
            f"gnarl: recall over {result.namespace!r} was partial "
            f"(coverage: {result.coverage}); some memories may be missing"
        )


def _to_result(resp: _GnarlRecall) -> RecallResult:
    extra = resp.model_extra or {}
    return RecallResult(
        memories=list(resp.memories),
        partial=bool(extra.get("partial", False)),
        coverage=extra.get("coverage"),
        namespace=resp.namespace,
    )


def recall(client: Client, query: str, **fields: Any) -> RecallResult:
    """``client.memory.recall``, with completeness carried alongside."""
    return _to_result(client.memory.recall(query, **fields))


async def arecall(client: AsyncClient, query: str, **fields: Any) -> RecallResult:
    """:func:`recall`, awaited."""
    return _to_result(await client.memory.recall(query, **fields))


def search_hits(client: Client, index: str, query: Query, **kw: Any) -> list[Hit]:
    """``client.search`` hits, with a missing index read as no hits."""
    try:
        hits: list[Hit] = client.search(index, query, **kw).hits
        return hits
    except NotFound:
        return []


async def asearch_hits(client: AsyncClient, index: str, query: Query, **kw: Any) -> list[Hit]:
    try:
        hits: list[Hit] = (await client.search(index, query, **kw)).hits
        return hits
    except NotFound:
        return []


def delete_doc(client: Client, index: str, doc_id: str) -> None:
    try:
        client.delete_document(index, doc_id)
    except NotFound:
        pass


async def adelete_doc(client: AsyncClient, index: str, doc_id: str) -> None:
    try:
        await client.delete_document(index, doc_id)
    except NotFound:
        pass


# ─── Encoding ───────────────────────────────────────────────────────────────


def b64e(text: str) -> str:
    """URL-safe base64 without padding: no ``,`` and no ``=``, so it survives the
    node's flattening of tags into a ``k=v,k=v`` string."""
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def b64d(text: str) -> str:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode()


def parse_tags(tags: str | None) -> dict[str, str]:
    """The node returns tags flattened to ``k=v,k=v``; turn them back into a map.

    Only values this package wrote are read back, and those are base64url or
    digits, so splitting on ``,`` and the first ``=`` is exact for them.
    """
    out: dict[str, str] = {}
    if not tags:
        return out
    for part in tags.split(","):
        k, sep, v = part.partition("=")
        if sep:
            out[k] = v
    return out


def parse_created(value: Any) -> datetime:
    """A memory's ``created`` as an aware datetime, whatever shape it came in."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    if isinstance(value, str) and value:
        # Python 3.10's fromisoformat does not accept a trailing "Z".
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def wildcard_prefix(field_name: str, prefix: str) -> tuple[Query, Callable[[str], bool]]:
    """A query for values starting with ``prefix``, and an exact check to apply
    to what it returns.

    The node's ``prefix`` query does not match keyword fields (it returns
    nothing), and its ``wildcard`` query has no escape for ``*`` and ``?``. So
    the pattern is built from the literal part of the prefix up to the first
    metacharacter — which can only widen the match — and the caller filters the
    hits down with the returned predicate.
    """
    cut = len(prefix)
    for i, ch in enumerate(prefix):
        if ch in "*?\\":
            cut = i
            break
    pattern = prefix[:cut] + "*"
    return q.wildcard(field_name, pattern), lambda v: v.startswith(prefix)


# ─── Read-your-deletes ──────────────────────────────────────────────────────


@dataclass
class Tombstones:
    """Ids this process deleted in the last few seconds.

    A write is searchable as soon as it is acknowledged, but a delete becomes
    invisible to search only after the node's next refresh (about a second).
    Without this, a ``put`` that replaces a value, or a ``clear``, could be
    followed by a read that still returns what was just removed.

    Scope is this process only; another process reading the same space sees the
    delete when the node refreshes.
    """

    ttl_s: float = 10.0
    _ids: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, doc_id: str) -> None:
        with self._lock:
            self._ids[doc_id] = time.monotonic() + self.ttl_s

    def __contains__(self, doc_id: object) -> bool:
        with self._lock:
            now = time.monotonic()
            for k in [k for k, exp in self._ids.items() if exp < now]:
                del self._ids[k]
            return doc_id in self._ids


def term_all(fields: Mapping[str, str | None]) -> Query:
    """A filter-only bool query: every non-None field must match exactly."""
    clauses = [q.term(k, v) for k, v in fields.items() if v is not None]
    return q.bool_().filter(*clauses).build()
