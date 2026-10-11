"""LangGraph long-term memory (``BaseStore``) backed by Gnarl.

    from langchain_gnarl import GnarlStore

    store = GnarlStore()                      # finds the node on this machine
    store.put(("memories", "alice"), "food", {"text": "Alice is vegetarian"})
    store.search(("memories", "alice"), query="what does alice eat?")

How LangGraph's model maps onto Gnarl's memory model:

====================  =====================================================
LangGraph             Gnarl memory
====================  =====================================================
the whole store       one memory ``namespace`` (default ``langgraph-store``)
item namespace tuple  ``user``, the labels joined with ``.``
item key              ``session``
item value            the ``lg_v`` tag (JSON, base64url); its text is embedded
``search(query=)``    ``memory.recall`` scoped to ``user`` (semantic)
====================  =====================================================

LangGraph forbids ``.`` inside a namespace label, so the join is reversible.
Scoping by ``user`` means the node itself keeps one namespace's items out of
another's recall; isolation does not depend on filtering in this process.

Every write goes through ``memory.remember``, so the node embeds the value
with its own model. There is no embedding configuration on this store, and an
``index=False`` put is still embedded.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from gnarl import AsyncClient, Client, Hit, Query
from gnarl import query as q
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from ._gnarl import (
    Clients,
    RecallResult,
    Tombstones,
    adelete_doc,
    arecall,
    asearch_hits,
    b64d,
    b64e,
    check_namespace,
    delete_doc,
    from_ms,
    now_ms,
    parse_created,
    parse_tags,
    recall,
    search_hits,
    term_all,
    wildcard_prefix,
)

__all__ = ["GnarlStore"]

#: ``fact_type`` stamped on every item, so this store never reads a memory it
#: did not write (a chat history or an agent's own facts in the same space).
FACT_TYPE = "langgraph.item"
_SEP = "."
_TAG_VALUE = "lg_v"
_TAG_CREATED = "lg_c"
_TAG_UPDATED = "lg_u"
_SOURCE = ["user", "session", "tags", "created", "fact_type", "content"]
_RECALL_MAX = 100

T = TypeVar("T")


# ─── The I/O an operation asks for ──────────────────────────────────────────
#
# Each operation is written ONCE, as a generator that yields what it needs from
# the node and receives the answer. A sync driver and an async driver run the
# same generator, so the sync and async halves of the store cannot drift.


@dataclass(frozen=True)
class _Remember:
    content: str
    user: str
    key: str
    tags: dict[str, str]


@dataclass(frozen=True)
class _Recall:
    query: str
    user: str
    k: int


@dataclass(frozen=True)
class _Search:
    query: Query
    size: int
    from_: int = 0


@dataclass(frozen=True)
class _Delete:
    doc_id: str


_Req = _Remember | _Recall | _Search | _Delete
_Gen = Generator[_Req, Any, T]


@dataclass
class _Row:
    """One stored item, read back from either recall or search."""

    doc_id: str
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    updated_ms: int
    score: float | None = None


def _ns_to_user(namespace: tuple[str, ...]) -> str:
    return _SEP.join(namespace)


def _user_to_ns(user: str) -> tuple[str, ...]:
    return tuple(user.split(_SEP)) if user else ()


def _embed_text(value: dict[str, Any], fields: Sequence[str] | None) -> str:
    """The text the node embeds for ``value``.

    With ``fields``, those top-level keys (``"$"`` meaning the whole value).
    Without, every string in the value, in order — a plain sentence embeds far
    better than its JSON. A value with no strings falls back to its JSON.
    """
    parts: list[str] = []
    if fields:
        for f in fields:
            v = value if f == "$" else value.get(f)
            if v is None:
                continue
            parts.append(v if isinstance(v, str) else json.dumps(v, sort_keys=True))
    else:
        _strings(value, parts)
    text = "\n".join(p for p in parts if p.strip())
    return text or json.dumps(value, sort_keys=True)


def _strings(v: Any, out: list[str]) -> None:
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, dict):
        for x in v.values():
            _strings(x, out)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _strings(x, out)


def _matches_filter(value: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    """LangGraph's filter semantics: equality, or ``$eq/$ne/$gt/$gte/$lt/$lte``."""
    if not flt:
        return True
    for k, cond in flt.items():
        actual = value.get(k)
        if isinstance(cond, dict) and cond and all(c.startswith("$") for c in cond):
            for op, expected in cond.items():
                if not _compare(op, actual, expected):
                    return False
        elif actual != cond:
            return False
    return True


def _compare(op: str, actual: Any, expected: Any) -> bool:
    if op == "$eq":
        return bool(actual == expected)
    if op == "$ne":
        return bool(actual != expected)
    if actual is None:
        return False
    try:
        if op == "$gt":
            return bool(actual > expected)
        if op == "$gte":
            return bool(actual >= expected)
        if op == "$lt":
            return bool(actual < expected)
        if op == "$lte":
            return bool(actual <= expected)
    except TypeError:
        return False
    raise ValueError(f"gnarl: unsupported filter operator {op!r}")


def _ns_matches(ns: tuple[str, ...], cond: Any) -> bool:
    path = tuple(cond.path)
    if len(ns) < len(path):
        return False
    seg = ns[: len(path)] if cond.match_type == "prefix" else ns[len(ns) - len(path) :]
    return all(p == "*" or p == s for p, s in zip(path, seg, strict=True))


class GnarlStore(BaseStore):
    """A LangGraph ``BaseStore`` whose items are Gnarl memories.

    :param client: a :class:`gnarl.Client`; when omitted, one built on first
        use finds the node (``$GNARL_URL``, else the local node's recorded
        endpoint, else ``http://127.0.0.1:43300``).
    :param async_client: a :class:`gnarl.AsyncClient` for ``abatch`` and the
        ``a*`` methods. When it is omitted but ``client`` is given, async
        calls run the sync client in a worker thread.
    :param namespace: the Gnarl memory namespace that holds every item. Must
        already be a valid index name (``[a-z0-9][a-z0-9_-]*``, at most 64).
    :param index_fields: which top-level value keys to embed; ``"$"`` is the
        whole value. Default: every string in the value. A put's own
        ``index=[...]`` overrides it for that item.
    :param page_size: rows per page when listing (search without a query, and
        ``list_namespaces``).

    Consistency: a ``put`` is visible to every reader as soon as it returns. A
    replaced or deleted item disappears from other processes' reads after the
    node's next refresh (about a second); in this process, immediately.
    ``put`` is not atomic: two concurrent puts of one key can both land, and a
    read then returns the newer one.
    """

    supports_ttl = False

    def __init__(
        self,
        client: Client | None = None,
        *,
        async_client: AsyncClient | None = None,
        namespace: str = "langgraph-store",
        index_fields: Sequence[str] | None = None,
        page_size: int = 500,
    ) -> None:
        self.namespace = check_namespace(namespace)
        self.index_fields = list(index_fields) if index_fields else None
        self.page_size = page_size
        self._clients = Clients(client, async_client)
        self._has_async = async_client is not None
        self._tombstones = Tombstones()

    # ── BaseStore ───────────────────────────────────────────────────────────

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return [self._run_sync(self._op(op)) for op in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        ops = list(ops)
        if not self._has_async and self._clients._sync is not None:
            return await asyncio.to_thread(self.batch, ops)
        return [await self._run_async(self._op(op)) for op in ops]

    # ── drivers ─────────────────────────────────────────────────────────────

    def _run_sync(self, gen: _Gen[T]) -> T:
        try:
            req = next(gen)
            while True:
                req = gen.send(self._exec_sync(req))
        except StopIteration as stop:
            return stop.value  # type: ignore[no-any-return]

    async def _run_async(self, gen: _Gen[T]) -> T:
        try:
            req = next(gen)
            while True:
                req = gen.send(await self._exec_async(req))
        except StopIteration as stop:
            return stop.value  # type: ignore[no-any-return]

    def _exec_sync(self, req: _Req) -> Any:
        c = self._clients.sync
        if isinstance(req, _Remember):
            return c.memory.remember(
                req.content, namespace=self.namespace, user=req.user, session=req.key,
                fact_type=FACT_TYPE, agent="langgraph", tags=req.tags,
            ).id
        if isinstance(req, _Recall):
            return recall(c, req.query, namespace=self.namespace, user=req.user, k=req.k)
        if isinstance(req, _Search):
            return search_hits(
                c, self.namespace, req.query, size=req.size, from_=req.from_ or None,
                sort=["created"], source=_SOURCE,
            )
        delete_doc(c, self.namespace, req.doc_id)
        return None

    async def _exec_async(self, req: _Req) -> Any:
        c = self._clients.aio
        if isinstance(req, _Remember):
            return (
                await c.memory.remember(
                    req.content, namespace=self.namespace, user=req.user, session=req.key,
                    fact_type=FACT_TYPE, agent="langgraph", tags=req.tags,
                )
            ).id
        if isinstance(req, _Recall):
            return await arecall(c, req.query, namespace=self.namespace, user=req.user, k=req.k)
        if isinstance(req, _Search):
            return await asearch_hits(
                c, self.namespace, req.query, size=req.size, from_=req.from_ or None,
                sort=["created"], source=_SOURCE,
            )
        await adelete_doc(c, self.namespace, req.doc_id)
        return None

    # ── operations, written once ────────────────────────────────────────────

    def _op(self, op: Op) -> _Gen[Result]:
        if isinstance(op, GetOp):
            return self._get(op)
        if isinstance(op, PutOp):
            return self._put(op)
        if isinstance(op, SearchOp):
            return self._search(op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(op)
        raise ValueError(f"gnarl: unknown store operation {type(op).__name__}")

    def _rows_for_key(self, namespace: tuple[str, ...], key: str) -> _Gen[list[_Row]]:
        query = term_all(
            {"fact_type": FACT_TYPE, "user": _ns_to_user(namespace), "session": key}
        )
        hits: list[Hit] = yield _Search(query, size=50)
        rows = [r for r in map(self._row_from_hit, hits) if r is not None]
        rows.sort(key=lambda r: r.updated_ms, reverse=True)
        return rows

    def _get(self, op: GetOp) -> _Gen[Item | None]:
        rows = yield from self._rows_for_key(op.namespace, op.key)
        if not rows:
            return None
        return _item(rows[0])

    def _put(self, op: PutOp) -> _Gen[None]:
        existing = yield from self._rows_for_key(op.namespace, op.key)
        if op.value is None:
            for row in existing:
                yield _Delete(row.doc_id)
                self._tombstones.add(row.doc_id)
            return None
        _validate_namespace(op.namespace)
        ts = now_ms()
        created = existing[0].created_at if existing else from_ms(ts)
        value_json = json.dumps(op.value, separators=(",", ":"), sort_keys=True)
        fields = self.index_fields if op.index is None else _fields(op.index)
        yield _Remember(
            content=_embed_text(op.value, fields),
            user=_ns_to_user(op.namespace),
            key=op.key,
            tags={
                _TAG_VALUE: b64e(value_json),
                _TAG_CREATED: str(int(created.timestamp() * 1000)),
                _TAG_UPDATED: str(ts),
            },
        )
        # Old copies go AFTER the new one is stored: a failure in between leaves
        # a duplicate, which reads resolve to the newest, rather than a loss.
        for row in existing:
            yield _Delete(row.doc_id)
            self._tombstones.add(row.doc_id)
        return None

    def _search(self, op: SearchOp) -> _Gen[list[SearchItem]]:
        prefix = op.namespace_prefix
        want = op.offset + op.limit
        if op.query:
            targets = [prefix] if prefix else []
            descendants = yield from self._descendant_namespaces(prefix)
            targets += [ns for ns in descendants if ns != prefix]
            k = min(_RECALL_MAX, want * (4 if op.filter else 1))
            rows: list[_Row] = []
            for ns in targets:
                res: RecallResult = yield _Recall(op.query, _ns_to_user(ns), k)
                for mem in res.memories:
                    row = self._row_from_memory(mem, ns)
                    if row is not None and _matches_filter(row.value, op.filter):
                        rows.append(row)
            rows = _newest_per_key(rows)
            rows.sort(key=lambda r: r.score or 0.0, reverse=True)
            return [_search_item(r) for r in rows[op.offset : want]]

        query, keep = self._prefix_query(prefix)
        out: list[_Row] = []
        start = 0
        while True:
            hits: list[Hit] = yield _Search(query, size=self.page_size, from_=start)
            for row in map(self._row_from_hit, hits):
                if row and keep(row.namespace) and _matches_filter(row.value, op.filter):
                    out.append(row)
            if len(hits) < self.page_size:
                break
            start += len(hits)
        out = _newest_per_key(out)
        out.sort(key=lambda r: r.updated_ms, reverse=True)
        return [_search_item(r) for r in out[op.offset : want]]

    def _list_namespaces(self, op: ListNamespacesOp) -> _Gen[list[tuple[str, ...]]]:
        found = yield from self._all_namespaces(())
        out: set[tuple[str, ...]] = set()
        for ns in found:
            if op.match_conditions and not all(_ns_matches(ns, c) for c in op.match_conditions):
                continue
            out.add(ns[: op.max_depth] if op.max_depth is not None else ns)
        return sorted(out)[op.offset : op.offset + op.limit]

    # ── helpers ─────────────────────────────────────────────────────────────

    def _prefix_query(self, prefix: tuple[str, ...]) -> tuple[Query, Any]:
        """Items in ``prefix`` or below it, plus an exact check on each one."""
        fact = q.term("fact_type", FACT_TYPE)
        if not prefix:
            return q.bool_().filter(fact).build(), lambda ns: True
        user = _ns_to_user(prefix)
        below, _ = wildcard_prefix("user", user + _SEP)
        either = q.bool_().should(q.term("user", user), below).build()
        return (
            q.bool_().filter(fact, either).build(),
            lambda ns: ns[: len(prefix)] == prefix,
        )

    def _descendant_namespaces(self, prefix: tuple[str, ...]) -> _Gen[list[tuple[str, ...]]]:
        """Namespaces strictly below ``prefix`` — usually none, checked cheaply."""
        fact = q.term("fact_type", FACT_TYPE)
        if prefix:
            below, _ = wildcard_prefix("user", _ns_to_user(prefix) + _SEP)
            probe = q.bool_().filter(fact, below).build()
        else:
            probe = q.bool_().filter(fact).build()
        hits: list[Hit] = yield _Search(probe, size=1)
        if not hits:
            return []
        found = yield from self._all_namespaces(prefix)
        return [ns for ns in found if len(ns) > len(prefix)]

    def _all_namespaces(self, prefix: tuple[str, ...]) -> _Gen[set[tuple[str, ...]]]:
        """Every namespace holding an item under ``prefix``. Reads one row per
        item: the node has no aggregation the memory API exposes."""
        query, keep = self._prefix_query(prefix)
        found: set[tuple[str, ...]] = set()
        start = 0
        while True:
            hits: list[Hit] = yield _Search(query, size=self.page_size, from_=start)
            for h in hits:
                if h.field_id in self._tombstones:
                    continue
                ns = _user_to_ns(str((h.field_source or {}).get("user", "")))
                if ns and keep(ns):
                    found.add(ns)
            if len(hits) < self.page_size:
                return found
            start += len(hits)

    def _row_from_hit(self, hit: Hit) -> _Row | None:
        src = hit.field_source or {}
        if src.get("fact_type") != FACT_TYPE or hit.field_id in self._tombstones:
            return None
        return _decode_row(
            hit.field_id, _user_to_ns(str(src.get("user", ""))), src.get("session"),
            src.get("tags"), src.get("created"), None,
        )

    def _row_from_memory(self, mem: Any, ns: tuple[str, ...]) -> _Row | None:
        if mem.fact_type != FACT_TYPE or mem.id in self._tombstones:
            return None
        return _decode_row(mem.id, ns, mem.session, mem.tags, mem.created, mem.score)


def _fields(index: Any) -> list[str] | None:
    if index is False:
        # Gnarl embeds every memory; there is no "store but do not index".
        return None
    return list(index) if isinstance(index, (list, tuple)) else None


def _validate_namespace(namespace: tuple[str, ...]) -> None:
    if not namespace:
        raise ValueError("gnarl: namespace cannot be empty")
    for label in namespace:
        if not isinstance(label, str) or not label:
            raise ValueError(f"gnarl: namespace labels must be non-empty strings: {namespace!r}")
        if _SEP in label:
            raise ValueError(f"gnarl: namespace labels cannot contain '.': {namespace!r}")


def _decode_row(
    doc_id: str,
    namespace: tuple[str, ...],
    key: Any,
    tags: Any,
    created: Any,
    score: float | None,
) -> _Row | None:
    t = parse_tags(tags if isinstance(tags, str) else None)
    if _TAG_VALUE not in t or not isinstance(key, str):
        return None
    value = json.loads(b64d(t[_TAG_VALUE]))
    updated_ms = int(t[_TAG_UPDATED]) if t.get(_TAG_UPDATED, "").isdigit() else 0
    updated_at = from_ms(updated_ms) if updated_ms else parse_created(created)
    created_at = (
        from_ms(int(t[_TAG_CREATED])) if t.get(_TAG_CREATED, "").isdigit() else updated_at
    )
    return _Row(doc_id, namespace, key, value, created_at, updated_at, updated_ms, score)


def _newest_per_key(rows: list[_Row]) -> list[_Row]:
    best: dict[tuple[tuple[str, ...], str], _Row] = {}
    for r in rows:
        k = (r.namespace, r.key)
        if k not in best or r.updated_ms > best[k].updated_ms:
            best[k] = r
    return list(best.values())


def _item(row: _Row) -> Item:
    return Item(
        value=row.value, key=row.key, namespace=row.namespace,
        created_at=row.created_at, updated_at=row.updated_at,
    )


def _search_item(row: _Row) -> SearchItem:
    return SearchItem(
        namespace=row.namespace, key=row.key, value=row.value,
        created_at=row.created_at, updated_at=row.updated_at, score=row.score,
    )
