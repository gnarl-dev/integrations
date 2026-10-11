"""Chat history for LangChain (``BaseChatMessageHistory``) stored as Gnarl memories.

    from langchain_gnarl import GnarlChatMessageHistory

    history = GnarlChatMessageHistory("thread-42", user="alice")
    history.add_user_message("I moved to Lisbon last spring")
    history.messages            # every turn of thread-42, oldest first

Each message is one memory with ``session`` = the session id, so turns are
also reachable by semantic recall — a :class:`~langchain_gnarl.GnarlRetriever`
over the same namespace and user finds them across sessions.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

from gnarl import AsyncClient, Client, Hit, NotFound
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from ._gnarl import Clients, Tombstones, b64d, b64e, check_namespace, parse_tags, term_all

__all__ = ["GnarlChatMessageHistory"]

#: ``fact_type`` stamped on every turn, so a history never reads memories it did
#: not write.
FACT_TYPE = "langchain.message"
_TAG_MESSAGE = "lc_m"
_TAG_SEQ = "lc_s"
_SORT = ["created"]


def _text_of(message: BaseMessage) -> str:
    """What the node embeds for a turn: its text, or a marker for a turn that
    has none (a tool call, an image) — the node refuses empty content."""
    text = message.text if isinstance(message.text, str) else str(message.text)
    return text if text.strip() else f"[{message.type} message]"


class GnarlChatMessageHistory(BaseChatMessageHistory):
    """The turns of one conversation, kept as Gnarl memories.

    :param session_id: the conversation. Stored as each memory's ``session``.
    :param client: a :class:`gnarl.Client`; when omitted, one that finds the
        node itself.
    :param async_client: a :class:`gnarl.AsyncClient` for the ``a*`` methods.
        Without one, but with ``client``, they run the sync client in a thread.
    :param namespace: the Gnarl memory namespace (default ``langchain-chat``).
    :param user: the Gnarl ``user`` the turns belong to. Two histories with the
        same ``session_id`` and different users never see each other's turns.

    Order is the order turns were added in this process and across processes to
    within the node clock's millisecond — each memory carries a nanosecond
    sequence tag, because the node's ``created`` sorts at one-second precision.
    """

    def __init__(
        self,
        session_id: str,
        *,
        client: Client | None = None,
        async_client: AsyncClient | None = None,
        namespace: str = "langchain-chat",
        user: str | None = None,
        page_size: int = 500,
    ) -> None:
        if not session_id:
            raise ValueError("gnarl: session_id is required")
        self.session_id = session_id
        self.namespace = check_namespace(namespace)
        self.user = user
        self.page_size = page_size
        self._clients = Clients(client, async_client)
        self._has_async = async_client is not None
        self._tombstones = Tombstones()

    # ── read ────────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        return self._decode(self._hits_sync())

    async def aget_messages(self) -> list[BaseMessage]:
        if not self._has_async and self._clients._sync is not None:
            return await asyncio.to_thread(lambda: self.messages)
        return self._decode(await self._hits_async())

    def _query(self) -> Any:
        return term_all({"fact_type": FACT_TYPE, "session": self.session_id, "user": self.user})

    def _hits_sync(self) -> list[Hit]:
        try:
            return list(
                self._clients.sync.iter_search(
                    self.namespace, self._query(), sort=_SORT, page_size=self.page_size
                )
            )
        except NotFound:
            return []

    async def _hits_async(self) -> list[Hit]:
        out: list[Hit] = []
        try:
            async for h in self._clients.aio.iter_search(
                self.namespace, self._query(), sort=_SORT, page_size=self.page_size
            ):
                out.append(h)
        except NotFound:
            return []
        return out

    def _decode(self, hits: list[Hit]) -> list[BaseMessage]:
        rows: list[tuple[int, dict[str, Any]]] = []
        for h in hits:
            if h.field_id in self._tombstones:
                continue
            tags = parse_tags((h.field_source or {}).get("tags"))
            if _TAG_MESSAGE not in tags:
                continue
            seq = int(tags[_TAG_SEQ]) if tags.get(_TAG_SEQ, "").isdigit() else 0
            rows.append((seq, json.loads(b64d(tags[_TAG_MESSAGE]))))
        rows.sort(key=lambda r: r[0])
        return messages_from_dict([d for _, d in rows])

    # ── write ───────────────────────────────────────────────────────────────

    def _remember_args(self, messages: Sequence[BaseMessage]) -> list[tuple[str, dict[str, Any]]]:
        base = time.time_ns()
        out = []
        for i, msg in enumerate(messages):
            encoded = json.dumps(message_to_dict(msg), separators=(",", ":"))
            out.append(
                (
                    _text_of(msg),
                    dict(
                        namespace=self.namespace,
                        user=self.user,
                        session=self.session_id,
                        fact_type=FACT_TYPE,
                        agent="langchain",
                        tags={
                            _TAG_MESSAGE: b64e(encoded),
                            _TAG_SEQ: str(base + i),
                            "role": msg.type,
                        },
                    ),
                )
            )
        return out

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # One remember per turn, in order. Not batched: the memory API stores
        # one memory per call, and each turn is embedded on its own.
        for content, kw in self._remember_args(messages):
            self._clients.sync.memory.remember(content, **kw)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not self._has_async and self._clients._sync is not None:
            await asyncio.to_thread(self.add_messages, messages)
            return
        for content, kw in self._remember_args(messages):
            await self._clients.aio.memory.remember(content, **kw)

    def clear(self) -> None:
        c = self._clients.sync
        for h in self._hits_sync():
            try:
                c.delete_document(self.namespace, h.field_id)
            except NotFound:
                pass
            self._tombstones.add(h.field_id)

    async def aclear(self) -> None:
        if not self._has_async and self._clients._sync is not None:
            await asyncio.to_thread(self.clear)
            return
        c = self._clients.aio
        for h in await self._hits_async():
            try:
                await c.delete_document(self.namespace, h.field_id)
            except NotFound:
                pass
            self._tombstones.add(h.field_id)
