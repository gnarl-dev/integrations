"""A LlamaIndex memory block and retriever backed by Gnarl agent memory.

    from llama_index.core.memory import Memory
    from llama_index.memory.gnarl import GnarlMemoryBlock

    memory = Memory.from_defaults(
        session_id="thread-42",
        memory_blocks=[GnarlMemoryBlock(namespace="agent-memory", user="alice")],
    )
"""

from __future__ import annotations

from typing import Any

from gnarl import AsyncClient, Client
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from ._recall import IncompleteRecall, RecallResult, arecall, recall

__all__ = ["GnarlMemoryBlock", "GnarlRetriever"]

#: ``fact_type`` stamped on what the block stores.
FACT_TYPE = "llamaindex.message"

PARTIAL_NOTE = (
    "(Some memories may be missing: part of the memory space did not answer.)"
)


def _text(message: ChatMessage) -> str:
    return " ".join(b.text for b in message.blocks if isinstance(b, TextBlock)).strip()


class GnarlMemoryBlock(BaseMemoryBlock[str]):
    """Long-term memory for LlamaIndex's ``Memory``, kept in Gnarl.

    ``aget`` recalls the memories most relevant to the latest message(s) and
    returns them as text for the memory template. ``aput`` stores each message
    as a memory; LlamaIndex's ``Memory`` calls it with messages as they are
    flushed out of short-term history. Call :meth:`aremember` to store a fact
    directly.

    Recall is scoped to ``user`` on the node, and spans every session unless
    ``session`` is set: that is what makes it long-term memory. Each stored
    message records the session it came from.
    """

    name: str = Field(default="GnarlMemory", description="The name of the memory block.")
    client: AsyncClient | None = Field(
        default=None, description="Async Gnarl client; built from $GNARL_URL when omitted."
    )
    namespace: str | None = Field(default=None, description="Gnarl memory namespace.")
    space: str | None = Field(
        default=None, description="'personal' (this device) or 'household' (shared)."
    )
    user: str | None = Field(default=None, description="Whose memories these are.")
    session: str | None = Field(
        default=None, description="Restrict recall to one session; default: all sessions."
    )
    k: int = Field(default=5, description="How many memories to recall (the node caps at 100).")
    min_score: float | None = Field(
        default=None, description="Drop recalled memories scoring below this."
    )
    retrieval_context_window: int = Field(
        default=1, description="How many of the latest messages form the recall query."
    )
    store_roles: list[MessageRole] = Field(
        default_factory=lambda: [MessageRole.USER, MessageRole.ASSISTANT],
        description="Roles whose messages aput stores.",
    )
    require_complete: bool = Field(
        default=False, description="Raise IncompleteRecall instead of returning a subset."
    )

    _client: AsyncClient | None = PrivateAttr(default=None)

    def _c(self) -> AsyncClient:
        if self.client is not None:
            return self.client
        if self._client is None:
            self._client = AsyncClient()
        return self._client

    def _scope(self) -> dict[str, Any]:
        f = {"namespace": self.namespace, "space": self.space, "user": self.user}
        return {k: v for k, v in f.items() if v is not None}

    async def _aget(
        self, messages: list[ChatMessage] | None = None, **block_kwargs: Any
    ) -> str:
        if not messages:
            return ""
        window = messages[-max(1, self.retrieval_context_window) :]
        query = " ".join(t for t in map(_text, window) if t)
        if not query:
            return ""
        fields = self._scope()
        if self.session is not None:
            fields["session"] = self.session
        res = await arecall(self._c(), query, k=self.k, **fields)
        if res.partial and self.require_complete:
            raise IncompleteRecall(res)
        lines = [
            f"- {m.content}"
            for m in res.memories
            if self.min_score is None or m.score >= self.min_score
        ]
        if not lines:
            return PARTIAL_NOTE if res.partial else ""
        if res.partial:
            lines.append(PARTIAL_NOTE)
        return "\n".join(lines)

    async def _aput(self, messages: list[ChatMessage]) -> None:
        for message in messages:
            if message.role not in self.store_roles:
                continue
            text = _text(message)
            if not text:
                continue
            session = message.additional_kwargs.get("session_id") or self.session
            await self.aremember(
                text, session=session, tags={"role": str(message.role.value)}
            )

    async def aremember(
        self,
        content: str,
        *,
        session: str | None = None,
        fact_type: str | None = FACT_TYPE,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Store one memory in this block's scope; returns its id."""
        res = await self._c().memory.remember(
            content, session=session, fact_type=fact_type, agent="llamaindex", tags=tags,
            **self._scope(),
        )
        return str(res.id)


class GnarlRetriever(BaseRetriever):
    """Recall Gnarl memories as LlamaIndex nodes, best first.

    Each node's metadata carries ``score``, ``knn_score``, ``fact_type``,
    ``session``, ``agent``, ``created``, ``tags`` and the recall's
    ``namespace``, ``partial`` and ``coverage``.
    """

    def __init__(
        self,
        client: Client | None = None,
        *,
        async_client: AsyncClient | None = None,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        k: int = 4,
        require_complete: bool = False,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self._client = client
        self._async_client = async_client
        self._fields = {
            k_: v
            for k_, v in {
                "namespace": namespace, "space": space, "user": user, "session": session, "k": k,
            }.items()
            if v is not None
        }
        self._require_complete = require_complete
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._client is None:
            self._client = Client()
        return self._nodes(recall(self._client, query_bundle.query_str, **self._fields))

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._async_client is None:
            if self._client is not None:
                return self._retrieve(query_bundle)
            self._async_client = AsyncClient()
        return self._nodes(
            await arecall(self._async_client, query_bundle.query_str, **self._fields)
        )

    def _nodes(self, res: RecallResult) -> list[NodeWithScore]:
        if res.partial and self._require_complete:
            raise IncompleteRecall(res)
        out = []
        for m in res.memories:
            node = TextNode(
                id_=m.id,
                text=m.content,
                metadata={
                    "score": m.score,
                    "knn_score": m.knn_score,
                    "fact_type": m.fact_type,
                    "session": m.session,
                    "agent": m.agent,
                    "created": m.created.isoformat() if m.created else None,
                    "tags": m.tags,
                    "namespace": res.namespace,
                    "partial": res.partial,
                    "coverage": res.coverage,
                },
            )
            out.append(NodeWithScore(node=node, score=m.score))
        return out
