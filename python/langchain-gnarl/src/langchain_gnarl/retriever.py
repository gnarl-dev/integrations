"""A LangChain retriever over Gnarl memory recall.

    from langchain_gnarl import GnarlRetriever

    retriever = GnarlRetriever(namespace="agent-memory", user="alice", k=5)
    docs = retriever.invoke("where does alice live?")
    docs[0].metadata["score"], docs[0].metadata["partial"]
"""

from __future__ import annotations

from typing import Any

from gnarl import AsyncClient, Client
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from ._gnarl import Clients, IncompleteRecall, RecallResult, arecall, recall

__all__ = ["GnarlRetriever"]


class GnarlRetriever(BaseRetriever):
    """Recall Gnarl memories as LangChain ``Document``s, best first.

    Each document's ``page_content`` is the memory's content; its metadata
    carries ``id``, ``score`` (fused), ``knn_score``, ``fact_type``, ``session``,
    ``agent``, ``created``, ``tags`` and the recall's ``namespace``, ``partial``
    and ``coverage``.

    ``partial`` is True when part of the memory space did not answer: the
    documents are then a subset, and a memory that is missing may exist. Set
    ``require_complete=True`` to raise :class:`IncompleteRecall` instead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Client | None = None
    """Sync client; when omitted, one that finds the node itself."""
    async_client: AsyncClient | None = None
    """Async client for ``ainvoke``; without one, LangChain runs the sync path
    in an executor."""
    namespace: str | None = None
    """Gnarl memory namespace. With neither this nor ``space``, the node's
    default personal space."""
    space: str | None = None
    """``personal`` (this device) or ``household`` (replicated to mesh peers)."""
    user: str | None = None
    session: str | None = None
    """Restrict recall to one session (e.g. one chat thread)."""
    k: int = 4
    """How many memories to return (the node clamps at 100)."""
    require_complete: bool = False

    _clients: Clients = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._clients = Clients(self.client, self.async_client)

    def _fields(self) -> dict[str, Any]:
        f = {
            "namespace": self.namespace, "space": self.space, "user": self.user,
            "session": self.session, "k": self.k,
        }
        return {k: v for k, v in f.items() if v is not None}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._documents(recall(self._clients.sync, query, **self._fields()))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.async_client is None and self.client is not None:
            # Only a sync client was given: do not open a second connection
            # pool from the environment behind the caller's back.
            return self._get_relevant_documents(query, run_manager=run_manager.get_sync())
        return self._documents(await arecall(self._clients.aio, query, **self._fields()))

    def _documents(self, res: RecallResult) -> list[Document]:
        if res.partial and self.require_complete:
            raise IncompleteRecall(res)
        docs = []
        for m in res.memories:
            docs.append(
                Document(
                    page_content=m.content,
                    id=m.id,
                    metadata={
                        "id": m.id,
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
            )
        return docs
