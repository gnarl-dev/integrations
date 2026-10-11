"""Recall through the published gnarl-client, with completeness kept."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pydantic
from gnarl import AsyncClient, Client
from gnarl import RecallResult as _GnarlRecall

__all__ = ["IncompleteRecall", "RecallResult", "arecall", "recall"]


@dataclass
class RecallResult:
    """``memory.recall``'s answer with its completeness alongside.

    ``partial`` is True when part of the memory space did not answer: the
    memories are a subset, and one that is missing may still exist.
    """

    memories: list[Any]
    partial: bool = False
    coverage: dict[str, Any] | None = None
    namespace: str | None = None


class IncompleteRecall(Exception):
    """A recall came back partial and the caller asked for completeness.
    The partial result is on ``result``."""

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
    """``client.memory.recall``, reading a never-written namespace as empty.

    The node answers a recall over a namespace with no index yet with a body
    that omits ``embedder``, which gnarl-client 0.2.0 requires, so the client
    raises a validation error on what is really "no memories yet". Only that
    case is absorbed; any other validation error is re-raised.
    """
    try:
        return _to_result(client.memory.recall(query, **fields))
    except pydantic.ValidationError:
        ns = fields.get("namespace")
        if ns and not client.index_exists(ns):
            return RecallResult(memories=[], namespace=ns)
        raise


async def arecall(client: AsyncClient, query: str, **fields: Any) -> RecallResult:
    """:func:`recall`, awaited."""
    try:
        return _to_result(await client.memory.recall(query, **fields))
    except pydantic.ValidationError:
        ns = fields.get("namespace")
        if ns and not await client.index_exists(ns):
            return RecallResult(memories=[], namespace=ns)
        raise
