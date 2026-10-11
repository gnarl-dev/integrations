from __future__ import annotations

import pytest
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.schema import QueryBundle

from llama_index.memory.gnarl import GnarlMemoryBlock, GnarlRetriever, IncompleteRecall
from llama_index.memory.gnarl.base import FACT_TYPE, PARTIAL_NOTE

from .fake_gnarl import AsyncFakeGnarl, FakeGnarl


@pytest.fixture
def fake() -> FakeGnarl:
    return FakeGnarl()


def block(fake: FakeGnarl, **kw: object) -> GnarlMemoryBlock:
    return GnarlMemoryBlock(client=AsyncFakeGnarl(fake), namespace="mem", **kw)  # type: ignore[arg-type]


def msg(role: str, text: str) -> ChatMessage:
    return ChatMessage(role=role, content=text)


def test_is_a_memory_block(fake: FakeGnarl) -> None:
    assert isinstance(block(fake), BaseMemoryBlock)


async def test_get_on_an_empty_space_is_empty(fake: FakeGnarl) -> None:
    b = block(fake, user="alice")
    assert await b.aget([msg("user", "hello")]) == ""
    assert await b.aget([]) == ""


async def test_put_then_get_recalls_relevant_memories(fake: FakeGnarl) -> None:
    b = block(fake, user="alice")
    await b.aput(
        [
            msg("system", "be terse"),
            msg("user", "I am vegetarian and love falafel"),
            msg("assistant", "noted"),
            msg("tool", "tool output"),
        ],
        session_id="s1",
    )
    stored = [(c, kw) for kind, c, kw in fake.calls if kind == "remember"]
    assert [c for c, _ in stored] == ["I am vegetarian and love falafel", "noted"]
    assert all(kw["user"] == "alice" and kw["session"] == "s1" for _, kw in stored)
    assert all(kw["fact_type"] == FACT_TYPE and kw["namespace"] == "mem" for _, kw in stored)
    assert stored[0][1]["tags"] == {"role": "user"}

    text = await b.aget([msg("user", "what do I love to eat? falafel?")], session_id="s9")
    assert text.splitlines()[0] == "- I am vegetarian and love falafel"
    (_, query, kw) = [c for c in fake.calls if c[0] == "recall"][-1]
    assert query == "what do I love to eat? falafel?"
    assert kw == {"namespace": "mem", "user": "alice", "k": 5}  # all sessions


async def test_users_are_isolated(fake: FakeGnarl) -> None:
    await block(fake, user="alice").aremember("alice lives in lisbon")
    await block(fake, user="bob").aremember("bob lives in oslo")
    text = await block(fake, user="bob").aget([msg("user", "where does alice live? lisbon?")])
    assert text == "- bob lives in oslo"


async def test_session_scope_window_and_min_score(fake: FakeGnarl) -> None:
    b = block(fake, user="a", session="s1", retrieval_context_window=2, min_score=0.5)
    await b.aremember("cats are great", session="s1")
    await b.aremember("cats in another session", session="s2")
    window = [msg("user", "unrelated"), msg("assistant", "x"), msg("user", "cats great")]
    text = await b.aget(window)
    (_, query, kw) = [c for c in fake.calls if c[0] == "recall"][-1]
    assert query == "x cats great" and kw["session"] == "s1"
    assert text == "- cats are great"


async def test_partial_recall_is_announced_or_refused() -> None:
    fake = FakeGnarl(partial=True)
    b = block(fake, user="a")
    await b.aremember("cats are great")
    text = await b.aget([msg("user", "cats")])
    assert text.endswith(PARTIAL_NOTE) and "- cats are great" in text
    with pytest.raises(IncompleteRecall):
        await block(fake, user="a", require_complete=True).aget([msg("user", "cats")])


async def test_flushed_short_term_messages_reach_gnarl(fake: FakeGnarl) -> None:
    b = block(fake, user="alice")
    memory = Memory.from_defaults(
        session_id="thread-1", token_limit=40, chat_history_token_ratio=0.5,
        token_flush_size=10, memory_blocks=[b],
    )
    await memory.aput_messages(
        [msg("user", f"fact number {i} about falafel") for i in range(6)]
    )
    stored = [kw for kind, _, kw in fake.calls if kind == "remember"]
    assert stored and all(kw["session"] == "thread-1" for kw in stored)
    out = await memory.aget(input="falafel")
    assert "<GnarlMemory>" in str(out[0].content)


def test_retriever_sync(fake: FakeGnarl) -> None:
    fake.memory.remember("alice lives in lisbon", namespace="mem", user="alice")
    fake.memory.remember("bob lives in oslo", namespace="mem", user="bob")
    r = GnarlRetriever(fake, namespace="mem", user="alice", k=3)
    nodes = r.retrieve("where does alice live")
    assert [n.node.get_content() for n in nodes] == ["alice lives in lisbon"]
    assert nodes[0].score == nodes[0].node.metadata["score"]
    assert nodes[0].node.metadata["partial"] is False
    assert r.retrieve(QueryBundle("x")) is not None
    assert GnarlRetriever(fake, namespace="nothing-here").retrieve("x") == []


async def test_retriever_async_and_partial() -> None:
    fake = FakeGnarl(partial=True)
    fake.memory.remember("bob lives in oslo", namespace="mem", user="bob")
    r = GnarlRetriever(async_client=AsyncFakeGnarl(fake), namespace="mem", user="bob")
    nodes = await r.aretrieve("oslo")
    assert nodes[0].node.metadata["partial"] is True
    strict = GnarlRetriever(fake, namespace="mem", require_complete=True)
    with pytest.raises(IncompleteRecall):
        strict.retrieve("oslo")
