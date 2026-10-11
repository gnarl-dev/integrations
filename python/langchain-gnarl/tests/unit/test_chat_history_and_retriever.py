from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_gnarl import GnarlChatMessageHistory, GnarlRetriever, IncompleteRecall
from langchain_gnarl.chat_history import FACT_TYPE

from .fake_gnarl import AsyncFakeGnarl, FakeGnarl

NS = "testchat"


@pytest.fixture
def fake() -> FakeGnarl:
    return FakeGnarl()


def history(
    fake: FakeGnarl, session: str = "t1", user: str | None = "alice"
) -> GnarlChatMessageHistory:
    return GnarlChatMessageHistory(session, client=fake, namespace=NS, user=user)


def test_messages_round_trip_in_order(fake: FakeGnarl) -> None:
    h = history(fake)
    assert h.messages == []  # no index yet
    msgs = [
        SystemMessage("be brief"),
        HumanMessage("I moved to Lisbon"),
        AIMessage("", tool_calls=[{"name": "lookup", "args": {"q": "lisbon"}, "id": "c1"}]),
        ToolMessage("sunny", tool_call_id="c1"),
        AIMessage("Lisbon is sunny"),
    ]
    h.add_messages(msgs)
    got = h.messages
    assert [type(m) for m in got] == [type(m) for m in msgs]
    assert [m.content for m in got] == [m.content for m in msgs]
    assert isinstance(got[2], AIMessage) and got[2].tool_calls[0]["name"] == "lookup"

    remembered = [(c, kw) for (kind, c, kw) in fake.calls if kind == "remember"]
    assert remembered[2][0] == "[ai message]"  # the node refuses empty content
    assert all(kw["session"] == "t1" and kw["user"] == "alice" for _, kw in remembered)
    assert all(kw["fact_type"] == FACT_TYPE for _, kw in remembered)


def test_order_does_not_depend_on_the_nodes_sort(fake: FakeGnarl) -> None:
    h = history(fake)
    h.add_messages([HumanMessage(str(i)) for i in range(5)])
    # Scramble storage order: the node's `created` sorts at 1s precision.
    docs = fake.indexes[NS]
    fake.indexes[NS] = dict(reversed(list(docs.items())))
    assert [m.content for m in h.messages] == ["0", "1", "2", "3", "4"]


def test_sessions_and_users_are_isolated(fake: FakeGnarl) -> None:
    history(fake, "t1", "alice").add_user_message("alice t1")
    history(fake, "t2", "alice").add_user_message("alice t2")
    history(fake, "t1", "bob").add_user_message("bob t1")
    assert [m.content for m in history(fake, "t1", "alice").messages] == ["alice t1"]
    assert [m.content for m in history(fake, "t1", "bob").messages] == ["bob t1"]


def test_clear_is_immediate_even_before_the_node_refreshes() -> None:
    fake = FakeGnarl(delete_lag=True)
    h = history(fake)
    h.add_user_message("forget me")
    h.clear()
    assert h.messages == []
    h.add_user_message("after")
    assert [m.content for m in h.messages] == ["after"]


def test_requires_session_id(fake: FakeGnarl) -> None:
    with pytest.raises(ValueError):
        history(fake, session="")


async def test_async_history(fake: FakeGnarl) -> None:
    h = GnarlChatMessageHistory(
        "t1", async_client=AsyncFakeGnarl(fake), namespace=NS
    )
    await h.aadd_messages([HumanMessage("hi"), AIMessage("hello")])
    assert [m.content for m in await h.aget_messages()] == ["hi", "hello"]
    await h.aclear()
    assert await h.aget_messages() == []


async def test_async_history_on_a_sync_client(fake: FakeGnarl) -> None:
    h = history(fake)
    await h.aadd_messages([HumanMessage("hi")])
    assert [m.content for m in await h.aget_messages()] == ["hi"]


# ─── Retriever ──────────────────────────────────────────────────────────────


def seed(fake: FakeGnarl) -> None:
    fake.memory.remember("alice lives in lisbon", namespace="mem", user="alice", session="s1")
    fake.memory.remember("alice has a cat", namespace="mem", user="alice", session="s2")
    fake.memory.remember("bob lives in oslo", namespace="mem", user="bob")


def test_retriever_returns_documents_with_metadata(fake: FakeGnarl) -> None:
    seed(fake)
    r = GnarlRetriever(client=fake, namespace="mem", user="alice", k=2)
    docs = r.invoke("where does alice live")
    assert docs[0].page_content == "alice lives in lisbon"
    md = docs[0].metadata
    assert md["score"] > 0 and md["id"] == docs[0].id
    assert md["namespace"] == "mem" and md["partial"] is False
    assert md["coverage"]["served_claims"] == 4
    assert all("bob" not in d.page_content for d in docs)
    (_, query, kw) = [c for c in fake.calls if c[0] == "recall"][-1]
    assert kw == {"namespace": "mem", "user": "alice", "k": 2}


def test_retriever_session_scope(fake: FakeGnarl) -> None:
    seed(fake)
    r = GnarlRetriever(client=fake, namespace="mem", user="alice", session="s2")
    assert [d.page_content for d in r.invoke("alice")] == ["alice has a cat"]


def test_retriever_on_an_empty_namespace(fake: FakeGnarl) -> None:
    r = GnarlRetriever(client=fake, namespace="never-written")
    assert r.invoke("anything") == []


def test_retriever_reports_partial_and_can_refuse_it() -> None:
    fake = FakeGnarl(partial=True)
    seed(fake)
    docs = GnarlRetriever(client=fake, namespace="mem", user="alice").invoke("alice")
    assert docs and all(d.metadata["partial"] is True for d in docs)
    strict = GnarlRetriever(client=fake, namespace="mem", require_complete=True)
    with pytest.raises(IncompleteRecall) as err:
        strict.invoke("alice")
    assert err.value.result.partial


async def test_retriever_async(fake: FakeGnarl) -> None:
    seed(fake)
    r = GnarlRetriever(async_client=AsyncFakeGnarl(fake), namespace="mem", user="bob")
    docs = await r.ainvoke("where does bob live")
    assert [d.page_content for d in docs] == ["bob lives in oslo"]
