"""The adapters against a real node: real embeddings, real recall, real isolation.

Each test stores three memories for user A through the framework's own API,
finds them again by a semantic query that shares no keywords with the answer,
and checks user B sees none of them.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

import pytest
from gnarl import AsyncClient, Client
from langchain_core.messages import AIMessage, HumanMessage

from langchain_gnarl import GnarlChatMessageHistory, GnarlRetriever, GnarlStore

pytestmark = pytest.mark.integration

ALICE = [
    ("diet", {"text": "Alice is vegetarian and her favourite dish is falafel"}),
    ("home", {"text": "Alice lives in Lisbon, near the river"}),
    ("pet", {"text": "Alice has a grey cat called Miso"}),
]


def test_store_semantic_recall_and_isolation(client: Client, run_id: str) -> None:
    store = GnarlStore(client, namespace=f"it-lg-{run_id}")
    alice, bob = ("memories", f"alice-{run_id}"), ("memories", f"bob-{run_id}")
    for key, value in ALICE:
        store.put(alice, key, value)
    store.put(bob, "diet", {"text": "Bob eats steak every day"})

    hit = store.search(alice, query="what kind of food does she eat?", limit=1)
    assert [h.key for h in hit] == ["diet"]
    assert store.search(alice, query="which city is her house in?", limit=1)[0].key == "home"
    assert store.search(alice, query="does she own an animal?", limit=1)[0].key == "pet"
    assert {h.key for h in store.search(alice, query="alice", limit=10)} == {"diet", "home", "pet"}

    bob_hits = store.search(bob, query="falafel lisbon cat", limit=10)
    assert [h.value["text"] for h in bob_hits] == ["Bob eats steak every day"]

    item = store.get(alice, "home")
    assert item is not None and item.value == ALICE[1][1]
    store.put(alice, "home", {"text": "Alice moved to Porto"})
    item = store.get(alice, "home")
    assert item is not None and item.value == {"text": "Alice moved to Porto"}
    store.delete(alice, "pet")
    assert store.get(alice, "pet") is None

    assert set(store.list_namespaces(prefix=("memories",))) >= {alice, bob}
    # Another process sees the replacement once the node has refreshed.
    other = GnarlStore(Client(), namespace=f"it-lg-{run_id}")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        keys = sorted(i.key for i in other.search(alice, limit=10))
        if keys == ["diet", "home"]:
            break
        time.sleep(0.25)
    assert keys == ["diet", "home"]


async def test_store_async(run_id: str, client: Client) -> None:
    async with AsyncClient() as ac:
        store = GnarlStore(async_client=ac, namespace=f"it-lg-{run_id}")
        ns = ("memories", f"alice-{run_id}")
        for key, value in ALICE:
            await store.aput(ns, key, value)
        hits = await store.asearch(ns, query="what does she like to eat", limit=1)
        assert [h.key for h in hits] == ["diet"]
        assert await store.asearch(("memories", f"bob-{run_id}"), query="food") == []


def test_store_inside_a_langgraph_graph(client: Client, run_id: str) -> None:
    from langgraph.graph import END, START, StateGraph
    from langgraph.store.base import BaseStore

    class State(TypedDict):
        user: str
        question: str
        answer: str

    def remember(state: State, *, store: BaseStore) -> dict[str, Any]:
        ns = ("memories", state["user"])
        for key, value in ALICE:
            store.put(ns, key, value)
        return {}

    def recall(state: State, *, store: BaseStore) -> dict[str, Any]:
        hits = store.search(("memories", state["user"]), query=state["question"], limit=1)
        return {"answer": hits[0].value["text"] if hits else ""}

    graph = (
        StateGraph(State)
        .add_node("remember", remember)
        .add_node("recall", recall)
        .add_edge(START, "remember")
        .add_edge("remember", "recall")
        .add_edge("recall", END)
        .compile(store=GnarlStore(client, namespace=f"it-lg-{run_id}"))
    )
    out = graph.invoke({"user": f"alice-{run_id}", "question": "where is her home?", "answer": ""})
    assert "Lisbon" in out["answer"]


def test_chat_history_and_retriever(client: Client, run_id: str) -> None:
    ns = f"it-lc-{run_id}"
    alice = GnarlChatMessageHistory("thread-1", client=client, namespace=ns, user="alice")
    alice.add_messages(
        [
            HumanMessage("I just adopted a grey cat called Miso"),
            AIMessage("Congratulations! How is Miso settling in?"),
            HumanMessage("She sleeps on my keyboard all day"),
        ]
    )
    assert [m.content for m in alice.messages] == [
        "I just adopted a grey cat called Miso",
        "Congratulations! How is Miso settling in?",
        "She sleeps on my keyboard all day",
    ]
    bob = GnarlChatMessageHistory("thread-1", client=client, namespace=ns, user="bob")
    assert bob.messages == []
    bob.add_user_message("I play the cello")

    docs = GnarlRetriever(client=client, namespace=ns, user="alice", k=1).invoke(
        "does alice have any pets?"
    )
    assert docs[0].page_content == "I just adopted a grey cat called Miso"
    assert docs[0].metadata["partial"] is False
    assert docs[0].metadata["coverage"]["served_claims"] >= 1

    bob_docs = GnarlRetriever(client=client, namespace=ns, user="bob", k=10).invoke("pets")
    assert [d.page_content for d in bob_docs] == ["I play the cello"]

    alice.clear()
    assert alice.messages == []
    assert [m.content for m in bob.messages] == ["I play the cello"]
