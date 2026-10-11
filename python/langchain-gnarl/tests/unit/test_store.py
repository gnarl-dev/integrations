from __future__ import annotations

import time

import pytest
from langgraph.store.base import BaseStore

from langchain_gnarl import GnarlStore
from langchain_gnarl._gnarl import wildcard_prefix
from langchain_gnarl.store import FACT_TYPE, _embed_text

from .fake_gnarl import AsyncFakeGnarl, FakeGnarl

NS = "teststore"


@pytest.fixture
def fake() -> FakeGnarl:
    return FakeGnarl()


@pytest.fixture
def store(fake: FakeGnarl) -> GnarlStore:
    return GnarlStore(fake, namespace=NS)


def test_is_a_langgraph_store(store: GnarlStore) -> None:
    assert isinstance(store, BaseStore)
    assert store.supports_ttl is False


def test_put_get_round_trips_the_value(store: GnarlStore, fake: FakeGnarl) -> None:
    value = {"text": "Alice is vegetarian", "n": 3, "nested": {"ok": True}}
    store.put(("memories", "alice"), "diet", value)

    item = store.get(("memories", "alice"), "diet")
    assert item is not None
    assert item.value == value
    assert item.key == "diet"
    assert item.namespace == ("memories", "alice")

    (_, content, kw) = next(c for c in fake.calls if c[0] == "remember")
    assert content == "Alice is vegetarian"  # strings embedded, not the JSON
    assert kw["namespace"] == NS
    assert kw["user"] == "memories.alice"
    assert kw["session"] == "diet"
    assert kw["fact_type"] == FACT_TYPE


def test_get_missing_is_none(store: GnarlStore, fake: FakeGnarl) -> None:
    assert store.get(("memories", "alice"), "nope") is None  # no index yet
    store.put(("memories", "alice"), "a", {"text": "x"})
    assert store.get(("memories", "alice"), "nope") is None


def test_put_replaces_and_keeps_created_at(store: GnarlStore) -> None:
    store.put(("m", "a"), "k", {"text": "v1"})
    first = store.get(("m", "a"), "k")
    store.put(("m", "a"), "k", {"text": "v2"})
    second = store.get(("m", "a"), "k")
    assert first is not None and second is not None
    assert second.value == {"text": "v2"}
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert [i.value for i in store.search(("m", "a"))] == [{"text": "v2"}]


def test_replace_is_read_your_write_while_delete_is_still_visible_on_the_node() -> None:
    fake = FakeGnarl(delete_lag=True)
    store = GnarlStore(fake, namespace=NS)
    store.put(("m", "a"), "k", {"text": "old value"})
    store.put(("m", "a"), "k", {"text": "new value"})
    store.put(("m", "a"), "gone", {"text": "old value"})
    store.delete(("m", "a"), "gone")

    item = store.get(("m", "a"), "k")
    assert item is not None and item.value == {"text": "new value"}
    assert store.get(("m", "a"), "gone") is None
    assert [i.key for i in store.search(("m", "a"), query="value")] == ["k"]
    assert [i.key for i in store.search(("m", "a"))] == ["k"]


def test_delete(store: GnarlStore) -> None:
    store.put(("m", "a"), "k", {"text": "v"})
    store.delete(("m", "a"), "k")
    assert store.get(("m", "a"), "k") is None
    store.delete(("m", "a"), "never-existed")  # no error


def test_semantic_search_is_scoped_to_the_namespace(store: GnarlStore, fake: FakeGnarl) -> None:
    store.put(("memories", "alice"), "diet", {"text": "alice eats falafel"})
    store.put(("memories", "alice"), "home", {"text": "alice lives in lisbon"})
    store.put(("memories", "bob"), "diet", {"text": "bob eats steak"})

    hits = store.search(("memories", "alice"), query="what does alice eat", limit=5)
    assert {h.key for h in hits} == {"diet", "home"}
    assert all(h.namespace == ("memories", "alice") for h in hits)
    assert hits[0].key == "diet" and hits[0].score is not None
    assert hits == sorted(hits, key=lambda h: h.score or 0, reverse=True)

    recalls = [c for c in fake.calls if c[0] == "recall"]
    # Scoping is the NODE's: every recall names the user.
    assert all(kw["user"] == "memories.alice" for _, _, kw in recalls)

    bob = store.search(("memories", "bob"), query="falafel lisbon")
    assert [h.value["text"] for h in bob] == ["bob eats steak"]


def test_search_over_a_prefix_recalls_each_sub_namespace(store: GnarlStore) -> None:
    store.put(("memories", "alice"), "pet", {"text": "alice has a cat"})
    store.put(("memories", "bob"), "pet", {"text": "bob has a dog"})
    store.put(("other", "carol"), "pet", {"text": "carol has a cat"})
    hits = store.search(("memories",), query="cat", limit=10)
    assert {h.namespace for h in hits} == {("memories", "alice"), ("memories", "bob")}
    assert hits[0].namespace == ("memories", "alice")
    everything = store.search((), query="cat", limit=10)
    assert len(everything) == 3


def test_search_filter_offset_limit(store: GnarlStore) -> None:
    for i in range(6):
        kind = "even" if i % 2 == 0 else "odd"
        store.put(("m",), f"k{i}", {"text": f"note {i}", "n": i, "kind": kind})
    evens = store.search(("m",), query="note", filter={"kind": "even"}, limit=10)
    assert sorted(h.key for h in evens) == ["k0", "k2", "k4"]
    gt = store.search(("m",), filter={"n": {"$gte": 4}})
    assert sorted(h.key for h in gt) == ["k4", "k5"]
    ne = store.search(("m",), filter={"kind": {"$ne": "odd"}}, limit=2)
    assert len(ne) == 2
    page1 = store.search(("m",), limit=3)
    page2 = store.search(("m",), limit=3, offset=3)
    assert len(page1) == 3 and len(page2) == 3
    assert not {h.key for h in page1} & {h.key for h in page2}


def test_listing_pages_through_everything(fake: FakeGnarl) -> None:
    store = GnarlStore(fake, namespace=NS, page_size=2)
    for i in range(5):
        store.put(("m", "x"), f"k{i}", {"text": f"t{i}"})
    assert len(store.search(("m",), limit=100)) == 5


def test_list_namespaces(store: GnarlStore) -> None:
    store.put(("memories", "alice", "prefs"), "a", {"text": "x"})
    store.put(("memories", "bob"), "a", {"text": "x"})
    store.put(("docs", "public"), "a", {"text": "x"})
    assert store.list_namespaces() == [
        ("docs", "public"), ("memories", "alice", "prefs"), ("memories", "bob"),
    ]
    assert store.list_namespaces(prefix=("memories",)) == [
        ("memories", "alice", "prefs"), ("memories", "bob"),
    ]
    assert store.list_namespaces(suffix=("public",)) == [("docs", "public")]
    assert store.list_namespaces(prefix=("memories", "*")) == [
        ("memories", "alice", "prefs"), ("memories", "bob"),
    ]
    assert store.list_namespaces(max_depth=1) == [("docs",), ("memories",)]
    assert store.list_namespaces(limit=1, offset=1) == [("memories", "alice", "prefs")]


def test_ignores_memories_it_did_not_write(store: GnarlStore, fake: FakeGnarl) -> None:
    store.put(("m", "a"), "k", {"text": "a cat"})
    fake.memory.remember("a cat too", namespace=NS, user="m.a", session="k")
    assert [h.value for h in store.search(("m", "a"), query="cat")] == [{"text": "a cat"}]
    item = store.get(("m", "a"), "k")
    assert item is not None and item.value == {"text": "a cat"}


def test_rejects_what_it_cannot_store_faithfully(fake: FakeGnarl) -> None:
    store = GnarlStore(fake, namespace=NS)
    with pytest.raises(ValueError):
        store.put(("has.dot",), "k", {"text": "x"})
    with pytest.raises(ValueError):
        GnarlStore(fake, namespace="Not Valid")
    with pytest.raises(NotImplementedError):
        store.put(("m",), "k", {"text": "x"}, ttl=5)


def test_index_fields(fake: FakeGnarl) -> None:
    store = GnarlStore(fake, namespace=NS, index_fields=["title"])
    store.put(("m",), "k", {"title": "the title", "body": "ignored"})
    store.put(("m",), "k2", {"title": "t", "body": "the body"}, index=["body"])
    contents = [c for (kind, c, _) in fake.calls if kind == "remember"]
    assert contents == ["the title", "the body"]


def test_embed_text_falls_back_to_json() -> None:
    assert _embed_text({"n": 1}, None) == '{"n": 1}'
    assert _embed_text({"a": "x", "b": ["y", {"c": "z"}]}, None) == "x\ny\nz"
    assert _embed_text({"a": {"b": 1}}, ["$"]) == '{"a": {"b": 1}}'


def test_wildcard_prefix_widens_then_filters() -> None:
    query, keep = wildcard_prefix("user", "a*b.")
    assert query.model_dump(exclude_none=True) == {"wildcard": {"user": "a*"}}
    assert keep("a*b.c") and not keep("axb.c")


async def test_async_with_an_async_client(fake: FakeGnarl) -> None:
    store = GnarlStore(async_client=AsyncFakeGnarl(fake), namespace=NS)
    await store.aput(("m", "a"), "k", {"text": "async tea"})
    item = await store.aget(("m", "a"), "k")
    assert item is not None and item.value == {"text": "async tea"}
    hits = await store.asearch(("m", "a"), query="tea")
    assert [h.key for h in hits] == ["k"]
    assert await store.alist_namespaces() == [("m", "a")]
    await store.adelete(("m", "a"), "k")
    assert await store.aget(("m", "a"), "k") is None


async def test_async_without_an_async_client_uses_the_sync_one(store: GnarlStore) -> None:
    await store.aput(("m",), "k", {"text": "v"})
    item = await store.aget(("m",), "k")
    assert item is not None and item.value == {"text": "v"}


def test_another_process_reads_the_newest_copy_before_the_node_refreshes() -> None:
    # Two stores = two processes: B's delete of A's copy is not in A's
    # tombstones, so until the node refreshes A sees both copies.
    fake = FakeGnarl(delete_lag=True)
    a = GnarlStore(fake, namespace=NS)
    b = GnarlStore(fake, namespace=NS)
    a.put(("m",), "k", {"text": "from a"})
    time.sleep(0.005)
    b.put(("m",), "k", {"text": "from b"})
    item = a.get(("m",), "k")
    assert item is not None and item.value == {"text": "from b"}
    assert [i.value for i in a.search(("m",))] == [{"text": "from b"}]
