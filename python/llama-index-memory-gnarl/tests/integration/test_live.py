"""The block and retriever against a real node: real embeddings and recall.

User A's three facts go in through LlamaIndex's own ``Memory`` (flushed out of
short-term history into the block), come back by semantic queries that share
no keywords with the facts, and never reach user B.
"""

from __future__ import annotations

import pytest
from gnarl import AsyncClient, Client
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory

from llama_index.memory.gnarl import GnarlMemoryBlock, GnarlRetriever

pytestmark = pytest.mark.integration

FACTS = [
    "I am vegetarian and my favourite dish is falafel",
    "I live in Lisbon, near the river",
    "I have a grey cat called Miso",
]


async def test_memory_block_round_trip_and_isolation(client: Client, run_id: str) -> None:
    ns = f"it-li-{run_id}"
    async with AsyncClient() as ac:
        alice = GnarlMemoryBlock(client=ac, namespace=ns, user="alice")
        memory = Memory.from_defaults(
            session_id=f"thread-{run_id}",
            token_limit=60,
            chat_history_token_ratio=0.5,
            token_flush_size=20,
            memory_blocks=[alice],
        )
        # One turn at a time, as an agent adds them. Small talk follows the
        # facts so that the facts are what overflows the short-term window and
        # is flushed to the block.
        def turn(text: str, reply: str) -> list[ChatMessage]:
            return [
                ChatMessage(role="user", content=text),
                ChatMessage(role="assistant", content=reply),
            ]

        for fact in FACTS:
            await memory.aput_messages(turn(fact, "ok"))
        for _ in range(6):
            await memory.aput_messages(
                turn("thanks, that is really helpful", "you are welcome, happy to help")
            )

        # Everything flushed from short-term memory is now in Gnarl.
        everything = GnarlRetriever(async_client=ac, namespace=ns, user="alice", k=20)
        all_of_it = await everything.aretrieve("me")
        stored = {n.node.get_content() for n in all_of_it}
        assert set(FACTS) <= stored

        def first(text: str) -> str:
            return text.splitlines()[0]

        q = GnarlMemoryBlock(client=ac, namespace=ns, user="alice", k=1)
        async def ask(question: str) -> str:
            return first(await q.aget([ChatMessage(role="user", content=question)]))

        assert await ask("what food should I cook for her?") == f"- {FACTS[0]}"
        assert await ask("which city is her home in?") == f"- {FACTS[1]}"
        assert await ask("does she own an animal?") == f"- {FACTS[2]}"

        out = await memory.aget(input="what should I eat tonight?")
        assert "<GnarlMemory>" in str(out[0].content) and "falafel" in str(out[0].content)

        bob = GnarlMemoryBlock(client=ac, namespace=ns, user="bob")
        assert await bob.aget([ChatMessage(role="user", content="falafel lisbon cat")]) == ""
        await bob.aremember("Bob plays the cello")
        assert await bob.aget([ChatMessage(role="user", content="pets")]) == "- Bob plays the cello"


def test_retriever_sync(client: Client, run_id: str) -> None:
    ns = f"it-li-{run_id}"
    for fact in FACTS:
        client.memory.remember(fact, namespace=ns, user="alice")
    retriever = GnarlRetriever(client, namespace=ns, user="alice", k=1)
    nodes = retriever.retrieve("does she have pets?")
    assert nodes[0].node.get_content() == FACTS[2]
    assert nodes[0].node.metadata["partial"] is False
    assert GnarlRetriever(client, namespace=ns, user="bob").retrieve("cat") == []
