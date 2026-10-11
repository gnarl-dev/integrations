"""LangChain and LangGraph integrations for Gnarl agent memory.

- :class:`GnarlStore` — LangGraph long-term memory (``BaseStore``).
- :class:`GnarlChatMessageHistory` — LangChain chat history, one memory per turn.
- :class:`GnarlRetriever` — LangChain retriever over semantic recall.

All three are thin wrappers over the published ``gnarl-client``; point them at
a node with ``$GNARL_URL`` or pass a :class:`gnarl.Client`.
"""

from ._gnarl import IncompleteRecall, RecallResult
from .chat_history import GnarlChatMessageHistory
from .retriever import GnarlRetriever
from .store import GnarlStore

__all__ = [
    "GnarlChatMessageHistory",
    "GnarlRetriever",
    "GnarlStore",
    "IncompleteRecall",
    "RecallResult",
]

__version__ = "0.1.0"
