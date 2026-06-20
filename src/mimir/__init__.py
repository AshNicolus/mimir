"""Mimir: experience-driven memory for autonomous agents."""

from importlib.metadata import PackageNotFoundError, version

from .core import Mimir
from .embeddings import Embedder, NullEmbedder
from .models import Experience, Outcome, Recommendation
from .storage import SQLiteStorage, Storage

try:
    __version__ = version("mimir-learn")
except PackageNotFoundError:  # running from source, not installed
    __version__ = "0.0.0"

__all__ = [
    "Mimir",
    "Experience",
    "Outcome",
    "Recommendation",
    "Storage",
    "SQLiteStorage",
    "Embedder",
    "NullEmbedder",
]
