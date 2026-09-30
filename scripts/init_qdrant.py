"""Initialize Qdrant collections."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.qdrant.collections import create_collections, get_collection_stats


if __name__ == "__main__":
    print("Initializing Qdrant collections...")
    create_collections()
    print("\nCollection stats:")
    stats = get_collection_stats()
    for name, info in stats.items():
        print(f"  {name}: {info}")
    print("\nDone!")
