#!/usr/bin/env python3
"""Example runner that demonstrates `booksai.recommender` usage.

This script will try to load `data/Books.csv` and `data/Ratings.csv` from the repository.
If they are not present it falls back to a small synthetic dataset so it runs as a smoke test.

Usage examples:
  python scripts/recommend_example.py --method all
  python scripts/recommend_example.py --method item --query "Lord of the Rings" --top-n 5
"""
from __future__ import annotations

import argparse
from typing import Optional

from booksai import recommender


def run(method: str = "all", books_path: Optional[str] = None, ratings_path: Optional[str] = None, query: str = "Lord of the Rings", top_n: int = 5) -> None:
    books, ratings = recommender.load_data(books_path or "data/Books.csv", ratings_path or "data/Ratings.csv")
    print('Loaded', books.shape, 'books and', ratings.shape, 'ratings')

    if method in ("all", "pop"):
        print('\nPopular:')
        for title, cnt in recommender.popularity_recommender(ratings, top_n=top_n):
            print(' -', title, cnt)

    if method in ("all", "item"):
        try:
            print(f"\nItem-item for '{query}':")
            for t, s in recommender.item_item_recommender(query, ratings, books, top_n=top_n):
                print(' -', t, f"({s:.3f})")
        except Exception as e:
            print('Item-item error:', e)

    if method in ("all", "content"):
        try:
            print(f"\nContent-based for '{query}':")
            for t, s in recommender.content_based_recommender(query, books, top_n=top_n):
                print(' -', t, f"({s:.3f})")
        except Exception as e:
            print('Content error:', e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run example book recommender")
    parser.add_argument("--method", choices=["all", "pop", "item", "content"], default="all", help="Which recommender to run")
    parser.add_argument("--books", help="Path to Books.csv")
    parser.add_argument("--ratings", help="Path to Ratings.csv")
    parser.add_argument("--query", default="Lord of the Rings", help="Query book title for similarity-based recommenders")
    parser.add_argument("--top-n", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()
    run(method=args.method, books_path=args.books, ratings_path=args.ratings, query=args.query, top_n=args.top_n)


if __name__ == "__main__":
    main()
