"""Deprecated copy of the previous recommender implementation.

This file is kept as an archive so other scripts can be migrated gradually.
Do not import this file directly in new code; use the updated APIs in `booksai`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(books_path: str = "data/Books.csv", ratings_path: str = "data/Ratings.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(books_path) and os.path.exists(ratings_path):
        books = pd.read_csv(books_path, encoding='latin-1')
        ratings = pd.read_csv(ratings_path, encoding='latin-1')
    else:
        books = pd.DataFrame([
            {"isbn": "0001", "title": "The Hobbit", "author": "J.R.R. Tolkien"},
            {"isbn": "0002", "title": "The Lord of the Rings", "author": "J.R.R. Tolkien"},
            {"isbn": "0003", "title": "Silmarillion", "author": "J.R.R. Tolkien"},
            {"isbn": "0004", "title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling"},
            {"isbn": "0005", "title": "A Game of Thrones", "author": "George R.R. Martin"},
        ])
        ratings = pd.DataFrame([
            {"user_id": 1, "isbn": "0001", "rating": 5},
            {"user_id": 1, "isbn": "0002", "rating": 5},
            {"user_id": 1, "isbn": "0003", "rating": 3},
            {"user_id": 2, "isbn": "0002", "rating": 4},
            {"user_id": 2, "isbn": "0004", "rating": 5},
            {"user_id": 3, "isbn": "0005", "rating": 5},
            {"user_id": 3, "isbn": "0002", "rating": 4},
        ])

    books.columns = [c.strip() for c in books.columns]
    ratings.columns = [c.strip() for c in ratings.columns]

    if 'Book-Rating' in ratings.columns or 'Book-Rating' in ratings.columns:
        ratings = ratings.rename(columns={
            'Book-Rating': 'rating', 'User-ID': 'user_id', 'ISBN': 'isbn'
        })
    else:
        if ratings.shape[1] >= 3:
            ratings = ratings.rename(columns={ratings.columns[0]: 'user_id', ratings.columns[1]: 'isbn', ratings.columns[2]: 'rating'})

    if books.shape[1] >= 3:
        books = books.rename(columns={books.columns[0]: 'isbn', books.columns[1]: 'title', books.columns[2]: 'author'})

    for col in ['isbn', 'title', 'author']:
        if col not in books.columns:
            raise RuntimeError(f"Missing expected book column: {col}")

    if 'rating' in ratings.columns:
        ratings = ratings[ratings['rating'] > 0]

    ratings = ratings.merge(books[['isbn', 'title', 'author']], on='isbn', how='left')

    return books, ratings


def popularity_recommender(ratings: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, int]]:
    counts = ratings.groupby(['isbn', 'title']).size().reset_index(name='count')
    top = counts.sort_values('count', ascending=False).head(top_n)
    return list(top[['title', 'count']].itertuples(index=False, name=None))


def build_item_item_matrix(ratings: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str], dict]:
    item_codes, items = pd.factorize(ratings['isbn'])
    user_codes, users = pd.factorize(ratings['user_id'])
    n_items = len(items)
    n_users = len(users)
    rows = item_codes
    cols = user_codes
    data = ratings['rating'].astype(float).values
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_items, n_users))
    isbn_list = list(items)
    isbn_to_index = {isbn: idx for idx, isbn in enumerate(isbn_list)}
    return mat, isbn_list, isbn_to_index


def item_item_recommender(target_title: str, ratings: pd.DataFrame, books: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, float]]:
    mat, isbn_list, isbn_to_index = build_item_item_matrix(ratings)
    match = books[books['title'].str.contains(target_title, case=False, na=False)]
    if match.empty:
        match = books[books['title'].str.lower() == target_title.lower()]
    if match.empty:
        raise ValueError(f"Could not find book matching title: {target_title}")
    target_isbn = match.iloc[0]['isbn']
    if target_isbn not in isbn_to_index:
        raise ValueError("Target ISBN not present in ratings data")
    target_idx = isbn_to_index[target_isbn]
    target_vec = mat.getrow(target_idx)
    sims = cosine_similarity(target_vec, mat).flatten()
    sims[target_idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for idx in top_idx:
        if sims[idx] <= 0:
            continue
        isbn = isbn_list[idx]
        title_series = books.loc[books['isbn'] == isbn, 'title']
        if isinstance(title_series, pd.Series):
            title = title_series.iloc[0]
        else:
            title = title_series
        results.append((title, float(sims[idx])))
    return results


def content_based_recommender(target_title: str, books: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, float]]:
    corpus = (books['title'].fillna('') + ' ' + books['author'].fillna('')).astype(str)
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(corpus)
    matches = books[books['title'].str.contains(target_title, case=False, na=False)]
    if matches.empty:
        matches = books[books['title'].str.lower() == target_title.lower()]
    if matches.empty:
        raise ValueError(f"Could not find book matching title: {target_title}")
    target_idx = matches.index[0]
    sims = cosine_similarity(vec.getrow(target_idx), vec).flatten()
    sims[target_idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for idx in top_idx:
        if sims[idx] <= 0:
            continue
        title = books.loc[idx, 'title']
        results.append((title, float(sims[idx])))
    return results


if __name__ == "__main__":
    books_df, ratings_df = load_data()
    print("Books:", books_df.shape)
    print("Ratings:", ratings_df.shape)
    print('\nTop popular:')
    for title, cnt in popularity_recommender(ratings_df, top_n=5):
        print(f" - {title} ({cnt})")
    query = 'Lord of the Rings'
    try:
        print(f"\nItem-item recommendations for '{query}':")
        for title, score in item_item_recommender(query, ratings_df, books_df, top_n=5):
            print(f" - {title} (score={score:.3f})")
    except Exception as exc:
        print('Item-item failed:', exc)
    try:
        print(f"\nContent-based recommendations for '{query}':")
        for title, score in content_based_recommender(query, books_df, top_n=5):
            print(f" - {title} (score={score:.3f})")
    except Exception as exc:
        print('Content-based failed:', exc)
