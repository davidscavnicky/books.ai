#!/usr/bin/env python3
"""A direct runnable demo implementing the `book_recommender_demo.py` flow.

Usage:
  PYTHONPATH=src python scripts/book_recommender_demo.py

This script expects `data/Books.csv` and `data/Ratings.csv` in the repo root.
If missing it will raise; it's intended as a straight conversion of your demo.
"""
from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_models_dir():
    os.makedirs("models", exist_ok=True)


def load_data(books_path: str = "data/Books.csv", ratings_path: str = "data/Ratings.csv", auto_download: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load book and rating data from CSV files or download from Kaggle.
    
    The function implements a 3-tier fallback strategy:
    1. First, try loading from local `data/Books.csv` and `data/Ratings.csv`
    2. If files don't exist AND auto_download=True, download from Kaggle using kagglehub
    3. If download fails or is disabled, use a small built-in demo dataset
    
    This satisfies the requirement: "alternatively feel free to grab any other relevant data set"
    by automatically fetching the recommended Kaggle dataset when local files are missing.
    
    Args:
        books_path: Path to Books.csv (default: "data/Books.csv")
        ratings_path: Path to Ratings.csv (default: "data/Ratings.csv")
        auto_download: If True, automatically download from Kaggle when files are missing
    
    Returns:
        Tuple of (books_df, ratings_df)
    """
    if os.path.exists(books_path) and os.path.exists(ratings_path):
        print(f"Loading local dataset from {books_path} and {ratings_path}")
        books = pd.read_csv(books_path, encoding='latin-1')
        ratings = pd.read_csv(ratings_path, encoding='latin-1')
    elif auto_download:
        # Attempt to download the recommended Kaggle dataset
        try:
            print("Local dataset not found. Attempting to download from Kaggle...")
            print("Dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset")
            import kagglehub
            
            # Download the dataset (kagglehub caches it automatically)
            dataset_path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
            print(f"Dataset downloaded to: {dataset_path}")
            
            # Load from downloaded location
            books = pd.read_csv(os.path.join(dataset_path, "Books.csv"), encoding='latin-1')
            ratings = pd.read_csv(os.path.join(dataset_path, "Ratings.csv"), encoding='latin-1')
            print(f"Successfully loaded {len(books)} books and {len(ratings)} ratings from Kaggle")
        except Exception as e:
            print(f"Failed to download from Kaggle: {e}")
            print("Falling back to built-in demo dataset...")
            books, ratings = _get_demo_dataset()
    else:
        print("Using built-in demo dataset (auto_download=False)")
        books, ratings = _get_demo_dataset()
    
    return books, ratings


def _get_demo_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a small built-in demo dataset for testing without external data."""
    books = pd.DataFrame([
        {"isbn": "0001", "title": "The Hobbit", "author": "J.R.R. Tolkien", "Publisher": "Allen & Unwin", "Year-Of-Publication": "1937"},
        {"isbn": "0002", "title": "The Lord of the Rings", "author": "J.R.R. Tolkien", "Publisher": "Allen & Unwin", "Year-Of-Publication": "1954"},
        {"isbn": "0003", "title": "Silmarillion", "author": "J.R.R. Tolkien", "Publisher": "Allen & Unwin", "Year-Of-Publication": "1977"},
        {"isbn": "0004", "title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling", "Publisher": "Scholastic", "Year-Of-Publication": "1997"},
        {"isbn": "0005", "title": "A Game of Thrones", "author": "George R.R. Martin", "Publisher": "Bantam", "Year-Of-Publication": "1996"},
    ])
    ratings = pd.DataFrame([
        {"User-ID": 1, "ISBN": "0001", "Book-Rating": 5},
        {"User-ID": 1, "ISBN": "0002", "Book-Rating": 5},
        {"User-ID": 1, "ISBN": "0003", "Book-Rating": 3},
        {"User-ID": 2, "ISBN": "0002", "Book-Rating": 4},
        {"User-ID": 2, "ISBN": "0004", "Book-Rating": 5},
        {"User-ID": 3, "ISBN": "0005", "Book-Rating": 5},
        {"User-ID": 3, "ISBN": "0002", "Book-Rating": 4},
    ])
    return books, ratings


def prepare_books_ratings(books: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    books.columns = [c.strip() for c in books.columns]
    ratings.columns = [c.strip() for c in ratings.columns]

    # normalize rating column names
    if 'Book-Rating' in ratings.columns:
        ratings = ratings.rename(columns={'Book-Rating': 'rating', 'User-ID': 'user_id', 'ISBN': 'isbn'})
    else:
        if ratings.shape[1] >= 3:
            ratings = ratings.rename(columns={ratings.columns[0]: 'user_id', ratings.columns[1]: 'isbn', ratings.columns[2]: 'rating'})

    # normalize books
    if books.shape[1] >= 3:
        books = books.rename(columns={books.columns[0]: 'isbn', books.columns[1]: 'title', books.columns[2]: 'author'})

    for col in ['title', 'author', 'isbn']:
        if col not in books.columns:
            raise RuntimeError("Unexpected book columns: " + ",".join(books.columns))

    ratings = ratings[ratings['rating'] > 0]
    ratings = ratings.merge(books[['isbn', 'title', 'author']], on='isbn', how='left')
    return books, ratings


def popularity_baseline(ratings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    book_stats = ratings.groupby(['isbn', 'title']).agg(n_ratings=('rating', 'size'), avg_rating=('rating', 'mean')).reset_index()
    popularity = book_stats.sort_values(['n_ratings', 'avg_rating'], ascending=[False, False]).head(top_n)
    return popularity


def build_content_tfidf(books: pd.DataFrame) -> tuple[TfidfVectorizer, csr_matrix]:
    books = books.copy()
    books['title'] = books['title'].astype(str)
    books['author'] = books['author'].astype(str)
    books['pub'] = ''
    for col in ['Publisher', 'publisher', 'Publisher\r', 'publisher\r']:
        if col in books.columns:
            books['pub'] = books[col].fillna('').astype(str)
            break
    books['year'] = ''
    for col in ['Year-Of-Publication', 'Year', 'year', 'YearOfPublication']:
        if col in books.columns:
            books['year'] = books[col].fillna('').astype(str)
            break

    books['content'] = (books['title'] + ' ' + books['author'] + ' ' + books['pub'] + ' ' + books['year']).str.lower()

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words='english')
    tfidf_matrix = tf.fit_transform(books['content'].fillna(''))
    # ensure the returned matrix is a csr_matrix to match the function annotation
    tfidf_matrix = csr_matrix(tfidf_matrix)
    return tf, tfidf_matrix


def get_content_recommendations(title: str, books: pd.DataFrame, tfidf_matrix: csr_matrix, tf: TfidfVectorizer, topn: int = 10) -> List[tuple]:
    # build title->index mapping lowercased
    books_reset = books.reset_index(drop=True)
    title_to_idx = {str(t).lower(): idx for idx, t in enumerate(books_reset['title'])}
    
    title_lower = title.lower()
    if title_lower not in title_to_idx:
        matches = [t for t in title_to_idx.keys() if title_lower in t]
        if not matches:
            return []
        idx = title_to_idx[matches[0]]
    else:
        idx = title_to_idx[title_lower]
    
    vec = tfidf_matrix[idx]
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1]
    recs = []
    for i in top_idx:
        if i == idx:
            continue
        recs.append((str(books_reset.iloc[i]['title']), float(sims[i])))
        if len(recs) >= topn:
            break
    return recs


def build_item_cf(ratings: pd.DataFrame, min_user_ratings: int = 5, min_item_ratings: int = 10):
    item_counts = ratings['isbn'].value_counts()
    active_isbns = item_counts[item_counts >= min_item_ratings].index
    user_counts = ratings['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index

    r_small = ratings[ratings['isbn'].isin(active_isbns) & ratings['user_id'].isin(active_users)]
    pivot = r_small.pivot_table(index='user_id', columns='isbn', values='rating').fillna(0)
    item_matrix = pivot.T
    item_sparse = csr_matrix(item_matrix.values)
    # compute dense similarities (warning: may be large)
    item_sim = cosine_similarity(item_sparse)
    isbn_to_itemidx = {isbn: i for i, isbn in enumerate(item_matrix.index)}
    return item_matrix, item_sim, isbn_to_itemidx


def item_based_recs_by_title(seed_title: str, books: pd.DataFrame, item_matrix, item_sim, isbn_to_itemidx, topn: int = 10):
    rows = books[books['title'].str.lower().str.contains(seed_title.lower(), na=False)]
    if rows.empty:
        return []
    isbn = rows.iloc[0]['isbn']
    if isbn not in isbn_to_itemidx:
        return []
    idx = isbn_to_itemidx[isbn]
    sims = item_sim[idx]
    top_idx = sims.argsort()[::-1]
    recs = []
    for i in top_idx:
        if i == idx:
            continue
        rec_isbn = item_matrix.index[i]
        rec_title = books.loc[books['isbn'] == rec_isbn, 'title'].to_numpy()
        # use .size to safely check NumPy array length (avoids typing issues with datetime-like dtypes)
        if getattr(rec_title, "size", 0) > 0:
            recs.append((rec_title[0], float(sims[i])))
        if len(recs) >= topn:
            break
    return recs


def save_artifacts(tf, tfidf_matrix, books: pd.DataFrame):
    """Save TF-IDF vectorizer and books dataframe for later use."""
    ensure_models_dir()
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tf, f)
    # Save only the columns that exist
    cols_to_save = ['isbn', 'title', 'author']
    if 'content' in books.columns:
        cols_to_save.append('content')
    with open('models/books_df.pkl', 'wb') as f:
        pickle.dump(books[cols_to_save], f)


def main():
    books, ratings = load_data()
    print('Books:', books.shape)
    print('Ratings:', ratings.shape)

    books, ratings = prepare_books_ratings(books, ratings)

    print('\nTop popular:')
    pop = popularity_baseline(ratings, top_n=10)
    print(pop[['title', 'n_ratings', 'avg_rating']].head(10))

    tf, tfidf_matrix = build_content_tfidf(books)
    seed = 'The Lord of the Rings'
    c_recs = get_content_recommendations(seed, books, tfidf_matrix, tf, topn=10)
    print('\nContent-based recs for:', seed)
    for t, s in c_recs:
        print(f' - {t} (sim={s:.3f})')

    try:
        item_matrix, item_sim, isbn_to_itemidx = build_item_cf(ratings)
        ib_recs = item_based_recs_by_title(seed, books, item_matrix, item_sim, isbn_to_itemidx, topn=10)
        print('\nItem-based CF recs for', seed)
        for t, s in ib_recs:
            print(' -', t, s)
    except Exception as exc:
        print('Item-CF failed (likely small dataset or filtering):', exc)

    # Save artifacts
    try:
        save_artifacts(tf, tfidf_matrix, books)
        print('\nSaved TF-IDF and books dataframe into models/')
    except Exception as exc:
        print('Saving artifacts failed:', exc)


if __name__ == '__main__':
    main()
