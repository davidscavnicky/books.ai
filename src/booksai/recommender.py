"""Book recommendation system implementation.

Provides three recommendation algorithms:
- Popularity-based: Most rated books
- Content-based: TF-IDF similarity on title/author
- Item-item collaborative filtering: Co-occurrence matrix
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
        books = pd.read_csv(books_path, encoding='latin-1', low_memory=False)
        ratings = pd.read_csv(ratings_path, encoding='latin-1', low_memory=False)
    else:
        # Fallback to demo dataset
        demo_books = "data/demo_data/demo_books.csv"
        demo_ratings = "data/demo_data/demo_ratings.csv"
        if os.path.exists(demo_books) and os.path.exists(demo_ratings):
            print(f"Using demo dataset from {demo_books} and {demo_ratings}")
            books = pd.read_csv(demo_books)
            ratings = pd.read_csv(demo_ratings)
        else:
            raise FileNotFoundError(
                f"Could not find data files. Expected: {books_path} and {ratings_path}, "
                f"or demo files: {demo_books} and {demo_ratings}"
            )

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
    """Item-item collaborative filtering using on-demand matrix computation.
    
    Note: This rebuilds the matrix on each call. For production use, consider
    using item_item_recommender_precomputed() with pre-trained similarity matrix.
    """
    mat, isbn_list, isbn_to_index = build_item_item_matrix(ratings)
    
    # Check if matrix is empty
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError("Ratings matrix is empty")
    
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
    
    # Get top indices, filtering out negative similarities
    valid_indices = np.where(sims > 0)[0]
    if len(valid_indices) == 0:
        return []  # No similar items found
    
    valid_sims = sims[valid_indices]
    top_local_idx = np.argsort(valid_sims)[::-1][:top_n]
    top_idx = valid_indices[top_local_idx]
    
    results = []
    for idx in top_idx:
        isbn = isbn_list[idx]
        matching_books = books[books['isbn'] == isbn]
        if matching_books.empty:
            continue
        title = matching_books.iloc[0]['title']
        results.append((title, float(sims[idx])))
    return results


def item_item_recommender_precomputed(
    target_title: str,
    books: pd.DataFrame,
    item_matrix,
    item_sim,
    isbn_to_itemidx: dict,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """Item-item collaborative filtering using pre-computed similarity matrix.
    
    This is the production-ready version that uses pre-trained artifacts from
    train_recommender.py, avoiding expensive matrix computation on each request.
    
    Args:
        target_title: Book title to find recommendations for
        books: DataFrame with book metadata (isbn, title, author)
        item_matrix: Pre-computed item matrix from training
        item_sim: Pre-computed pairwise similarity matrix
        isbn_to_itemidx: Dict mapping ISBN to matrix index
        top_n: Number of recommendations to return
    
    Returns:
        List of (title, similarity_score) tuples
    """
    # Find the target book
    match = books[books['title'].str.contains(target_title, case=False, na=False)]
    if match.empty:
        match = books[books['title'].str.lower() == target_title.lower()]
    if match.empty:
        raise ValueError(f"Could not find book matching title: {target_title}")
    
    target_isbn = match.iloc[0]['isbn']
    if target_isbn not in isbn_to_itemidx:
        raise ValueError(f"Target ISBN {target_isbn} not present in pre-computed similarity matrix")
    
    # Get pre-computed similarities for this book
    target_idx = isbn_to_itemidx[target_isbn]
    sims = item_sim[target_idx]
    
    # Get top similar items (excluding the target itself)
    top_idx = sims.argsort()[::-1]
    
    results = []
    for idx in top_idx:
        if idx == target_idx:
            continue
        if sims[idx] <= 0:
            continue
            
        rec_isbn = item_matrix.index[idx]
        matching_books = books[books['isbn'] == rec_isbn]
        if matching_books.empty:
            continue
            
        title = matching_books.iloc[0]['title']
        results.append((title, float(sims[idx])))
        
        if len(results) >= top_n:
            break
    
    return results


def content_based_recommender(target_title: str, books: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, float]]:
    """Content-based filtering using on-demand TF-IDF computation.
    
    Note: This rebuilds the TF-IDF matrix on each call. For production use, consider
    using content_based_recommender_precomputed() with pre-trained TF-IDF matrix.
    """
    corpus = (books['title'].fillna('') + ' ' + books['author'].fillna('')).astype(str)
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(corpus)
    matches = books[books['title'].str.contains(target_title, case=False, na=False)]
    if matches.empty:
        matches = books[books['title'].str.lower() == target_title.lower()]
    if matches.empty:
        raise ValueError(f"Could not find book matching title: {target_title}")
    target_idx = matches.index[0]
    # extract a sparse row vector to avoid using __getitem__ on spmatrix
    target_vec = vec.getrow(target_idx)
    sims = cosine_similarity(target_vec, vec).flatten()
    sims[target_idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for idx in top_idx:
        if sims[idx] <= 0:
            continue
        title = books.loc[idx, 'title']
        results.append((title, float(sims[idx])))
    return results


def content_based_recommender_precomputed(
    target_title: str,
    books: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """Content-based filtering using pre-computed TF-IDF matrix.
    
    This is the production-ready version that uses pre-trained TF-IDF matrix from
    train_recommender.py, avoiding expensive matrix computation on each request.
    
    Args:
        target_title: Book title to find recommendations for
        books: DataFrame with book metadata (must match training data order)
        tfidf_matrix: Pre-computed TF-IDF matrix from training
        top_n: Number of recommendations to return
    
    Returns:
        List of (title, similarity_score) tuples
    """
    # Find the target book
    matches = books[books['title'].str.contains(target_title, case=False, na=False)]
    if matches.empty:
        matches = books[books['title'].str.lower() == target_title.lower()]
    if matches.empty:
        raise ValueError(f"Could not find book matching title: {target_title}")
    
    target_idx = matches.index[0]
    
    # Use pre-computed TF-IDF matrix
    target_vec = tfidf_matrix.getrow(target_idx)
    sims = cosine_similarity(target_vec, tfidf_matrix).flatten()
    sims[target_idx] = -1
    
    # Get top similar books
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
