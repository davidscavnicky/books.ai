"""
Test file for book recommender functionality.
"""

from booksai import recommender


def test_load_data_fallback():
    """Test that load_data returns synthetic fallback when files don't exist."""
    books, ratings = recommender.load_data("nonexistent.csv", "nonexistent.csv")
    assert books.shape[0] == 5  # 5 demo books
    assert ratings.shape[0] == 7  # 7 demo ratings
    assert 'isbn' in books.columns
    assert 'title' in books.columns
    assert 'rating' in ratings.columns


def test_popularity_recommender():
    """Test popularity-based recommendations."""
    books, ratings = recommender.load_data("nonexistent.csv", "nonexistent.csv")
    results = recommender.popularity_recommender(ratings, top_n=3)
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_content_based_recommender():
    """Test content-based recommendations."""
    books, ratings = recommender.load_data("nonexistent.csv", "nonexistent.csv")
    results = recommender.content_based_recommender("Lord of the Rings", books, top_n=3)
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
