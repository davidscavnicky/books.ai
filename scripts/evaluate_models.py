"""Script to evaluate and compare recommendation algorithms using cross-validation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from booksai import recommender, evaluation


def popularity_recommender_for_user(user_id, train_ratings, books, top_n=10):
    """Wrapper for popularity recommender (ignores user_id)."""
    popular = recommender.popularity_recommender(train_ratings, top_n=top_n)
    return [(row['isbn'], row['rating_count']) for _, row in popular.iterrows()]


def content_recommender_for_user(user_id, train_ratings, books, top_n=10):
    """Wrapper for content-based recommender."""
    # Get a book the user has rated highly
    user_ratings = train_ratings[train_ratings['user_id'] == user_id]
    if len(user_ratings) == 0:
        return []
    
    # Pick the highest-rated book by this user
    best_book = user_ratings.nlargest(1, 'rating').iloc[0]
    
    # Title is already in the ratings DataFrame
    title = best_book['title']
    
    # Get content-based recommendations
    try:
        recs = recommender.content_based_recommender(title, books, top_n=top_n)
        # recs is a list of (title, score) tuples
        # Map titles back to ISBNs
        return [(books[books['title'] == t]['isbn'].iloc[0], score) for t, score in recs if not books[books['title'] == t].empty]
    except Exception:
        return []


def item_cf_recommender_for_user(user_id, train_ratings, books, top_n=10):
    """Wrapper for item-item collaborative filtering."""
    # Get user's highest-rated book
    user_ratings = train_ratings[train_ratings['user_id'] == user_id]
    if len(user_ratings) == 0:
        return []
    
    best_book = user_ratings.nlargest(1, 'rating').iloc[0]
    title = best_book['title']
    
    # Get item-CF recommendations
    try:
        recs = recommender.item_item_recommender(title, train_ratings, books, top_n=top_n)
        return [(row['isbn'], row['similarity_score']) for _, row in recs.iterrows()]
    except Exception:
        return []


def main():
    print("=" * 80)
    print("Book Recommender Cross-Validation Evaluation")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    books, ratings = recommender.load_data()
    print(f"   Loaded {len(books)} books and {len(ratings)} ratings")
    
    # Filter to active users (at least 10 ratings) for meaningful evaluation
    print("\n2. Filtering to active users (min 10 ratings)...")
    user_counts = ratings.groupby('user_id').size()
    active_users = user_counts[user_counts >= 10].index
    ratings_filtered = ratings[ratings['user_id'].isin(active_users)]
    print(f"   Filtered to {len(active_users)} users, {len(ratings_filtered)} ratings")
    
    # Simple train/test split
    print("\n3. Creating train/test split (80/20)...")
    train_ratings, test_ratings = evaluation.train_test_split_by_user(
        ratings_filtered, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(train_ratings)} ratings")
    print(f"   Test: {len(test_ratings)} ratings")
    
    # Evaluate each algorithm
    k = 10
    print(f"\n4. Evaluating recommenders (top-{k} recommendations)...")
    print("-" * 80)
    
    algorithms = [
        ("Popularity Baseline", popularity_recommender_for_user),
        ("Content-Based (TF-IDF)", content_recommender_for_user),
        ("Item-Item Collaborative Filtering", item_cf_recommender_for_user),
    ]
    
    results_summary = []
    
    for name, recommender_func in algorithms:
        print(f"\n{name}:")
        metrics = evaluation.evaluate_recommender(
            recommender_func,
            train_ratings,
            test_ratings,
            books,
            k=k,
            rating_threshold=7.0
        )
        
        print(f"  Precision@{k}: {metrics['precision@k']:.4f}")
        print(f"  Recall@{k}:    {metrics['recall@k']:.4f}")
        print(f"  MAP@{k}:       {metrics['map@k']:.4f}")
        print(f"  Users evaluated: {metrics['n_users_evaluated']}")
        
        results_summary.append({
            'Algorithm': name,
            f'Precision@{k}': metrics['precision@k'],
            f'Recall@{k}': metrics['recall@k'],
            f'MAP@{k}': metrics['map@k'],
            'Users': metrics['n_users_evaluated']
        })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # Optional: Run cross-validation (time-consuming)
    run_cv = input("\n\nRun 5-fold cross-validation? (y/n): ").lower().strip() == 'y'
    
    if run_cv:
        print("\n5. Running 5-fold cross-validation (this may take several minutes)...")
        print("-" * 80)
        
        # Use smaller subset for CV
        sample_ratings = ratings_filtered.sample(min(10000, len(ratings_filtered)), random_state=42)
        
        for name, recommender_func in algorithms:
            print(f"\n{name}:")
            cv_results = evaluation.cross_validate_recommender(
                recommender_func,
                sample_ratings,
                books,
                n_splits=5,
                k=k,
                rating_threshold=7.0
            )
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
