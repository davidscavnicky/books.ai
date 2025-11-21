"""Cross-validation and evaluation metrics for recommendation systems."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold


def train_test_split_by_user(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train/test sets, keeping some ratings per user in test.
    
    Args:
        ratings: DataFrame with columns ['user_id', 'isbn', 'rating']
        test_size: Fraction of ratings per user to hold out for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, test_df
    """
    np.random.seed(random_state)
    
    train_list = []
    test_list = []
    
    for user_id, user_ratings in ratings.groupby('user_id'):
        n_ratings = len(user_ratings)
        if n_ratings < 2:
            # Keep users with only 1 rating in training
            train_list.append(user_ratings)
            continue
            
        # Shuffle and split
        shuffled = user_ratings.sample(frac=1, random_state=random_state)
        n_test = max(1, int(n_ratings * test_size))
        
        test_list.append(shuffled.iloc[:n_test])
        train_list.append(shuffled.iloc[n_test:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, test_df


def precision_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    """
    Calculate Precision@K: fraction of recommended items that are relevant.
    
    Args:
        recommended: List of recommended ISBNs (ordered by score)
        relevant: List of ISBNs the user actually liked in test set
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score
    """
    if k == 0 or len(recommended) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for isbn in recommended_k if isbn in relevant_set)
    
    return hits / k


def recall_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    """
    Calculate Recall@K: fraction of relevant items that were recommended.
    
    Args:
        recommended: List of recommended ISBNs
        relevant: List of ISBNs the user actually liked in test set
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for isbn in recommended_k if isbn in relevant_set)
    
    return hits / len(relevant)


def mean_average_precision(
    recommended_lists: List[List[str]],
    relevant_lists: List[List[str]],
    k: int = 10
) -> float:
    """
    Calculate Mean Average Precision (MAP@K).
    
    Args:
        recommended_lists: List of recommendation lists (one per user)
        relevant_lists: List of relevant item lists (one per user)
        k: Number of recommendations to consider
        
    Returns:
        MAP@K score
    """
    if len(recommended_lists) != len(relevant_lists):
        raise ValueError("Recommended and relevant lists must have same length")
    
    aps = []
    for recommended, relevant in zip(recommended_lists, relevant_lists):
        if len(relevant) == 0:
            continue
            
        relevant_set = set(relevant)
        hits = 0
        sum_precisions = 0.0
        
        for i, isbn in enumerate(recommended[:k], 1):
            if isbn in relevant_set:
                hits += 1
                sum_precisions += hits / i
        
        if hits > 0:
            aps.append(sum_precisions / min(len(relevant), k))
    
    return np.mean(aps) if aps else 0.0


def evaluate_recommender(
    recommender_func,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    books: pd.DataFrame,
    k: int = 10,
    rating_threshold: float = 7.0
) -> Dict[str, float]:
    """
    Evaluate a recommender on test set.
    
    Args:
        recommender_func: Function that takes (user_id, train_ratings, books, k) 
                          and returns list of (isbn, score) tuples
        train_ratings: Training ratings
        test_ratings: Test ratings
        books: Books DataFrame
        k: Number of recommendations
        rating_threshold: Minimum rating to consider as "relevant"
        
    Returns:
        Dictionary with precision@k, recall@k, map@k
    """
    precisions = []
    recalls = []
    recommended_all = []
    relevant_all = []
    
    # Get users who appear in both train and test
    test_users = test_ratings['user_id'].unique()
    train_users = set(train_ratings['user_id'].unique())
    common_users = [u for u in test_users if u in train_users]
    
    for user_id in common_users[:100]:  # Limit to 100 users for speed
        # Get user's test items (ground truth)
        user_test = test_ratings[test_ratings['user_id'] == user_id]
        relevant = user_test[user_test['rating'] >= rating_threshold]['isbn'].tolist()
        
        if len(relevant) == 0:
            continue
        
        try:
            # Get recommendations
            recs = recommender_func(user_id, train_ratings, books, k)
            recommended = [isbn for isbn, _ in recs]
            
            # Calculate metrics
            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            recommended_all.append(recommended)
            relevant_all.append(relevant)
        except Exception:
            # Skip users where recommender fails
            continue
    
    return {
        'precision@k': np.mean(precisions) if precisions else 0.0,
        'recall@k': np.mean(recalls) if recalls else 0.0,
        'map@k': mean_average_precision(recommended_all, relevant_all, k),
        'n_users_evaluated': len(precisions)
    }


def cross_validate_recommender(
    recommender_func,
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    n_splits: int = 5,
    k: int = 10,
    rating_threshold: float = 7.0,
    random_state: int = 42
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation on a recommender.
    
    Args:
        recommender_func: Recommender function to evaluate
        ratings: Full ratings DataFrame
        books: Books DataFrame
        n_splits: Number of CV folds
        k: Number of recommendations
        rating_threshold: Minimum rating to consider as relevant
        random_state: Random seed
        
    Returns:
        Dictionary with lists of scores for each fold
    """
    # Filter to users with enough ratings
    user_counts = ratings.groupby('user_id').size()
    valid_users = user_counts[user_counts >= n_splits].index
    ratings_filtered = ratings[ratings['user_id'].isin(valid_users)]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = {
        'precision@k': [],
        'recall@k': [],
        'map@k': []
    }
    
    print(f"Running {n_splits}-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(ratings_filtered), 1):
        print(f"  Fold {fold}/{n_splits}...", end=' ')
        
        train_df = ratings_filtered.iloc[train_idx]
        test_df = ratings_filtered.iloc[test_idx]
        
        metrics = evaluate_recommender(
            recommender_func,
            train_df,
            test_df,
            books,
            k,
            rating_threshold
        )
        
        results['precision@k'].append(metrics['precision@k'])
        results['recall@k'].append(metrics['recall@k'])
        results['map@k'].append(metrics['map@k'])
        
        print(f"P@{k}={metrics['precision@k']:.4f}, R@{k}={metrics['recall@k']:.4f}, MAP@{k}={metrics['map@k']:.4f}")
    
    # Print summary
    print(f"\nCross-validation results (mean ± std):")
    for metric, values in results.items():
        print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return results
