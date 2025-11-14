"""Example runner that demonstrates `booksai.recommender` usage.

This script will try to load `data/Books.csv` and `data/Ratings.csv` from the repository.
If they are not present it falls back to a small synthetic dataset so it runs as a smoke test.
"""
from booksai import recommender


def main():
    books, ratings = recommender.load_data()
    print('Loaded', books.shape, 'books and', ratings.shape, 'ratings')

    print('\nPopular:')
    for title, cnt in recommender.popularity_recommender(ratings, top_n=5):
        print(' -', title, cnt)

    query = 'Lord of the Rings'
    try:
        print(f"\nItem-item for '{query}':")
        for t, s in recommender.item_item_recommender(query, ratings, books, top_n=5):
            print(' -', t, f"({s:.3f})")
    except Exception as e:
        print('Item-item error:', e)

    try:
        print(f"\nContent-based for '{query}':")
        for t, s in recommender.content_based_recommender(query, books, top_n=5):
            print(' -', t, f"({s:.3f})")
    except Exception as e:
        print('Content error:', e)


if __name__ == '__main__':
    main()
