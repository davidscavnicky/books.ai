# Data Directory

This directory is for storing the book recommendation dataset files.

## Expected Files

- `Books.csv` - Book metadata (ISBN, title, author, publisher, year, etc.)
- `Ratings.csv` - User ratings data (user_id, ISBN, rating)

## Getting the Data

### Option 1: Automatic Download (Recommended)
The scripts automatically download the dataset from Kaggle when these files are missing:
```bash
PYTHONPATH=src python scripts/book_recommender_demo.py
```

### Option 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
2. Download `Books.csv` and `Ratings.csv`
3. Place them in this `data/` directory

### Option 3: Kaggle CLI
```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('arashnic/book-recommendation-dataset')"
```

## Dataset Info

- **Source**: Kaggle - Book Recommendation Dataset
- **Books**: ~271,000 books
- **Ratings**: ~1,150,000 user ratings
- **Format**: CSV files with headers

## Note

This directory is typically added to `.gitignore` to avoid committing large datasets to version control.
The dataset is cached at: `~/.cache/kagglehub/datasets/arashnic/book-recommendation-dataset/`
