# booksai

A book recommendation system implementing three different approaches: popularity-based, content-based filtering, and item-item collaborative filtering.

## Overview

This project demonstrates different recommendation algorithms using the [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) (271K books, 1.1M ratings). The system implements:

1. **Popularity Baseline**: Recommends most-rated books
2. **Content-Based Filtering**: Uses TF-IDF on book metadata (title, author, publisher, year)
3. **Item-Item Collaborative Filtering**: Finds similar books based on user co-rating patterns

### Key Features

- **Automatic Dataset Download**: Uses `kagglehub` to auto-download the Kaggle dataset when local files are missing
- **3-Tier Fallback Strategy**: Local files → Kaggle download → Built-in demo dataset
- **Model Persistence**: Saves trained TF-IDF vectorizers and book dataframes to `models/`
- **REST API**: Flask API for serving recommendations
- **Production-Ready**: Type-checked (mypy), tested (pytest), CI/CD ready

## Project Structure

```
booksai/
├── src/booksai/
│   ├── recommender.py          # Core recommendation algorithms
│   └── _version.py             # Version management
├── scripts/
│   ├── train_recommender.py    # Training script (loads data, trains models, saves artifacts)
│   ├── api.py                  # Flask REST API for serving recommendations
│   └── recommend_example.py    # Simple CLI demo
├── tests/
│   └── recommender_test.py     # Unit tests for recommender functions
├── data/                        # Place Books.csv and Ratings.csv here (gitignored)
├── models/                      # Saved model artifacts (tfidf_vectorizer.pkl, books_df.pkl)
└── docs/                        # Sphinx documentation
```

## Quick Start

### 1. Training the Recommender

Run the training script to process the dataset and save trained models:

```bash
# With local data files (place in data/)
python scripts/train_recommender.py

# Or let it auto-download from Kaggle (requires kagglehub)
python scripts/train_recommender.py
```

**Output:**
- Prints top popular books
- Shows content-based recommendations for a sample query
- Shows item-based collaborative filtering recommendations
- Saves trained artifacts to `models/`

### 2. Using the REST API

Start the Flask API server:

```bash
python scripts/api.py
```

**Endpoints:**

```bash
# Content-based recommendations
curl "http://localhost:5000/recommend?title=Lord%20of%20the%20Rings&method=content&top_n=5"

# Item-item collaborative filtering
curl "http://localhost:5000/recommend?title=Harry%20Potter&method=item&top_n=5"

# Popularity baseline
curl "http://localhost:5000/recommend?method=pop&top_n=10"

# Health check
curl "http://localhost:5000/healthz"
```

### 3. Using as a Library

```python
from booksai import recommender

# Load data (auto-downloads from Kaggle if missing)
books, ratings = recommender.load_data()

# Get popular books
popular = recommender.popularity_recommender(ratings, top_n=10)
for title, count in popular:
    print(f"{title}: {count} ratings")

# Content-based recommendations
content_recs = recommender.content_based_recommender(
    "The Lord of the Rings", 
    books, 
    top_n=10
)
for title, score in content_recs:
    print(f"{title} (similarity: {score:.3f})")

# Item-item collaborative filtering
item_recs = recommender.item_item_recommender(
    "Harry Potter",
    ratings,
    books,
    top_n=10
)
for title, score in item_recs:
    print(f"{title} (similarity: {score:.3f})")
```

## Installation

### From Source

Clone the repository and install dependencies:

```bash
git clone https://github.com/davidscavnicky/books.ai.git
cd books.ai

# Create conda environment (recommended)
conda create -n books python=3.12
conda activate books

# Install package with dependencies
pip install -e .

# For training/running scripts, install recommender dependencies
pip install -e .[recommender]

# For API server
pip install -e .[web]

# For development (includes all extras)
pip install -e .[dev]
```

### Required Dependencies

**Core:**
- pandas 2.3.3+
- numpy 2.3.4+
- scipy 1.16.3+
- scikit-learn 1.7.2+

**Optional:**
- `kagglehub` - For automatic dataset download
- `flask` - For REST API server

## Dataset

The project uses the [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset):
- **271,360 books** with metadata (ISBN, title, author, publisher, year)
- **1,149,780 ratings** from users (User-ID, ISBN, Book-Rating)

**Data Loading Strategy:**
1. Tries loading from local `data/Books.csv` and `data/Ratings.csv`
2. If missing, auto-downloads from Kaggle using `kagglehub` (requires API credentials)
3. Falls back to small built-in demo dataset (5 books, 7 ratings)

To use local files, place them in the `data/` directory:
```bash
mkdir data
# Place Books.csv and Ratings.csv here
```

## Algorithm Details

### 1. Popularity Baseline

Simple counting approach:
- Groups ratings by book
- Sorts by number of ratings and average rating
- Returns top-N most popular books

**Use case:** Cold start, trending books, general recommendations

### 2. Content-Based Filtering

TF-IDF similarity on book metadata:
- Combines title, author, publisher, year into text corpus
- Builds TF-IDF matrix with unigrams and bigrams
- Computes cosine similarity between books
- Returns most similar books to query

**Use case:** "Find books like this one" based on metadata

### 3. Item-Item Collaborative Filtering

User behavior-based recommendations:
- Builds sparse user-item matrix from ratings
- Computes item-item cosine similarity
- Filters active items (min 10 ratings) and users (min 5 ratings)
- Returns books that users co-rated with the query book

**Use case:** "Users who liked this also liked..." recommendations

## Model Artifacts

Training saves the following to `models/`:
- `tfidf_vectorizer.pkl` - Trained TF-IDF vectorizer for content-based filtering
- `books_df.pkl` - Processed books dataframe with metadata

These can be loaded for serving predictions without retraining:

```python
import pickle

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/books_df.pkl', 'rb') as f:
    books = pickle.load(f)
```

## Usage

### Running the Tests

```bash
# Install test dependencies
pip install .[test]

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=booksai --cov-report=html
```

### Type Checking

```bash
# Install mypy
pip install mypy

# Run type checking
mypy src/booksai tests/
```

### Building the Documentation

The documentation is built with Sphinx and lives in the `docs/` directory.

1. Install documentation dependencies:

```bash
pip install .[docs]
```

2. Ensure `pandoc` is available (used by some Sphinx extensions). With conda:

```bash
conda install pandoc
```

3. Build the docs:

```bash
cd docs
make html
```

Could run `make clean` before you run `make html`.

The generated HTML will appear in `docs/_build/html`.

## Development

### Project Setup

```bash
# Clone and setup
git clone https://github.com/davidscavnicky/books.ai.git
cd books.ai

# Create environment
conda create -n books python=3.12
conda activate books

# Install with all dev dependencies
pip install -e .[dev]

# Setup git hooks (if .githooks/ exists)
git config core.hooksPath .githooks
```

### CI/CD

The project uses GitHub Actions for continuous integration:
- **Linting & Type Checking**: mypy on all source files
- **Testing**: pytest on Python 3.12
- **Documentation**: Builds and deploys to GitHub Pages

### Making Changes

1. Create a new branch: `git checkout -b feature-name`
2. Make changes and add tests
3. Run type checking: `mypy src/booksai tests/`
4. Run tests: `pytest tests/ -v`
5. Commit and push
6. Create pull request

## Limitations & Future Improvements

### Current Limitations

1. **Item-CF Scalability**: Dense similarity matrix (O(n²) space for n items with sufficient ratings)
2. **Cold Start**: New books/users have no recommendations (content-based helps but limited)
3. **No Personalization**: All users get same recommendations for a given book
4. **Simple Features**: Only uses basic metadata (title, author, publisher, year)

### Potential Improvements

**Short Term:**
- Add user-based collaborative filtering
- Implement matrix factorization (SVD, ALS)
- Add hybrid recommendations (combine multiple approaches)
- Include book descriptions/summaries in content features
- Add genre/category information

**Long Term:**
- Deploy as scalable microservice (Docker, Kubernetes)
- Add real-time recommendation updates
- Implement A/B testing framework
- Add recommendation explanations ("Because you liked X")
- Build user preference profiles
- Add deep learning models (neural collaborative filtering)

### Productionalization Architecture

For production deployment, consider:

**Data Pipeline:**
- Spark/Dask for large-scale data processing
- Airflow/Prefect for orchestration
- S3/GCS for data storage

**Model Serving:**
- FastAPI/Flask with Gunicorn/Uvicorn
- Redis for caching recommendations
- Pre-compute recommendations for popular items
- Model registry (MLflow) for versioning

**Infrastructure:**
- Docker containers
- Kubernetes for orchestration
- Load balancer (NGINX)
- Monitoring (Prometheus, Grafana)
- Logging (ELK stack)

**Database:**
- PostgreSQL for user/book metadata
- Redis for caching
- Vector database (Pinecone, Weaviate) for similarity search

## Results & Discussion

### Sample Results

**Popular Books (Most Rated):**
1. The Lovely Bones: A Novel (707 ratings, avg 8.2)
2. Wild Animus (581 ratings, avg 4.4)
3. The Da Vinci Code (487 ratings, avg 8.4)

**Content-Based for "The Lord of the Rings":**
- Finds different editions of LOTR trilogy
- Identifies related books (The Hobbit, Silmarillion)
- Similarity scores based on title/author overlap

**Item-CF for "The Lord of the Rings":**
- The Return of the King (0.54 similarity)
- The Fellowship of the Ring (0.42)
- The Hobbit (0.27)
- The Silmarillion (0.18)

Shows strong co-rating patterns among Tolkien fans.

### Observations

- **Popularity** works well for cold start but lacks personalization
- **Content-based** effectively finds similar books but limited by metadata quality
- **Item-CF** captures user behavior patterns but requires sufficient rating data
- Sparse ratings (many books have few ratings) limit collaborative filtering effectiveness


## GitHub Pages

Set up GitHub Pages to host the documentation (public repository):

1. Go to `https://github.com/davidscavnicky/books.ai/settings`
2. Navigate to Pages section
3. Select branch: `gh-pages`
4. Select directory: `root` or `/ (root)`
5. Save

Documentation will be available at: `https://davidscavnicky.github.io/books.ai`

### GitHub Actions Permissions

Under Settings → Actions → General:
- **Workflow Permissions**: Read and Write permissions
- Enable: "Allow GitHub Actions to approve pull requests"

## License

See LICENSE.rst file for details.

## Contact

**Author**: David Scavnicky  
**Repository**: https://github.com/davidscavnicky/books.ai

## Acknowledgments

- Dataset: [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) by arashnic
- Libraries: pandas, numpy, scipy, scikit-learn, Flask
- Inspired by classic recommendation system approaches


