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

**Books.csv** (271,360 books):
- ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher
- Image URLs (S, M, L)
- **No book descriptions, summaries, genres, or keywords**

**Ratings.csv** (1,149,780 ratings):
- User-ID, ISBN, Book-Rating (0-10 scale)
- **62% are implicit zeros** (716,109 out of 1.1M)
- Explicit ratings: 433,671 (ratings > 0)

### ⚠️ Dataset Limitations

**This dataset has significant limitations for recommendation systems:**

1. **No Rich Content Features**: Books only have title, author, publisher, and year. No descriptions, summaries, genres, keywords, or tags that would enable meaningful semantic content-based recommendations.

2. **Mostly Implicit Zeros**: 62% of ratings are 0s (likely meaning "not rated" rather than "terrible book"), creating noise in collaborative filtering.

3. **Sparse Rating Matrix**: Most books have very few ratings, making collaborative filtering challenging for the long tail.

4. **Content-Based is Limited**: With only title/author/publisher/year, content-based recommendations essentially become "find books with similar title words or same author" - not true semantic similarity.

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
- Groups ratings by book (filtering zeros)
- Sorts by number of ratings and average rating
- Returns top-N most popular books

**Use case:** Cold start, trending books, general recommendations

**Limitations:** No personalization, popularity bias (favors books with many ratings)

### 2. Content-Based Filtering

TF-IDF similarity on **limited** book metadata:
- Combines title, author, publisher, year into text corpus
- Builds TF-IDF matrix with unigrams and bigrams
- Computes cosine similarity between books
- Returns most similar books to query

**What it actually does:**
- Finds books with similar titles (e.g., "The Lord of the Rings" → other LOTR editions)
- Finds books by the same author
- Finds books from the same publisher/era

**⚠️ Critical Limitation:** Without book descriptions, genres, or summaries, this is NOT true semantic content-based filtering. It's essentially lexical matching on titles and metadata. For example:
- "Harry Potter and the Sorcerer's Stone" → other Harry Potter books (good)
- "The Great Gatsby" → other books with "Great" in the title (not semantically meaningful)

**Use case:** "Find other books by this author" or "Find other editions of this book"

### 3. Item-Item Collaborative Filtering

User behavior-based recommendations:
- Builds sparse user-item matrix from ratings (filters implicit zeros)
- Filters active items (min 10 ratings) and users (min 5 ratings)
- Computes item-item cosine similarity
- Returns books that users co-rated with the query book

**What it captures:**
- Users who rated "The Lord of the Rings" also rated "The Hobbit" and "Silmarillion"
- Discovers patterns beyond metadata (e.g., readers who like fantasy epics)

**Limitations:** 
- Requires sufficient rating data (many books filtered out)
- O(n²) space complexity for similarity matrix
- Cold start problem for new books

**Use case:** "Users who liked this also liked..." recommendations based on actual user behavior

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

1. **Dataset Limitations (Most Critical)**
   - **No rich content features**: Dataset only has title/author/publisher/year - no descriptions, genres, keywords, or summaries
   - **Content-based is essentially lexical matching**: Without semantic features, recommendations are based on word overlap, not meaning
   - **62% implicit zeros**: Most "ratings" are likely missing data, not true negative feedback
   - **Sparse ratings**: Many books have few or no ratings

2. **Algorithm Limitations**
   - **Item-CF Scalability**: Dense similarity matrix (O(n²) space for n items with sufficient ratings)
   - **Cold Start**: New books/users have no recommendations
   - **No Personalization**: All users get same recommendations for a given book
   - **Simple Features**: Content-based only uses basic metadata (not true semantic similarity)

3. **Implementation Limitations**
   - Recommendations computed on-the-fly (no pre-computation)
   - No caching or optimization for serving
   - Single-machine implementation (not distributed)

### What This Implementation Actually Does Well

**Honest Assessment:**

✅ **Popularity Baseline**: Works perfectly for identifying trending/popular books

✅ **Item-Item CF**: Captures real user behavior patterns when enough rating data exists
- Example: LOTR fans also like The Hobbit, Silmarillion (meaningful co-rating patterns)

⚠️ **Content-Based**: Limited to lexical similarity, not semantic
- Good for: Finding other books by same author, other editions of same book
- Bad for: Finding semantically similar books (e.g., "other epic fantasy novels" requires genre/description data)

### Potential Improvements

**To Address Dataset Limitations:**
- **Scrape book descriptions** from GoodReads, Google Books API, or Amazon
- **Add genre/category tags** from library classification systems
- **Extract keywords** from book summaries using NLP
- **Use book cover images** with computer vision (ResNet, ViT)
- **Leverage pre-trained embeddings** (BERT on book descriptions)

**To Improve Algorithms:**
- Add user-based collaborative filtering
- Implement matrix factorization (SVD, ALS, NMF)
- Use neural collaborative filtering (NCF)
- Build hybrid models (combine content + collaborative)
- Add deep learning: Two-tower models, BERT-based semantic search
- Implement implicit feedback handling (better than filtering zeros)

**To Scale for Production:**
- Pre-compute item-item similarities (batch processing)
- Use approximate nearest neighbors (Annoy, FAISS) instead of exact cosine
- Add Redis caching for frequent queries
- Implement model serving with FastAPI + Gunicorn
- Use vector databases (Pinecone, Weaviate) for similarity search

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
- The Lord of the Rings (1.000) - exact match
- The Lord of the Rings (0.894) - different edition
- The Two Towers (The Lord of the Rings, Part 2) (0.880)
- The Return of the King (The Lord of the Rings, Part 3) (0.830)
- The Fellowship of the Ring (The Lord of the Rings, Part 1) (0.807)

**Analysis:** Content-based successfully finds other LOTR editions and trilogy books because they share title words. This is **lexical similarity**, not semantic understanding. It wouldn't find "other epic fantasy novels" without genre data.

**Item-CF for "The Lord of the Rings":**
- The Return of the King (0.54 similarity)
- The Fellowship of the Ring (0.42)
- The Hobbit (0.27)
- The Silmarillion (0.18)
- Crazy in Alabama (0.11) - likely spurious correlation

**Analysis:** Item-CF captures real user behavior - people who rated LOTR also rated other Tolkien books. Shows strong co-rating patterns among fantasy readers. The last item (Crazy in Alabama) shows noise from sparse data.

### Critical Observations

**Dataset Quality Issues:**
1. **62% implicit zeros**: Most "ratings" are 0, likely meaning "not rated" not "terrible book"
   - We filter these out, reducing dataset to 433K explicit ratings
   - But this removes potentially useful implicit feedback

2. **Content features are weak**: Without book descriptions/genres, content-based is limited to:
   - ✅ Finding other editions of the same book (good)
   - ✅ Finding other books by same author (good)
   - ❌ Finding semantically similar books (not possible with this data)

3. **Rating sparsity**: Many books have few ratings
   - Item-CF requires min 10 ratings per book → many books filtered out
   - Long-tail books get no collaborative recommendations

**What Works:**
- **Popularity baseline**: Reliable for cold start and general recommendations
- **Item-CF**: Captures meaningful patterns when sufficient data exists
- **Content-based for same-author**: Good for finding more books by authors you like

**What Doesn't Work Well:**
- **Content-based semantic similarity**: Impossible without descriptions/genres
- **Long-tail recommendations**: Sparse data limits collaborative filtering
- **Implicit feedback**: Zeros are ambiguous (not rated vs. disliked)

### Honest Assessment for Interview

**What I would discuss:**
1. "I chose this dataset because it's a standard benchmark, but I quickly realized it has serious limitations for content-based recommendations"
2. "The lack of book descriptions means my content-based approach is really just lexical matching, not semantic similarity"
3. "If I had more time, I would scrape book descriptions from GoodReads API or use pre-trained embeddings"
4. "The 62% implicit zeros suggest this data needs better handling (implicit feedback models vs. filtering)"
5. "Item-CF works well when data exists, but the sparse long tail is a challenge"

**Demonstrating critical thinking:**
- Acknowledge dataset limitations upfront
- Explain what each algorithm actually captures vs. what you'd want ideally
- Discuss tradeoffs between approaches
- Propose concrete improvements with data augmentation


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


