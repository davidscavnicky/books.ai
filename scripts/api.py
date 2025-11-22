#!/usr/bin/env python3
"""Minimal Flask API exposing the recommender.

Endpoints:
  GET /recommend?title=...&method=content|item|pop&top_n=5

Notes:
- This uses `booksai.recommender` and will fall back to the built-in demo dataset if
  `data/Books.csv` and `data/Ratings.csv` are not present.
- For item-item CF, it attempts to load pre-computed similarity matrix from 
  models/item_similarity.pkl. If not found, it falls back to on-demand computation.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Union, Optional
from flask import Flask, request, jsonify

from booksai import recommender


app = Flask(__name__)

# load data once at startup (can be reloaded later)
BOOKS_PATH = os.environ.get("BOOKS_CSV", "data/Books.csv")
RATINGS_PATH = os.environ.get("RATINGS_CSV", "data/Ratings.csv")
books_df, ratings_df = recommender.load_data(BOOKS_PATH, RATINGS_PATH)

# Try to load pre-computed TF-IDF matrix
tfidf_matrix = None
CONTENT_TFIDF_PATH = "models/content_tfidf.pkl"

try:
    if os.path.exists(CONTENT_TFIDF_PATH):
        with open(CONTENT_TFIDF_PATH, 'rb') as f:
            tfidf_data = pickle.load(f)
        tfidf_matrix = tfidf_data['tfidf_matrix']
        print(f"✓ Loaded pre-computed TF-IDF matrix: {tfidf_matrix.shape}")
    else:
        print(f"⚠ Pre-computed TF-IDF matrix not found at {CONTENT_TFIDF_PATH}")
        print("  Content-based will use on-demand computation (slower)")
except Exception as e:
    print(f"⚠ Failed to load pre-computed TF-IDF matrix: {e}")
    print("  Content-based will use on-demand computation (slower)")

# Try to load pre-computed item-item similarity matrix
item_matrix = None
item_sim = None
isbn_to_itemidx = None
ITEM_SIM_PATH = "models/item_similarity.pkl"

try:
    if os.path.exists(ITEM_SIM_PATH):
        with open(ITEM_SIM_PATH, 'rb') as f:
            sim_data = pickle.load(f)
        item_matrix = sim_data['item_matrix']
        item_sim = sim_data['item_sim']
        isbn_to_itemidx = sim_data['isbn_to_itemidx']
        print(f"✓ Loaded pre-computed item-item similarity matrix: {item_sim.shape}")
    else:
        print(f"⚠ Pre-computed similarity matrix not found at {ITEM_SIM_PATH}")
        print("  Item-item CF will use on-demand computation (slower)")
except Exception as e:
    print(f"⚠ Failed to load pre-computed similarity matrix: {e}")
    print("  Item-item CF will use on-demand computation (slower)")

if tfidf_matrix is None or item_sim is None:
    print(f"\n💡 Run training to generate pre-computed matrices:")
    print(f"   PYTHONPATH=src python scripts/train_recommender.py\n")


def _format_results(results: List[tuple]) -> List[dict]:
    return [{"title": t, "score": float(s)} for t, s in results]


@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "missing 'title' parameter"}), 400

    method = request.args.get("method", "content")
    top_n = int(request.args.get("top_n", 10))

    try:
        if method == "content":
            # Use pre-computed TF-IDF matrix if available, otherwise fall back to on-demand
            if tfidf_matrix is not None:
                res = recommender.content_based_recommender_precomputed(
                    title, books_df, tfidf_matrix, top_n=top_n
                )
            else:
                # Fallback: rebuild TF-IDF on each request (slower)
                res = recommender.content_based_recommender(title, books_df, top_n=top_n)
            return jsonify({"method": "content", "query": title, "results": _format_results(res)})

        elif method == "item":
            # Use pre-computed similarity matrix if available, otherwise fall back to on-demand
            if item_sim is not None and item_matrix is not None and isbn_to_itemidx is not None:
                res = recommender.item_item_recommender_precomputed(
                    title, books_df, item_matrix, item_sim, isbn_to_itemidx, top_n=top_n
                )
            else:
                # Fallback: rebuild matrix on each request (slower)
                res = recommender.item_item_recommender(title, ratings_df, books_df, top_n=top_n)
            return jsonify({"method": "item", "query": title, "results": _format_results(res)})

        elif method == "pop":
            pop_res = recommender.popularity_recommender(ratings_df, top_n=top_n)
            # popularity returns (title, count)
            return jsonify({"method": "pop", "results": [{"title": t, "count": int(c)} for t, c in pop_res]})

        else:
            return jsonify({"error": f"unknown method '{method}'"}), 400

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
