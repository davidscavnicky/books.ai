#!/usr/bin/env python3
"""Minimal Flask API exposing the recommender.

Endpoints:
  GET /recommend?title=...&method=content|item|pop&top_n=5

Notes:
- This uses `booksai.recommender` and will fall back to the built-in demo dataset if
  `data/Books.csv` and `data/Ratings.csv` are not present.
"""
from __future__ import annotations

import os
from typing import List, Union
from flask import Flask, request, jsonify

from booksai import recommender


app = Flask(__name__)

# load data once at startup (can be reloaded later)
BOOKS_PATH = os.environ.get("BOOKS_CSV", "data/Books.csv")
RATINGS_PATH = os.environ.get("RATINGS_CSV", "data/Ratings.csv")
books_df, ratings_df = recommender.load_data(BOOKS_PATH, RATINGS_PATH)


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
            res = recommender.content_based_recommender(title, books_df, top_n=top_n)
            return jsonify({"method": "content", "query": title, "results": _format_results(res)})

        elif method == "item":
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
