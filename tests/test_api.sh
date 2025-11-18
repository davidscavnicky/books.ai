#!/bin/bash
# Quick API testing script

BASE_URL="${BASE_URL:-http://localhost:5000}"

echo "=== Book Recommender API Tests ==="
echo "Testing: ${BASE_URL}"
echo ""

echo "1. Health Check"
curl -s "${BASE_URL}/healthz" | python3 -m json.tool
echo ""

echo "2. Top 5 Popular Books"
curl -s "${BASE_URL}/recommend?method=pop&top_n=5" | python3 -m json.tool
echo ""

echo "3. Content-Based: Lord of the Rings"
curl -s "${BASE_URL}/recommend?title=Lord%20of%20the%20Rings&method=content&top_n=5" | python3 -m json.tool
echo ""

echo "4. Content-Based: Harry Potter"
curl -s "${BASE_URL}/recommend?title=Harry%20Potter&method=content&top_n=5" | python3 -m json.tool
echo ""

echo "5. Item-Item CF: Lord of the Rings"
curl -s "${BASE_URL}/recommend?title=Lord%20of%20the%20Rings&method=item&top_n=5" | python3 -m json.tool
echo ""

echo "6. Error Case: Missing title for content method"
curl -s "${BASE_URL}/recommend?method=content" | python3 -m json.tool
echo ""

echo "7. Error Case: Invalid method"
curl -s "${BASE_URL}/recommend?title=Test&method=invalid" | python3 -m json.tool
