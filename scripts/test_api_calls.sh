#!/bin/bash
# Test script for the recommender API
# Run the Flask API first: python scripts/recommender_api.py

API_URL="http://localhost:5000"

# Use jq if available, otherwise use python for pretty printing
if command -v jq &> /dev/null; then
    FORMAT="jq"
else
    FORMAT="python3 -m json.tool"
fi

echo "=== Testing Recommender API ==="
echo ""

echo "1. Health Check"
curl -s "$API_URL/healthz" | $FORMAT
echo -e "\n"

echo "2. Popularity Recommendations (Top 5)"
curl -s "$API_URL/recommend?title=any&method=pop&top_n=5" | $FORMAT
echo -e "\n"

echo "3. Content-Based: Harry Potter"
curl -s "$API_URL/recommend?title=Harry%20Potter&method=content&top_n=5" | $FORMAT
echo -e "\n"

echo "4. Item-Item CF: Lord of the Rings"
curl -s "$API_URL/recommend?title=Lord%20of%20the%20Rings&method=item&top_n=5" | $FORMAT
echo -e "\n"

echo "5. Content-Based: Da Vinci Code"
curl -s "$API_URL/recommend?title=Da%20Vinci%20Code&method=content&top_n=3" | $FORMAT
echo -e "\n"

echo "6. Error Case: Missing Title"
curl -s "$API_URL/recommend?method=content" | $FORMAT
echo -e "\n"

echo "=== All Tests Complete ==="
