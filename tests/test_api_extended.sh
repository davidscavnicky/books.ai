#!/bin/bash
# Extended API testing script with 20+ test calls
# Tests content-based, item-item CF, and popularity methods with various books

BASE_URL="${BASE_URL:-http://localhost:5000}"

echo "=== Extended Book Recommender API Tests (20 calls) ==="
echo "Testing: ${BASE_URL}"
echo "This tests pre-computed matrix performance"
echo ""

# Health check
echo "1. Health Check"
curl -s "${BASE_URL}/healthz" | python3 -m json.tool
echo -e "\n---\n"

# Popularity tests (3 calls)
echo "2. Top 10 Popular Books"
curl -s "${BASE_URL}/recommend?method=pop&top_n=10" | python3 -m json.tool | head -20
echo -e "\n---\n"

echo "3. Top 3 Popular Books"
curl -s "${BASE_URL}/recommend?method=pop&top_n=3" | python3 -m json.tool
echo -e "\n---\n"

echo "4. Top 20 Popular Books"
curl -s "${BASE_URL}/recommend?method=pop&top_n=20" | python3 -m json.tool | head -30
echo -e "\n---\n"

# Content-Based tests (8 calls)
echo "5. Content: Lord of the Rings"
curl -s "${BASE_URL}/recommend?title=Lord%20of%20the%20Rings&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "6. Content: Harry Potter"
curl -s "${BASE_URL}/recommend?title=Harry%20Potter&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "7. Content: Da Vinci Code"
curl -s "${BASE_URL}/recommend?title=Da%20Vinci%20Code&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "8. Content: To Kill a Mockingbird"
curl -s "${BASE_URL}/recommend?title=To%20Kill%20a%20Mockingbird&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "9. Content: Pride and Prejudice"
curl -s "${BASE_URL}/recommend?title=Pride%20and%20Prejudice&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "10. Content: The Hobbit"
curl -s "${BASE_URL}/recommend?title=The%20Hobbit&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "11. Content: 1984"
curl -s "${BASE_URL}/recommend?title=1984&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "12. Content: The Great Gatsby"
curl -s "${BASE_URL}/recommend?title=The%20Great%20Gatsby&method=content&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

# Item-Item CF tests (8 calls)
echo "13. Item-Item CF: Lord of the Rings"
curl -s "${BASE_URL}/recommend?title=Lord%20of%20the%20Rings&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "14. Item-Item CF: Harry Potter"
curl -s "${BASE_URL}/recommend?title=Harry%20Potter&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "15. Item-Item CF: Da Vinci Code"
curl -s "${BASE_URL}/recommend?title=Da%20Vinci%20Code&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "16. Item-Item CF: The Hobbit"
curl -s "${BASE_URL}/recommend?title=The%20Hobbit&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "17. Item-Item CF: Lovely Bones"
curl -s "${BASE_URL}/recommend?title=Lovely%20Bones&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "18. Item-Item CF: Secret Life of Bees"
curl -s "${BASE_URL}/recommend?title=Secret%20Life%20of%20Bees&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "19. Item-Item CF: Divine Secrets"
curl -s "${BASE_URL}/recommend?title=Divine%20Secrets&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

echo "20. Item-Item CF: Wild Animus"
curl -s "${BASE_URL}/recommend?title=Wild%20Animus&method=item&top_n=5" | python3 -m json.tool
echo -e "\n---\n"

# Error cases (3 calls)
echo "21. Error: Missing title for content method"
curl -s "${BASE_URL}/recommend?method=content" | python3 -m json.tool
echo -e "\n---\n"

echo "22. Error: Invalid method"
curl -s "${BASE_URL}/recommend?title=Test&method=invalid" | python3 -m json.tool
echo -e "\n---\n"

echo "23. Error: Book not found"
curl -s "${BASE_URL}/recommend?title=ThisBookDoesNotExist12345&method=content" | python3 -m json.tool
echo -e "\n---\n"

echo "=== Test Complete: 23 API calls executed ==="
