curl http://localhost:8080/embeddings \
                -H "Content-Type: application/json" \
                -d '{
                  "input": "The food was delicious and the waiter...",
                  "model": "all-MiniLM-L6-v2",
                  "encoding_format": "float"
                }'
