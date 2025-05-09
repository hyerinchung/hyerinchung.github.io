---
layout: md_layout
title: AI Chatbot
---

# AI Chatbot

![AI Chatbot illustration](https://github.com/hyerinchung/hyerinchung.github.io/blob/main/images/chatbot_head.jpg?raw=true)

## What is AI chatbot?

In today’s digital world, AI chatbots are becoming essential tools for businesses, offering human-like conversations through natural language processing. From virtual assistants like Siri and Alexa to customer support bots, they enhance user experience and streamline service. In this portfolio, I introduce an AI-powered troubleshooting chatbot I developed—designed to help users quickly resolve common product issues with intelligent, step-by-step guidance.

## How does the chatbot generate an answer?

<div align="center">
  <img src="https://github.com/hyerinchung/hyerinchung.github.io/blob/main/images/chatbot_diagram.png?raw=true" alt="AI Chatbot Logic" style="max-width: 35%; height: auto;">
</div>


## Vector Embedding and Sentence Transformer

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load a sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Sample corpus of documents
documents = [
    "Python is a programming language.",
    "The Eiffel Tower is in Paris.",
    "Cats are great pets.",
    "Machine learning enables computers to learn from data."
]

# 3. Convert documents to embeddings (vectors)
document_embeddings = model.encode(documents)

# 4. Create a FAISS index for Inner Product (used for cosine similarity if vectors are normalized)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product

# 5. Normalize vectors to use cosine similarity
faiss.normalize_L2(document_embeddings)
index.add(document_embeddings)  # Add vectors to the index

# 6. Encode a user query
query = "What is machine learning?"
query_embedding = model.encode([query])
faiss.normalize_L2(query_embedding)

# 7. Search for top-k most similar documents
top_k = 2
distances, indices = index.search(query_embedding, top_k)

# 8. Print the results
print("Query:", query)
print("\nTop results:")
for i in range(top_k):
    print(f"{i+1}: {documents[indices[0][i]]} (Score: {distances[0][i]:.4f})")

```

## Resut

```python
Query: What is machine learning?

Top results:
1: Machine learning enables computers to learn from data. (Score: 0.9123)
2: Python is a programming language. (Score: 0.5348)

```
