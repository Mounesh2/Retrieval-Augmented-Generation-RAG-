import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        },
        timeout=600
    )

    r.raise_for_status()

    embedding = r.json()["embeddings"]

    return embedding


# Load saved embeddings
df = joblib.load("chunks/embeddings.joblib")

print("Total chunks:", len(df))

# Ask question
incoming_query = input("Ask a Question: ")

# Create embedding for question
question_embedding = create_embedding([incoming_query])[0]

# Find similarity between question and all chunks
similarities = cosine_similarity(
    np.vstack(df["embedding"].values),
    [question_embedding]
).flatten()

# Number of results to inspect
top_results = 30

# Get indexes of highest similarity
max_index = similarities.argsort()[::-1][0:top_results]

# Get top matching chunks
new_df = df.iloc[max_index].copy()

# Add similarity score
new_df["similarity"] = similarities[max_index]

print("\n" + "=" * 80)
print("TOP MATCHING RESULTS")
print("=" * 80)

for rank, (_, item) in enumerate(new_df.iterrows(), start=1):

    print("\n" + "-" * 80)

    print("Rank:", rank)
    print("Chunk ID:", item["chunk_id"])
    print("Video:", item["video"])
    print("Similarity:", round(float(item["similarity"]), 4))
    print("Start:", item["start"])
    print("End:", item["end"])

    print("\nText:")
    print(item["text"])