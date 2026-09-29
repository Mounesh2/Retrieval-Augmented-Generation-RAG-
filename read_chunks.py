import requests
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CREATE EMBEDDINGS USING OLLAMA
# ============================================================

def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        },
        timeout=600
    )

    if not r.ok:
        print("Ollama error:", r.text)

    r.raise_for_status()

    return r.json()["embeddings"]


# ============================================================
# 2. LOAD ALL CHUNKS
# ============================================================

with open(
    "chunks/all_chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print("Total chunks:", len(chunks))


# ============================================================
# 3. CREATE EMBEDDINGS IN BATCHES
# ============================================================

BATCH_SIZE = 20

all_embeddings = []


for start in range(
    0,
    len(chunks),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(chunks)
    )

    batch = chunks[start:end]

    texts = [
        chunk["text"]
        for chunk in batch
    ]

    print(
        f"Creating embeddings: "
        f"{start + 1}-{end} / {len(chunks)}"
    )

    embeddings = create_embedding(texts)

    all_embeddings.extend(embeddings)


print(
    "\nEmbeddings created:",
    len(all_embeddings)
)


# ============================================================
# 4. ADD EMBEDDINGS TO CHUNKS
# ============================================================

for i, chunk in enumerate(chunks):

    chunk["embedding"] = all_embeddings[i]


# ============================================================
# 5. CREATE PANDAS DATAFRAME
# ============================================================

df = pd.DataFrame.from_records(chunks)


print(
    "\nDataFrame created successfully!"
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    df.columns.tolist()
)

print(
    "Embedding dimensions:",
    len(df.iloc[0]["embedding"])
)


# ============================================================
# 6. SHOW FIRST 5 CHUNKS
# ============================================================

print("\nFirst 5 chunks:")

print(
    df[
        [
            "chunk_id",
            "video",
            "text"
        ]
    ].head()
)


# ============================================================
# 7. SAVE DATAFRAME USING JOBLIB
# ============================================================

joblib.dump(
    df,
    "chunks/embeddings.joblib"
)

print(
    "\nDataFrame saved successfully!"
)

print(
    "Saved to: chunks/embeddings.joblib"
)


# ============================================================
# 8. FUNCTION TO FIND TOP MATCHING CHUNKS
# ============================================================

def get_top_matching_chunks(
    query,
    top_k=5
):

    # Create embedding for user question

    query_embedding = create_embedding(
        [query]
    )[0]


    # Get embeddings from DataFrame

    chunk_embeddings = np.array(
        df["embedding"].tolist()
    )


    # Calculate cosine similarity

    similarities = cosine_similarity(
        [query_embedding],
        chunk_embeddings
    )[0]


    # Get indexes of top matches

    top_indexes = similarities.argsort()[
        -top_k:
    ][::-1]


    # Get matching chunks

    top_chunks = df.iloc[
        top_indexes
    ].copy()


    # Add similarity score

    top_chunks["similarity"] = similarities[
        top_indexes
    ]


    return top_chunks


# ============================================================
# 9. TEST QUERY
# ============================================================

query = "What is HTML?"

print(
    "\nUser Question:",
    query
)


# ============================================================
# 10. GET TOP 5 MATCHING CHUNKS
# ============================================================

top_chunks = get_top_matching_chunks(
    query,
    top_k=5
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "TOP 5 MATCHING CHUNKS"
)

print(
    "=" * 80
)


for _, chunk in top_chunks.iterrows():

    print(
        "\nChunk ID:",
        chunk["chunk_id"]
    )

    print(
        "Video:",
        chunk["video"]
    )

    print(
        "Similarity:",
        round(
            float(chunk["similarity"]),
            4
        )
    )

    print(
        "Text:",
        chunk["text"]
    )

    print(
        "-" * 80
    )