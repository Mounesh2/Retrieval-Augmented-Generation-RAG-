import numpy as np
import joblib
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity


# =========================================
# LOAD EMBEDDINGS
# =========================================

df = joblib.load("chunks/embeddings.joblib")


# =========================================
# CREATE EMBEDDING
# =========================================

def create_embedding(text):

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": [text]
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


# =========================================
# RETRIEVE RELEVANT CHUNKS
# =========================================

def retrieve(question):

    question_embedding = create_embedding(question)

    similarities = cosine_similarity(
        np.vstack(df["embedding"].values),
        [question_embedding]
    ).flatten()

    top_results = 10

    max_index = similarities.argsort()[::-1][:top_results]

    top_chunks = df.iloc[max_index].copy()

    top_chunks["similarity"] = similarities[max_index]

    threshold = 0.60

    relevant_chunks = top_chunks[
        top_chunks["similarity"] >= threshold
    ].copy()

    return relevant_chunks


# =========================================
# GENERATE ANSWER
# =========================================

def generate_answer(question):

    chunks = retrieve(question)

    if len(chunks) == 0:

        return (
            "I could not find the answer in the "
            "provided course material."
        )


    context = ""

    for _, item in chunks.iterrows():

        context += f"""
{item["text"]}

--------------------
"""


    prompt = f"""
You are an AI assistant for the Sigma Web Development course.

Answer the user's question using ONLY the course material below.

Do not use your general knowledge.

Do not invent information.

Give a clear and concise answer.

If the answer is not present in the course material, say:

"I could not find the answer in the provided course material."

Course Material:
====================
{context}
====================

User Question:
{question}

Answer:
"""


    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "gemma3:1b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# =========================================
# TEST QUESTIONS
# =========================================

questions = [
    "What is HTML?",
    "What is the basic structure of an HTML website?",
    "What are semantic tags in HTML?",
    "What is CSS?",
    "What are CSS selectors?",
    "What is the CSS box model?",
    "What is Python Django?",
    "What is React?",
    "What is machine learning?",
]


# =========================================
# RUN TEST
# =========================================

print("=" * 70)
print("SIGMA RAG ANSWER EVALUATION")
print("=" * 70)


for i, question in enumerate(questions, start=1):

    print("\n" + "=" * 70)
    print(f"QUESTION {i}")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    try:

        answer = generate_answer(question)

        print("\nAnswer:")
        print(answer)

    except Exception as e:

        print("\nERROR:")
        print(e)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)