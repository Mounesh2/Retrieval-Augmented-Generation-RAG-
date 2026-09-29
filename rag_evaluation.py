import numpy as np
import joblib
import requests
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
# RETRIEVE DOCUMENTS
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

    return top_chunks


# =========================================
# TEST QUESTIONS
# =========================================

course_questions = [
    "What is HTML?",
    "What is the basic structure of an HTML website?",
    "What are headings in HTML?",
    "How are images added in HTML?",
    "What are semantic tags in HTML?",
    "What is CSS?",
    "What are inline, internal and external CSS?",
    "What are CSS selectors?",
    "What is the CSS box model?",
    "What is the difference between margin and padding?",
]


out_of_course_questions = [
    "What is Python Django?",
    "What is React?",
    "What is machine learning?",
    "What is MongoDB?",
    "What is artificial intelligence?",
]


# =========================================
# THRESHOLD
# =========================================

similarity_threshold = 0.60


# =========================================
# RUN EVALUATION
# =========================================

print("=" * 70)
print("SIGMA RAG RETRIEVAL EVALUATION")
print("=" * 70)


# =========================================
# COURSE QUESTIONS
# =========================================

print("\nCOURSE QUESTIONS")
print("=" * 70)

course_pass = 0

for i, question in enumerate(course_questions, start=1):

    results = retrieve(question)

    highest_similarity = results["similarity"].max()

    passed = highest_similarity >= similarity_threshold

    if passed:
        course_pass += 1

    status = "PASS" if passed else "FAIL"

    print(f"\n{i}. {question}")
    print(f"Highest similarity: {highest_similarity:.4f}")
    print(f"Status: {status}")


# =========================================
# OUT-OF-COURSE QUESTIONS
# =========================================

print("\n\nOUT-OF-COURSE QUESTIONS")
print("=" * 70)

out_course_pass = 0

for i, question in enumerate(
    out_of_course_questions,
    start=1
):

    results = retrieve(question)

    highest_similarity = results["similarity"].max()

    rejected = highest_similarity < similarity_threshold

    if rejected:
        out_course_pass += 1

    status = "PASS" if rejected else "FAIL"

    print(f"\n{i}. {question}")
    print(f"Highest similarity: {highest_similarity:.4f}")
    print(f"Status: {status}")


# =========================================
# FINAL RESULTS
# =========================================

total_course = len(course_questions)
total_out_course = len(out_of_course_questions)

total_pass = course_pass + out_course_pass
total_questions = total_course + total_out_course

accuracy = (
    total_pass / total_questions
) * 100


print("\n\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

print(
    f"\nCourse questions: "
    f"{course_pass}/{total_course} passed"
)

print(
    f"Out-of-course questions: "
    f"{out_course_pass}/{total_out_course} passed"
)

print(
    f"\nOverall evaluation: "
    f"{total_pass}/{total_questions}"
)

print(
    f"Evaluation score: "
    f"{accuracy:.2f}%"
)

print("=" * 70)