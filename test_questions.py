# =========================================
# RAG TEST QUESTIONS
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


print("=" * 60)
print("SIGMA RAG EVALUATION")
print("=" * 60)

print("\nCOURSE QUESTIONS")
print("-" * 60)

for i, question in enumerate(course_questions, start=1):
    print(f"{i}. {question}")


print("\nOUT-OF-COURSE QUESTIONS")
print("-" * 60)

for i, question in enumerate(out_of_course_questions, start=1):
    print(f"{i}. {question}")


print("\n" + "=" * 60)
print(
    "TOTAL QUESTIONS:",
    len(course_questions) + len(out_of_course_questions)
)
print("=" * 60)