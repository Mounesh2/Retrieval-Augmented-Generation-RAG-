import requests


def get_llm_response(prompt):

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


# -----------------------------------------
# Test the LLM
# -----------------------------------------

prompt = """
You are an AI assistant for the Sigma Web Development course.

Answer the question using the information provided below.

Context:
HTML is the standard language used to create the basic
structure of a website.

Question:
What is HTML?

Answer:
"""

answer = get_llm_response(prompt)

print("\nLLM RESPONSE:")
print(answer)