def create_prompt(context, question):

    prompt = f"""
You are an AI assistant for the Sigma Web Development course.

Answer the user's question using only the information provided in the context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided course material."

Do not make up information.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""

    return prompt


# Example
context = """
HTML is the standard language used to create the basic structure
of a website. HTML uses tags to define different parts of a webpage.
"""

question = "What is HTML?"

prompt = create_prompt(context, question)

print(prompt)