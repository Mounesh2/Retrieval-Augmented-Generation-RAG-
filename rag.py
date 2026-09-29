import streamlit as st
import numpy as np
import joblib
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Sigma AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================
# LOAD EMBEDDINGS
# =========================================

@st.cache_resource
def load_embeddings():
    return joblib.load("chunks/embeddings.joblib")


df = load_embeddings()


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
# STREAM LLM RESPONSE
# =========================================

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
            "stream": True
        },
        stream=True,
        timeout=600
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        data = json.loads(line.decode("utf-8"))

        if "message" in data:

            content = data["message"].get("content", "")

            if content:
                yield content


# =========================================
# RAG
# =========================================

def ask_rag(question):

    # -------------------------------------
    # Create question embedding
    # -------------------------------------

    question_embedding = create_embedding(question)


    # -------------------------------------
    # Calculate similarity
    # -------------------------------------

    similarities = cosine_similarity(
        np.vstack(df["embedding"].values),
        [question_embedding]
    ).flatten()


    # -------------------------------------
    # Get top results
    # -------------------------------------

    top_results = 10

    max_index = similarities.argsort()[::-1][:top_results]

    top_chunks = df.iloc[max_index].copy()

    top_chunks["similarity"] = similarities[max_index]


    # -------------------------------------
    # Create context
    # -------------------------------------

    context = ""

    for _, item in top_chunks.iterrows():

        context += f"""
{item["text"]}

--------------------
"""


    # -------------------------------------
    # Create RAG prompt
    # -------------------------------------

    prompt = f"""
You are an AI assistant for the Sigma Web Development course.

Answer the user's question using ONLY the course material provided below.

Understand the meaning of the question even if the wording is different.

Give a clear and concise answer.

Do not mention videos, timestamps, chunks, similarity scores, retrieval, or the context.

Do not make up information.

If the answer cannot be found in the course material, say:

"I could not find the answer in the provided course material."

Course Material:
====================
{context}
====================

User Question:
{question}

Answer:
"""

    return prompt, top_chunks


# =========================================
# SESSION STATE
# =========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.title("🤖 Sigma AI")

    st.write(
        "Sigma Web Development RAG Assistant"
    )

    st.divider()

    st.write(
        f"📚 {len(df)} course chunks"
    )

    st.write("🎥 18 course videos")

    st.write("🧠 BGE-M3")

    st.write("🤖 Gemma 3 1B")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================
# HEADER
# =========================================

st.title(
    "🤖 Sigma Web Development AI Assistant"
)

st.caption(
    "Ask questions about the Sigma Web Development course."
)


# =========================================
# DISPLAY PREVIOUS CHAT
# =========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander("📚 View Sources"):

                for rank, (_, item) in enumerate(
                    message["sources"].iterrows(),
                    start=1
                ):

                    st.write(
                        f"**#{rank} — Video "
                        f"{item['video']}**"
                    )

                    st.write(
                        f"Similarity: "
                        f"{item['similarity']:.4f}"
                    )

                    st.write(
                        f"Time: "
                        f"{item['start']:.2f}s - "
                        f"{item['end']:.2f}s"
                    )

                    st.info(item["text"])

                    st.divider()


# =========================================
# CHAT INPUT
# =========================================

question = st.chat_input(
    "Ask something about the course..."
)


# =========================================
# PROCESS QUESTION
# =========================================

if question:

    # -------------------------------------
    # Save user message
    # -------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -------------------------------------
    # Display user message
    # -------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------------------
    # Generate assistant response
    # -------------------------------------

    with st.chat_message("assistant"):

        try:

            # =============================
            # SEARCH COURSE MATERIAL
            # =============================

            with st.spinner(
                "🔎 Searching course material..."
            ):

                prompt, sources = ask_rag(
                    question
                )


            # =============================
            # STREAM GEMMA RESPONSE
            # =============================

            answer_placeholder = st.empty()

            full_answer = ""

            for token in get_llm_response(prompt):

                full_answer += token

                answer_placeholder.markdown(
                    full_answer
                )


            # =============================
            # SAVE ASSISTANT MESSAGE
            # =============================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources
                }
            )


            # =============================
            # SHOW SOURCES
            # =============================

            with st.expander(
                "📚 View Sources"
            ):

                for rank, (_, item) in enumerate(
                    sources.iterrows(),
                    start=1
                ):

                    st.write(
                        f"**#{rank} — Video "
                        f"{item['video']}**"
                    )

                    st.write(
                        f"Similarity: "
                        f"{item['similarity']:.4f}"
                    )

                    st.write(
                        f"Time: "
                        f"{item['start']:.2f}s - "
                        f"{item['end']:.2f}s"
                    )

                    st.info(item["text"])

                    st.divider()


        # =================================
        # ERROR HANDLING
        # =================================

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to Ollama. "
                "Make sure Ollama is running."
            )


        except requests.exceptions.Timeout:

            st.error(
                "⏱️ Ollama took too long to respond."
            )


        except Exception as e:

            st.error(
                f"❌ Error: {e}"
            )