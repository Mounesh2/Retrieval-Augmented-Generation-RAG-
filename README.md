\# 🤖 Sigma AI — Video RAG Assistant



> An AI-powered Retrieval-Augmented Generation (RAG) assistant that answers questions from CodeWithHarry's Sigma Web Development course videos using local AI models.



\---



\## 📌 Overview



\*\*Sigma AI\*\* is a local Video RAG application that allows users to ask questions about web-development course videos and receive answers based only on the available course material.



The system processes course videos using \*\*Whisper\*\*, converts transcript chunks into vector embeddings using \*\*BGE-M3\*\*, retrieves relevant content using \*\*cosine similarity\*\*, and generates answers using the locally running \*\*Gemma 3 1B\*\* model through \*\*Ollama\*\*.



The application provides a \*\*ChatGPT-style Streamlit interface\*\* with:



\- 💬 AI-generated answers

\- 🔎 Semantic search

\- 📚 Retrieved course sources

\- 🎥 Video information

\- ⏱️ Source timestamps

\- ⚡ Streaming responses

\- 🛡️ Out-of-course question protection



\---



\# ✨ Features



\- 🎥 Process course videos into searchable knowledge

\- 🎙️ Automatic speech-to-text transcription using Whisper

\- ✂️ Transcript chunking with overlapping chunks

\- 🧠 BGE-M3 semantic embeddings

\- 🔎 Cosine-similarity based retrieval

\- 📊 Similarity threshold filtering

\- 🤖 Local Gemma 3 1B LLM

\- 🦙 Ollama local AI runtime

\- 💬 ChatGPT-style Streamlit UI

\- ⚡ Streaming AI responses

\- 📚 Source and timestamp display

\- 🛡️ Prevents answers from general knowledge when information is not available

\- 🧪 Retrieval and answer evaluation scripts

\- 🔐 Runs locally without requiring a cloud LLM API



\---



\# 🎯 Project Goal



The main goal of this project is to build a question-answering system that can understand and retrieve information from educational videos.



Instead of asking an LLM to answer from its general knowledge, the system first searches the course material and then provides the retrieved information to the LLM.



This helps the application stay focused on the selected course content.



\---



\# 🏗️ Project Architecture



```text

&#x20;                        ┌─────────────────────┐

&#x20;                        │    Course Videos    │

&#x20;                        │      18 Videos      │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │  Audio Extraction   │

&#x20;                        │        MP3          │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │      Whisper        │

&#x20;                        │   Speech-to-Text    │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │     Transcripts     │

&#x20;                        │       JSON          │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │   Text Chunking     │

&#x20;                        │ 120 words / 20      │

&#x20;                        │ word overlap        │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │       BGE-M3        │

&#x20;                        │    Embeddings       │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │ Embedding Storage   │

&#x20;                        │ embeddings.joblib   │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   │

&#x20;             ┌─────────────────────┘

&#x20;             │

&#x20;             ▼

&#x20;      ┌───────────────────┐

&#x20;      │   User Question   │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │      BGE-M3       │

&#x20;      │ Query Embedding   │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │ Cosine Similarity │

&#x20;      │    Retrieval      │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │ Similarity Filter │

&#x20;      │      ≥ 0.60       │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │ Relevant Chunks   │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │     RAG Prompt    │

&#x20;      │ Question + Data  │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │     Gemma 3 1B    │

&#x20;      │     Local LLM     │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │ Streaming Answer  │

&#x20;      └─────────┬─────────┘

&#x20;                │

&#x20;                ▼

&#x20;      ┌───────────────────┐

&#x20;      │    Streamlit UI   │

&#x20;      │                   │

&#x20;      │ 💬 Answer         │

&#x20;      │ 📚 Sources        │

&#x20;      │ 🎥 Video          │

&#x20;      │ ⏱️ Timestamp      │

&#x20;      └───────────────────┘

🔄 Complete RAG Pipeline



The project has two major stages.



1\. Offline Knowledge Preparation



Course Videos

&#x20;     ↓

Audio

&#x20;     ↓

Whisper Transcription

&#x20;     ↓

Transcript JSON

&#x20;     ↓

Text Chunking

&#x20;     ↓

BGE-M3 Embeddings

&#x20;     ↓

embeddings.joblib



2\. Online Question Answering

User Question

&#x20;     ↓

BGE-M3 Query Embedding

&#x20;     ↓

Cosine Similarity

&#x20;     ↓

Top Relevant Chunks

&#x20;     ↓

Similarity Threshold

&#x20;     ↓

RAG Context

&#x20;     ↓

Gemma 3 1B

&#x20;     ↓

Streaming Response

&#x20;     ↓

Streamlit UI



🛠️ Tech Stack

| Layer                | Technology        | Purpose                   |

| -------------------- | ----------------- | ------------------------- |

| Programming Language | Python 3.12       | Main development language |

| UI                   | Streamlit         | Web-based chat interface  |

| Speech-to-Text       | Whisper           | Video transcription       |

| Embedding Model      | BGE-M3            | Semantic embeddings       |

| LLM                  | Gemma 3 1B        | Answer generation         |

| LLM Runtime          | Ollama            | Local model execution     |

| Retrieval            | Cosine Similarity | Semantic search           |

| Numerical Processing | NumPy             | Vector operations         |

| Data Processing      | Pandas            | Data handling             |

| ML Utilities         | Scikit-learn      | Similarity calculation    |

| Storage              | JSON              | Transcript/chunk storage  |

| Storage              | Joblib            | Embedding storage         |

| HTTP Client          | Requests          | Ollama API communication  |

| Environment          | Anaconda          | Python environment        |

| Version Control      | Git               | Source control            |

| Repository           | GitHub            | Project hosting           |



📊 Project Dataset



The project currently processes:



🎥 18 course videos

📝 18 transcripts

🧩 422 text chunks

🧠 1024-dimensional BGE-M3 embeddings

🔎 Cosine similarity retrieval

🎯 Similarity threshold: 0.60



🧩 Project Structure

RAG Project/

│

├── videos/

│   └── Course video files

│

├── audios/

│   └── Extracted audio files

│

├── transcripts/

│   └── Whisper transcript JSON files

│

├── chunks/

│   ├── all\_chunks.json

│   └── embeddings.joblib

│

├── sst.py

│   └── Video/audio transcription

│

├── create\_chunks.py

│   └── Transcript chunking

│

├── read\_chunks.py

│   └── BGE-M3 embedding generation

│

├── introspect.py

│   └── Retrieval inspection

│

├── prompt.py

│   └── Prompt testing

│

├── llm.py

│   └── Local LLM testing

│

├── rag.py

│   └── Command-line RAG pipeline

│

├── app.py

│   └── Streamlit application

│

├── test\_questions.py

│   └── Evaluation questions

│

├── rag\_evaluation.py

│   └── Retrieval evaluation

│

├── answer\_evaluation.py

│   └── Answer evaluation

│

└── README.md







