# 🤖 Sigma AI — Video RAG Assistant

> An AI-powered Retrieval-Augmented Generation (RAG) assistant that answers questions from CodeWithHarry's Sigma Web Development course videos using local AI models.

---

## 📌 Overview

**Sigma AI** is a local Video RAG application that allows users to ask questions about web-development course videos and receive answers based only on the available course material.

The system processes course videos using **Whisper**, converts transcript chunks into vector embeddings using **BGE-M3**, retrieves relevant content using **cosine similarity**, and generates answers using the locally running **Gemma 3 1B** model through **Ollama**.

The application provides a **ChatGPT-style Streamlit interface** with:

- 💬 AI-generated answers
- 🔎 Semantic search
- 📚 Retrieved course sources
- 🎥 Video information
- ⏱️ Source timestamps
- ⚡ Streaming responses
- 🛡️ Out-of-course question protection

---

# ✨ Features

- 🎥 Process course videos into searchable knowledge
- 🎙️ Automatic speech-to-text transcription using Whisper
- ✂️ Transcript chunking with overlapping chunks
- 🧠 BGE-M3 semantic embeddings
- 🔎 Cosine-similarity based retrieval
- 📊 Similarity threshold filtering
- 🤖 Local Gemma 3 1B LLM
- 🦙 Ollama local AI runtime
- 💬 ChatGPT-style Streamlit UI
- ⚡ Streaming AI responses
- 📚 Source and timestamp display
- 🛡️ Prevents answers from general knowledge when information is not available
- 🧪 Retrieval and answer evaluation scripts
- 🔐 Runs locally without requiring a cloud LLM API

---

# 🎯 Project Goal

The main goal of this project is to build a question-answering system that can understand and retrieve information from educational videos.

Instead of asking an LLM to answer from its general knowledge, the system first searches the course material and then provides the retrieved information to the LLM.

This helps the application stay focused on the selected course content.

---

# 🏗️ Project Architecture

```text
                         ┌─────────────────────┐
                         │    Course Videos    │
                         │      18 Videos      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Audio Extraction   │
                         │        MP3          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Whisper        │
                         │   Speech-to-Text    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Transcripts     │
                         │       JSON          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Text Chunking     │
                         │ 120 words / 20      │
                         │ word overlap        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       BGE-M3        │
                         │    Embeddings       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Embedding Storage   │
                         │ embeddings.joblib   │
                         └──────────┬──────────┘
                                    │
                                    │
              ┌─────────────────────┘
              │
              ▼
       ┌───────────────────┐
       │   User Question   │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │      BGE-M3       │
       │ Query Embedding   │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Cosine Similarity │
       │    Retrieval      │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Similarity Filter │
       │      ≥ 0.60       │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Relevant Chunks   │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │     RAG Prompt    │
       │ Question + Data   │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │     Gemma 3 1B    │
       │     Local LLM     │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Streaming Answer  │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │    Streamlit UI   │
       │                   │
       │ 💬 Answer         │
       │ 📚 Sources        │
       │ 🎥 Video          │
       │ ⏱️ Timestamp      │
       └───────────────────┘
```

---

# 🔄 Complete RAG Pipeline

The project has two major stages.

## 1. Offline Knowledge Preparation

```text
Course Videos
      ↓
Audio
      ↓
Whisper Transcription
      ↓
Transcript JSON
      ↓
Text Chunking
      ↓
BGE-M3 Embeddings
      ↓
embeddings.joblib
```

This stage converts the course videos into a searchable knowledge base.

---

## 2. Online Question Answering

```text
User Question
      ↓
BGE-M3 Query Embedding
      ↓
Cosine Similarity
      ↓
Top Relevant Chunks
      ↓
Similarity Threshold
      ↓
RAG Context
      ↓
Gemma 3 1B
      ↓
Streaming Response
      ↓
Streamlit UI
```

---

# 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming Language | Python 3.12 | Main development language |
| UI | Streamlit | Web-based chat interface |
| Speech-to-Text | Whisper | Video transcription |
| Embedding Model | BGE-M3 | Semantic embeddings |
| LLM | Gemma 3 1B | Answer generation |
| LLM Runtime | Ollama | Local model execution |
| Retrieval | Cosine Similarity | Semantic search |
| Numerical Processing | NumPy | Vector operations |
| Data Processing | Pandas | Data handling |
| ML Utilities | Scikit-learn | Similarity calculation |
| Storage | JSON | Transcript/chunk storage |
| Storage | Joblib | Embedding storage |
| HTTP Client | Requests | Ollama API communication |
| Environment | Anaconda | Python environment |
| Version Control | Git | Source control |
| Repository | GitHub | Project hosting |

---

# 📊 Project Dataset

The project currently processes:

- 🎥 **18 course videos**
- 📝 **18 transcripts**
- 🧩 **422 text chunks**
- 🧠 **1024-dimensional BGE-M3 embeddings**
- 🔎 **Cosine similarity retrieval**
- 🎯 **Similarity threshold: 0.60**

### Chunking Strategy

```text
Chunk size : 120 words
Overlap    : 20 words
```

The overlap helps preserve context between neighboring chunks.

---

# 🧩 Project Structure

```text
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
│   ├── all_chunks.json
│   └── embeddings.joblib
│
├── sst.py
│   └── Video/audio transcription
│
├── create_chunks.py
│   └── Transcript chunking
│
├── read_chunks.py
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
├── test_questions.py
│   └── Evaluation questions
│
├── rag_evaluation.py
│   └── Retrieval evaluation
│
├── answer_evaluation.py
│   └── Answer evaluation
│
└── README.md
```

---

# 🧠 How RAG Works

RAG stands for:

> **Retrieval-Augmented Generation**

Instead of directly asking the LLM:

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

this project uses:

```text
User Question
      ↓
Retrieve Relevant Course Information
      ↓
Add Retrieved Information to Prompt
      ↓
LLM
      ↓
Answer
```

This allows the model to use the course material as its knowledge source.

---

# 🔎 Retrieval Process

When the user asks a question, **BGE-M3** converts the question into an embedding.

The system compares this embedding against the stored course embeddings.

```text
Question
   ↓
BGE-M3
   ↓
Query Vector
   ↓
Compare with Course Vectors
   ↓
Cosine Similarity
   ↓
Rank Results
```

The system retrieves the most relevant chunks and applies a similarity threshold:

```text
Similarity Threshold = 0.60
```

Only chunks meeting the threshold are passed to the generation stage.

---

# 🤖 Answer Generation

The retrieved course content is placed inside a controlled RAG prompt.

The model is instructed to:

- Use only the provided course material
- Avoid general knowledge
- Avoid inventing information
- Give short and clear answers
- Avoid unnecessary repetition
- Return a fallback message when the information is unavailable

Example fallback:

```text
I could not find the answer in the provided course material.
```

---

# 💬 Example

### Question

```text
What are semantic tags in HTML?
```

### RAG Process

```text
Question
   ↓
BGE-M3
   ↓
Semantic Search
   ↓
Relevant HTML Chunks
   ↓
RAG Prompt
   ↓
Gemma 3 1B
   ↓
Answer
```

The application also displays retrieved source information such as:

- 🎥 Video
- 📊 Similarity
- ⏱️ Timestamp
- 📝 Transcript

---

# 🚀 How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mounesh2/Retrieval-Augmented-Generation-RAG-.git
```

Go to the project directory:

```bash
cd Retrieval-Augmented-Generation-RAG-
```

---

## 2️⃣ Create the Python Environment

Create a Conda environment:

```bash
conda create -n sigma-rag python=3.12
```

Activate it:

```bash
conda activate sigma-rag
```

Check Python:

```bash
python --version
```

---

## 3️⃣ Install Dependencies

```bash
pip install streamlit numpy pandas scikit-learn joblib requests openai-whisper
```

---

# 🦙 Ollama Setup

Install Ollama and verify the installation:

```bash
ollama --version
```

Download the embedding model:

```bash
ollama pull bge-m3
```

Download the language model:

```bash
ollama pull gemma3:1b
```

Check installed models:

```bash
ollama list
```

Expected models:

```text
bge-m3
gemma3:1b
```

---

# ▶️ Start the Application

Make sure Ollama is running.

Then run:

```bash
python -m streamlit run app.py
```

Open the Streamlit application:

```text
http://localhost:8501
```

---

# 💡 Example Questions

Try questions related to the course:

```text
What is HTML?
```

```text
What is the basic structure of an HTML website?
```

```text
What are semantic tags in HTML?
```

```text
What is CSS?
```

```text
What are CSS selectors?
```

```text
What is the CSS box model?
```

---

# 🧪 Evaluation

The project contains separate evaluation scripts.

## Retrieval Evaluation

Run:

```bash
python rag_evaluation.py
```

The evaluation checks:

- Course-related questions
- Out-of-course questions
- Retrieval similarity
- Similarity threshold behavior

---

## 📝 Answer Evaluation

Run:

```bash
python answer_evaluation.py
```

This evaluates the complete pipeline:

```text
Question
   ↓
Embedding
   ↓
Retrieval
   ↓
Threshold
   ↓
RAG Prompt
   ↓
Gemma
   ↓
Answer
```

---

# 🔍 Retrieval Inspection

To inspect which chunks are retrieved for a question:

```bash
python introspect.py
```

Example:

```text
What are CSS selectors?
```

The tool displays:

- Rank
- Video
- Similarity
- Start Time
- End Time
- Transcript

This helps inspect the internal retrieval process.

---

# 🔄 Rebuild the Knowledge Base

If new videos are added, run the processing pipeline again.

### Step 1 — Transcription

```bash
python sst.py
```

### Step 2 — Create Chunks

```bash
python create_chunks.py
```

### Step 3 — Generate Embeddings

```bash
python read_chunks.py
```

Then start the application:

```bash
python -m streamlit run app.py
```

---

# 📁 Important Generated Files

### Transcripts

```text
transcripts/
```

Contains Whisper-generated transcript files.

### Chunks

```text
chunks/all_chunks.json
```

Contains the processed text chunks.

### Embeddings

```text
chunks/embeddings.joblib
```

Contains BGE-M3 embeddings used during retrieval.

---

# ⚠️ Troubleshooting

## Ollama Connection Error

Check whether Ollama is available:

```bash
ollama list
```

Make sure Ollama is running.

---

## Model Not Found

Run:

```bash
ollama pull bge-m3
ollama pull gemma3:1b
```

---

## Streamlit Not Found

Install Streamlit:

```bash
pip install streamlit
```

Then run:

```bash
python -m streamlit run app.py
```

---

## Embeddings File Not Found

Make sure this file exists:

```text
chunks/embeddings.joblib
```

If it does not exist, generate the embeddings:

```bash
python read_chunks.py
```

---

# ⚡ Quick Start

If everything is already installed:

```bash
conda activate sigma-rag
```

```bash
ollama list
```

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🔐 Local AI Architecture

The current system runs the AI components locally:

```text
User
 ↓
Streamlit
 ↓
BGE-M3
 ↓
Local Retrieval
 ↓
Ollama
 ↓
Gemma 3 1B
 ↓
Answer
```

The current implementation does not require a cloud LLM API.

---

# 📈 Current Pipeline Summary

```text
18 Videos
     ↓
Whisper
     ↓
18 Transcripts
     ↓
120-word Chunks
     ↓
20-word Overlap
     ↓
422 Chunks
     ↓
BGE-M3
     ↓
1024-D Embeddings
     ↓
Cosine Similarity
     ↓
Top 10 Results
     ↓
Similarity ≥ 0.60
     ↓
Relevant Context
     ↓
Gemma 3 1B
     ↓
Streaming Response
     ↓
Streamlit Chat UI
```

---

# 🔮 Future Improvements

Possible future improvements:

- 🔎 Better retrieval and reranking
- 🧠 Larger local LLM support
- 📚 Support for multiple courses
- 🎥 Direct video playback from source timestamps
- 🔊 Voice-based questions
- 🗂️ Multiple knowledge bases
- 👤 User authentication
- 📊 Advanced evaluation metrics
- ☁️ Optional cloud deployment
- ⚡ Retrieval and generation optimization

---

# 🎓 Learning Outcomes

Through this project, the following concepts were implemented:

- Retrieval-Augmented Generation
- Vector embeddings
- Semantic search
- Cosine similarity
- Text chunking
- Speech-to-text
- Prompt engineering
- Local LLM inference
- Ollama
- Streamlit
- Python API integration
- RAG evaluation
- AI application development

---

# 👨‍💻 Author

**Mounesh Pattar**

Computer Science & Engineering

---

# ⭐ Project Summary

```text
🎥 Video Processing
        +
🎙️ Speech-to-Text
        +
🧩 Text Chunking
        +
🧠 Embeddings
        +
🔎 Semantic Retrieval
        +
🤖 Local LLM
        +
💬 Streamlit
        =
🚀 Video RAG Assistant
```

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.
