# Knowledge Assistant (Agentic RAG POC)

A **multi-agent RAG (Retrieval-Augmented Generation)** prototype that builds an internal knowledge assistant using:

-  **Local embeddings** (Sentence Transformers)
-  **FAISS vector search**
-  **GPT-4.1-mini reasoning**
-  **Multi-agent orchestration** (Planner → Retriever → Answerer → Critic)
-  **Markdown/Text document ingestion**

This project demonstrates planning, retrieval, reasoning, critique, and feedback in a clean, modular architecture.

---

##  Project Structure

```
knowledge-assistant/
│
├── data/
│   └── docs/                  # .md or .txt data files here
│
├── src/
│   ├── config.py              # API keys, model configuration
│   ├── ingestion.py           # Document loading + chunking
│   ├── vector_store.py        # Embeddings + FAISS storage/search
│   ├── agents.py              # Planner, Retriever, Answerer, Critic, Feedback agents
│   └── app.py                 # Main orchestrator (CLI)
│
└── README.md
```

---

##  Requirements

- Python 3.9+
- OpenAI API Key
- Local installation of SentenceTransformers + FAISS

Install dependencies:

```bash
pip install -r requirements.txt
```

Add `.env` file:

```
OPENAI_API_KEY=api_key_here
```

---

##  How It Works 

The system implements a **multi-agent RAG pipeline**:

### ** Ingestion**
- Reads all `.md` and `.txt` documents.
- Splits them into overlapping text chunks.

### ** Vector Store**
- Embeds chunks using `all-MiniLM-L6-v2`.
- Stores vectors in a **FAISS L2 index** for similarity search.

### ** Planner Agent**
- Decomposes a complex question into 2–5 sub-questions.

### ** Retrieval Agent**
- Finds top-K relevant chunks from FAISS.

### ** Answer Agent**
- Uses OpenAI GPT-4.1-mini.
- Generates grounded answers using only retrieved sources.
- Tailors tone to selected role (developer, manager, support) and task (debugging, onboarding, planning).

### ** Critic Agent**
- Evaluates the answer.
- Labels it `[OK]`, `[PARTIAL]`, or `[RISKY]`.

### ** Feedback Agent**
- Collects optional human feedback (yes/no + comment).

---

##  Running the App

```bash
cd src
python app.py
```

You’ll be asked:

```
Choose your role: [developer / manager / support]
Choose your task: [debugging / planning / onboarding]
```

Then enter questions, for example:

```
How do I onboard a new backend engineer?
```

---

##  Example Interaction

**Planner Agent output:**

```
1. Identify tools required for onboarding.
2. Summarize onboarding policy.
3. Outline first-week developer setup steps.
```

**Retriever Agent output:**
```
Retrieved 5 chunks.
```

**Critic Agent output:**
```
[PARTIAL] The answer is mostly grounded, but misses policy details from doc 3.
```

---

##  Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

Models used:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Chat model: `gpt-4.1-mini`

---

##  Customization

**Change embedding model** (src/config.py):

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

**Change chat model:**

```python
CHAT_MODEL = "gpt-4.1-mini"
```

**Add data documents:**

Put files into:

```
data/docs/
```

---

##  Future Enhancements

We can extend the assistant with:

- FastAPI HTTP API instead of CLI
- Docker support
- User feedback logging + analytics
- Replace FAISS FlatL2 with HNSW
- Add citation markers in the generated answer
- Support for PDFs, HTML, Notion export, etc.
- Long-term memory module


---


##  License

MIT License.
