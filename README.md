# Document RAG Multi-Chat Assistant

This is a Streamlit app where you can create multiple chats, upload documents per chat, and ask grounded questions using RAG.

Each chat has:
- its own message history
- its own system prompt
- its own Chroma vector store

So documents from one chat do not leak into another.

## What you can do
- Create a new chat (with a custom name)
- Switch between chats
- Rename or delete a chat
- Upload docs (`.txt`, `.md`, `.csv`, `.pdf`, `.docx`)
- Ingest docs into that chat's vector DB
- Ask questions in chat UI (`st.chat_input`)
- See retrieved chunks used as evidence
- Choose RAG behavior:
  - `Auto decide`
  - `Always refer documents`
  - `Do not refer documents`

## Tech stack
- Streamlit
- OpenAI API
- LangChain
- LangGraph
- ChromaDB
- PostgreSQL + SQLAlchemy
- python-dotenv

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/rag_app
```

Optional local fallback (if you want SQLite while developing):

```env
DATABASE_URL=sqlite:///./app.db
```

## Run
```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## Project structure
```text
RAG/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── app.db            # optional SQLite fallback DB
├── chroma_db/        # per-chat vector stores
└── README.md
```

## Architecture
```mermaid
flowchart TD
    A[User creates or selects chat] --> B[Load chat state from SQL DB]
    B --> C[Load chat-specific Chroma DB]

    D[User uploads files] --> E[Parse and chunk documents]
    E --> F[Embed chunks]
    F --> G[Store in Chroma for active chat]
    G --> H[Store doc metadata in SQL DB]

    I[User asks question] --> J[Single agent reads history + mode + file names]
    J --> K{RAG mode decision}
    K -->|Always| L[Retrieve chunks]
    K -->|Never| M[Skip retrieval]
    K -->|Auto| N[Agent decides yes/no]
    N -->|Yes| L
    N -->|No| M

    L --> O[Generate grounded answer]
    M --> P[Generate direct answer or I do not know]
    O --> Q[Show answer + evidence]
    P --> Q
    Q --> R[Persist message + evidence in SQL DB]
```

## Notes
- Duplicate uploads are skipped per chat using file-content hash.
- Deleting a chat also deletes that chat's vector DB folder.
