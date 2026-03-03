import os
import io
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from docx import Document as DocxDocument
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


# ------------------------------
# Configuration
# ------------------------------
load_dotenv(dotenv_path=Path(".env"), override=False)

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "uploaded_docs"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. If RAG context is provided, prioritize it. "
    "If sufficient context is missing, say you don't know. "
    "Do not guess or fabricate facts. "
    "Keep answers concise and accurate."
)


# ------------------------------
# Helpers: parsing and ingestion
# ------------------------------
def parse_file(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".csv"}:
        return file_bytes.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if suffix == ".docx":
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")


def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def is_hash_indexed(vectorstore: Chroma, content_hash: str) -> bool:
    matches = vectorstore._collection.get(
        where={"content_hash": content_hash},
        include=["metadatas"],
        limit=1,
    )
    return bool(matches.get("ids"))


def ingest_documents(vectorstore: Chroma, uploaded_files) -> Dict[str, int]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    ingested_doc_ids = []
    skipped_duplicates = 0

    for file in uploaded_files:
        file_bytes = file.getvalue()
        content_hash = file_hash(file_bytes)
        if is_hash_indexed(vectorstore, content_hash):
            skipped_duplicates += 1
            continue

        text = parse_file(file_bytes, file.name)
        if not text.strip():
            continue

        doc_id = str(uuid.uuid4())
        chunks = splitter.split_text(text)

        docs = []
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-{i}"
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "content_hash": content_hash,
                        "source": file.name,
                        "chunk_index": i,
                    },
                )
            )
            ids.append(chunk_id)

        vectorstore.add_documents(docs, ids=ids)
        ingested_doc_ids.append(doc_id)

    return {"ingested_count": len(ingested_doc_ids), "skipped_duplicates": skipped_duplicates}


def list_uploaded_documents(vectorstore: Chroma) -> Dict[str, str]:
    data = vectorstore.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    docs = {}
    for md in metadatas:
        if not md:
            continue
        doc_id = md.get("doc_id")
        source = md.get("source", "unknown")
        if doc_id and doc_id not in docs:
            docs[doc_id] = source

    return docs


def delete_document(vectorstore: Chroma, doc_id: str):
    vectorstore._collection.delete(where={"doc_id": doc_id})


def recent_chat_context(messages: List[Dict[str, Any]], limit: int = 6) -> str:
    lines = []
    for msg in messages[-limit:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def docs_to_evidence(docs: List[Document]) -> List[Dict[str, Any]]:
    evidence = []
    for d in docs:
        evidence.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "chunk_index": d.metadata.get("chunk_index", "?"),
                "content": d.page_content,
            }
        )
    return evidence


# ------------------------------
# LangGraph single-agent flow
# ------------------------------
class GraphState(TypedDict):
    question: str
    system_prompt: str
    chat_history: List[Dict[str, Any]]
    rag_mode: str
    docs_available: bool
    available_sources: List[str]
    use_rag: bool
    retrieved_docs: List[Document]
    answer: str


def build_graph(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    def agent(state: GraphState):
        use_rag = False
        question = state["question"]
        question_lc = question.lower()
        rag_mode = state["rag_mode"]
        available_sources = state.get("available_sources", [])
        retrieved_docs: List[Document] = []

        # Deterministic handling for "what files/documents are uploaded?" requests.
        file_list_intents = ["what documents", "which documents", "what files", "which files", "uploaded docs", "uploaded files"]
        if any(intent in question_lc for intent in file_list_intents):
            if available_sources:
                listed = "\n".join([f"- {name}" for name in available_sources])
                return {
                    "answer": f"I can see these uploaded documents:\n{listed}",
                    "retrieved_docs": [],
                    "use_rag": False,
                }
            return {
                "answer": "No uploaded documents are currently indexed.",
                "retrieved_docs": [],
                "use_rag": False,
            }

        if state["docs_available"]:
            if rag_mode == "always":
                use_rag = True
            elif rag_mode == "never":
                use_rag = False
            elif any(src.lower() in question_lc for src in available_sources if src):
                use_rag = True
            else:
                history = recent_chat_context(state.get("chat_history", []))
                sources_text = ", ".join(available_sources) if available_sources else "None"
                decision_prompt = (
                    "Decide whether retrieval from uploaded docs is necessary for the user's question. "
                    "Reply with only YES or NO.\n\n"
                    f"Uploaded file names:\n{sources_text}\n\n"
                    f"Recent chat:\n{history}\n\n"
                    f"Question:\n{state['question']}"
                )
                decision = llm.invoke(
                    [
                        {"role": "system", "content": "Return only YES or NO."},
                        {"role": "user", "content": decision_prompt},
                    ]
                )
                use_rag = str(decision.content).strip().upper().startswith("Y")

        if use_rag:
            retrieved_docs = retriever.invoke(state["question"])

        history = recent_chat_context(state.get("chat_history", []))
        if use_rag and retrieved_docs:
            context = "\n\n".join(
                [
                    (
                        f"[Source: {d.metadata.get('source', 'unknown')} | "
                        f"Chunk: {d.metadata.get('chunk_index', '?')}]\n{d.page_content}"
                    )
                    for d in retrieved_docs
                ]
            )
            user_prompt = (
                f"Recent chat:\n{history}\n\n"
                f"Question:\n{state['question']}\n\n"
                f"Context:\n{context}\n\n"
                "Use the context when answering. End with a short 'Sources used' list."
            )
        else:
            sources_text = ", ".join(available_sources) if available_sources else "None"
            user_prompt = (
                f"Recent chat:\n{history}\n\n"
                f"Uploaded file names:\n{sources_text}\n\n"
                f"Question:\n{question}\n\n"
                "Answer directly without using uploaded-document facts. "
                "If the question depends on uploaded docs or missing evidence, reply with: "
                "'I don't know based on the current context.' "
                "Do not guess or invent resume/profile/work-history details. "
                "Do not claim there are no uploaded docs if file names are provided above."
            )

        msg = llm.invoke(
            [
                {"role": "system", "content": state["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ]
        )
        return {"answer": msg.content, "retrieved_docs": retrieved_docs, "use_rag": use_rag}

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile()


# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
    st.title("Document RAG Chat Assistant")

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "Missing OPENAI_API_KEY. Create a `.env` file in the project root with:\n\n"
            "OPENAI_API_KEY=your_key_here"
        )
        st.stop()

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    vectorstore = get_vectorstore()
    graph = build_graph(vectorstore)

    with st.sidebar:
        st.header("Settings")
        st.session_state["system_prompt"] = st.text_area(
            "System Prompt",
            value=st.session_state["system_prompt"],
            height=180,
            help="Edit assistant behavior.",
        )
        rag_mode_ui = st.radio(
            "RAG Mode",
            options=["Auto decide", "Always refer documents", "Do not refer documents"],
            index=0,
        )
        rag_mode_map = {
            "Auto decide": "auto",
            "Always refer documents": "always",
            "Do not refer documents": "never",
        }
        rag_mode = rag_mode_map[rag_mode_ui]

        if st.button("Clear chat history", type="primary", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, .csv, .pdf, .docx",
            type=["txt", "md", "csv", "pdf", "docx"],
            accept_multiple_files=True,
            key=f"uploaded_files_{st.session_state['uploader_version']}",
        )
        if st.button("Ingest Uploaded Files", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("No files selected.")
            else:
                try:
                    stats = ingest_documents(vectorstore, uploaded_files)
                    st.success(
                        f"Ingested {stats['ingested_count']} document(s). "
                        f"Skipped {stats['skipped_duplicates']} duplicate(s)."
                    )
                    # Force a fresh uploader widget to clear currently selected files.
                    st.session_state["uploader_version"] += 1
                    st.rerun()
                except Exception as e:
                    st.exception(e)

        st.subheader("Manage Vector DB")
        documents = list_uploaded_documents(vectorstore)
        if not documents:
            st.caption("No documents in vector DB.")
        else:
            selected_doc_id = st.selectbox(
                "Select document to delete",
                options=list(documents.keys()),
                format_func=lambda x: f"{documents[x]} ({x[:8]}...)",
            )
            if st.button("Delete Selected Document", type="primary", use_container_width=True):
                delete_document(vectorstore, selected_doc_id)
                st.success("Document deleted from vector DB.")
                st.rerun()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            evidence = msg.get("evidence", [])
            if msg["role"] == "assistant" and evidence:
                with st.expander("Retrieved Chunks (Evidence)"):
                    for idx, ev in enumerate(evidence, start=1):
                        st.markdown(f"**{idx}. {ev['source']} | chunk {ev['chunk_index']}**")
                        st.write(ev["content"])

    prompt = st.chat_input("Ask a question about your uploaded documents")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        docs_available = len(list_uploaded_documents(vectorstore)) > 0
        available_sources = list(list_uploaded_documents(vectorstore).values())

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = graph.invoke(
                    {
                        "question": prompt,
                        "system_prompt": st.session_state["system_prompt"],
                        "chat_history": st.session_state["messages"][:-1],
                        "rag_mode": rag_mode,
                        "docs_available": docs_available,
                        "available_sources": available_sources,
                        "use_rag": False,
                        "retrieved_docs": [],
                        "answer": "",
                    }
                )

            answer = result["answer"]
            evidence = docs_to_evidence(result.get("retrieved_docs", []))
            st.markdown(answer)
            if evidence:
                with st.expander("Retrieved Chunks (Evidence)"):
                    for idx, ev in enumerate(evidence, start=1):
                        st.markdown(f"**{idx}. {ev['source']} | chunk {ev['chunk_index']}**")
                        st.write(ev["content"])

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "evidence": evidence,
                "used_rag": bool(result.get("use_rag", False)),
            }
        )


if __name__ == "__main__":
    main()
