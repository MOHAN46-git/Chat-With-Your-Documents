
import os
import uuid

import chromadb
import fitz
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------- CONFIGURATION ----------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.8-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
BATCH_SIZE = 50

st.set_page_config(
    page_title="Chat With Your Documents",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat With Your Documents")
st.caption("A Retrieval-Augmented Generation (RAG) application")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(
    name="document_knowledge",
    metadata={"hnsw:space": "cosine"},
)


# ---------------- DOCUMENT PROCESSING ----------------

def extract_pdf(uploaded_file):
    """Extract text and page numbers from an uploaded PDF."""
    pdf = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    )

    pages = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append({
                "text": text,
                "page": page_number,
            })

    pdf.close()
    return pages


def split_into_chunks(pages, filename):
    """Split page text into overlapping chunks."""
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": page["page"],
                })

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# ---------------- EMBEDDINGS ----------------

def create_embeddings(texts, task_type):
    """Generate Gemini embeddings in batches."""
    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )

        vectors.extend([
            embedding.values
            for embedding in response.embeddings
        ])

    return vectors


# ---------------- INDEX DOCUMENTS ----------------

def index_documents(uploaded_files):
    all_chunks = []

    for uploaded_file in uploaded_files:
        pages = extract_pdf(uploaded_file)
        chunks = split_into_chunks(
            pages,
            uploaded_file.name,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "No readable text found. The PDFs may be scanned images."
        )

    texts = [item["text"] for item in all_chunks]

    vectors = create_embeddings(
        texts,
        "RETRIEVAL_DOCUMENT",
    )

    # Replace the previous index with this upload batch.
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [str(uuid.uuid4()) for _ in all_chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "source": item["source"],
                "page": item["page"],
            }
            for item in all_chunks
        ],
    )

    return len(all_chunks)


# ---------------- RETRIEVAL ----------------

def retrieve_context(question):
    query_vector = create_embeddings(
        [question],
        "RETRIEVAL_QUERY",
    )[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
        })

    return retrieved


# ---------------- GENERATION ----------------

def generate_answer(question, retrieved_chunks):
    context_parts = []

    for index, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {index}: {item['source']}, "
            f"page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not invent facts or use outside knowledge.
2. If the context does not contain the answer, clearly say:
   "I could not find this information in the uploaded documents."
3. Cite supporting sources using [Source 1], [Source 2], etc.
4. Distinguish direct evidence from reasonable inference.
5. Be clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1200,
        ),
    )

    return response.text or "The model returned an empty response."


# ---------------- STREAMLIT UI ----------------

with st.sidebar:
    st.header("Upload documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Process and Index Documents",
        type="primary",
        disabled=not uploaded_files,
    ):
        try:
            with st.spinner(
                "Extracting text, creating embeddings and indexing..."
            ):
                chunk_count = index_documents(uploaded_files)

            st.session_state["indexed"] = True
            st.success(
                f"Successfully indexed {chunk_count} chunks."
            )

        except Exception as error:
            st.error(f"Indexing failed: {error}")

    st.divider()
    st.metric("Indexed chunks", collection.count())

st.subheader("Ask questions about your documents")

if collection.count() == 0:
    st.info("Upload one or more text-based PDFs and click Process and Index Documents.")
else:
    question = st.text_input(
        "Your question",
        placeholder="What are the main topics covered in this document?",
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching documents and generating answer..."):
                    retrieved = retrieve_context(question)

                    if not retrieved:
                        st.warning("No relevant document content was found.")
                    else:
                        answer = generate_answer(
                            question,
                            retrieved,
                        )

                        st.markdown("### Answer")
                        st.write(answer)

                        st.markdown("### Retrieved sources")

                        for index, item in enumerate(retrieved, start=1):
                            with st.expander(
                                f"Source {index} — {item['source']} "
                                f"(Page {item['page']})"
                            ):
                                st.write(item["text"])
                                st.caption(
                                    f"Vector distance: {item['distance']:.4f}"
                                )

            except Exception as error:
                st.error(f"Question answering failed: {error}")

st.divider()
st.caption(
    "Built with Streamlit, Gemini API, PyMuPDF and ChromaDB."
)
