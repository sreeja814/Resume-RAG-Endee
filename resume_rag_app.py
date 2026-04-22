import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
from openai import OpenAI

INDEX_NAME = "resume_chat_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_STORAGE = Path("uploaded_resume.pdf")

st.set_page_config(page_title="Resume RAG Assistant", page_icon="📄", layout="wide")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path_or_file):
    reader = PdfReader(pdf_path_or_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def ensure_index(client):
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8
        )
    except Exception:
        pass


def build_vectors(chunks, embeddings):
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        vectors.append({
            "id": f"chunk_{i}",
            "vector": emb.tolist(),
            "meta": {
                "chunk_number": i,
                "text": chunk
            }
        })
    return vectors


def build_prompt(question, context_text):
    return f"""
You are a helpful resume assistant.

Answer the user's question using only the context below.
If the answer is not present in the context, say: "I could not find that in the resume."
Keep the answer clear, short, and professional.

Context:
{context_text}

Question:
{question}

Answer:
""".strip()


def call_llm(prompt):
    api_key = get_openai_api_key()

    if not api_key:
        return "LLM not configured. Add OPENAI_API_KEY to .streamlit/secrets.toml or set it in your environment."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions strictly from resume context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"


st.title("📄 Resume RAG Assistant using Endee")
st.write("Upload a resume PDF, index it in Endee, and ask questions about it.")

with st.sidebar:
    st.subheader("Debug")
    env_key_found = bool(os.getenv("OPENAI_API_KEY"))
    try:
        secrets_key_found = "OPENAI_API_KEY" in st.secrets
    except Exception:
        secrets_key_found = False

    st.write("Environment key found:", env_key_found)
    st.write("Secrets key found:", secrets_key_found)

    st.markdown("### Local secrets file")
    st.code(
        '[project-folder]/.streamlit/secrets.toml\n\nOPENAI_API_KEY = "your_key_here"',
        language="toml"
    )

uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

if "client" not in st.session_state:
    st.session_state.client = Endee()

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None


if uploaded_file is not None:
    if st.session_state.last_uploaded_name != uploaded_file.name:
        with open(PDF_STORAGE, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Extracting text from PDF..."):
            raw_text = extract_text_from_pdf(str(PDF_STORAGE))
            raw_text = clean_text(raw_text)

        if not raw_text.strip():
            st.error("No text could be extracted from the PDF.")
            st.stop()

        with st.spinner("Chunking resume text..."):
            chunks = chunk_text(raw_text)

        with st.spinner("Loading embedding model..."):
            model = load_embedding_model()

        with st.spinner("Generating embeddings..."):
            embeddings = model.encode(chunks)

        with st.spinner("Creating Endee index..."):
            ensure_index(st.session_state.client)

        index = st.session_state.client.get_index(name=INDEX_NAME)

        with st.spinner("Upserting chunks into Endee..."):
            vectors = build_vectors(chunks, embeddings)
            index.upsert(vectors)

        st.session_state.index_ready = True
        st.session_state.last_uploaded_name = uploaded_file.name
        st.success(f"Resume indexed successfully with {len(chunks)} chunks.")
    else:
        st.info("This resume is already indexed in the current session.")

question = st.text_input("Ask a question about the resume")

if question and st.session_state.index_ready:
    model = load_embedding_model()
    query_embedding = model.encode([question])[0]

    index = st.session_state.client.get_index(name=INDEX_NAME)
    results = index.query(vector=query_embedding.tolist(), top_k=3)

    context_parts = []
    source_chunks = []

    for item in results:
        meta = item.get("meta", {})
        text = meta.get("text", "").strip()
        chunk_no = meta.get("chunk_number", "unknown")
        if text:
            context_parts.append(text)
            source_chunks.append((chunk_no, text))

    context_text = "\n\n".join(context_parts)
    prompt = build_prompt(question, context_text)

    with st.spinner("Generating answer..."):
        answer = call_llm(prompt)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources"):
        for chunk_no, text in source_chunks:
            st.markdown(f"**Chunk {chunk_no}**")
            st.write(text)

elif question and not st.session_state.index_ready:
    st.warning("Please upload and index a resume first.")