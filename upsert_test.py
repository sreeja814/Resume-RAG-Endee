from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

INDEX_NAME = "documents_v2"
PDF_PATH = Path(r"C:\Users\umama\Desktop\endee project\sreeja_resumeupdate.pdf")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)
    return "\n".join(full_text)

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    client = Endee()

    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8
        )
        print(f"Index '{INDEX_NAME}' created successfully")
    except Exception as e:
        print("Create index error:", e)
        return

    index = client.get_index(name=INDEX_NAME)
    print(f"Index '{INDEX_NAME}' fetched successfully")

    text = extract_text_from_pdf(PDF_PATH)
    if not text.strip():
        raise ValueError("No text extracted from PDF")

    chunks = chunk_text(text)
    print("Total chunks:", len(chunks))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        vectors.append({
            "id": f"resume_chunk_{i}",
            "vector": emb.tolist(),
            "meta": {
                "chunk_number": i,
                "text": chunk
            }
        })

    index.upsert(vectors)
    print(f"Upsert successful: {len(vectors)} vectors inserted")

    results = index.query(
        vector=embeddings[0].tolist(),
        top_k=3
    )

    print("\nQuery results:")
    for item in results:
        print(item)

if __name__ == "__main__":
    main()