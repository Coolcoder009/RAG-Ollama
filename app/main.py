import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import ollama
import uuid

PDF_PATH = "../Attention.pdf"
COLLECTION_NAME = "Transformers"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = os.path.abspath("./chroma_store")

embedder = SentenceTransformer(EMBED_MODEL)

chroma_client = PersistentClient(path=CHROMA_DIR)
existing_collections = [c.name for c in chroma_client.list_collections()]

if COLLECTION_NAME in existing_collections:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"[Loaded] Existing collection: {COLLECTION_NAME}")
else:
    print(f"[Creating] New collection: {COLLECTION_NAME}")

    def extract_chunks_from_pdf(path, chunk_size=103):
        reader = PdfReader(path)
        all_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text.strip() + "\n"
        return [all_text[i:i + chunk_size] for i in range(0, len(all_text), chunk_size)]

    chunks = extract_chunks_from_pdf(PDF_PATH)

    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    embeddings = embedder.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"[Inserted] {len(chunks)} chunks into: {COLLECTION_NAME}")

query = "What is Dynamic Programming"
query_results = collection.query(query_texts=[query], n_results=3)
top_chunks = "\n---\n".join(query_results["documents"][0])

response = ollama.chat(
    model="deepseek-r1:1.5b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a highly concise and focused assistant. "
                "Only answer based on the provided context. "
                "Avoid explanations beyond the context. "
                "Respond clearly and briefly (3–5 lines max)."
            )
        },
        {
            "role": "user",
            "content": (
                f"Use only the following context to answer the question. "
                f"If not relevant, say 'Not found in the document.'\n\n"
                f"{top_chunks}\n\n"
                f"Question: {query}"
            )
        }
    ],
    stream=True
)


after_think_started = False
buffer = ""

for chunk in response:
    content = chunk['message']['content']

    if not after_think_started:
        buffer += content
        if "</think>" in buffer:
            after_think_started = True
            after_content = buffer.split("</think>", 1)[-1].strip()
            print(after_content, end='', flush=True)
    else:

        print(content, end='', flush=True)
