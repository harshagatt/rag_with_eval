# embed_and_store.py
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore          # ← new import
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

def embed_and_store(markdown_path: str, collection_name: str):
    # 1. Load the parsed markdown
    with open(markdown_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)
    chunks = splitter.create_documents([text])
    print(f"Created {len(chunks)} chunks")

    # 3. Initialize the local embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # 4. Create Qdrant client and explicitly recreate the collection
    #    This avoids the broken `force_recreate` path in langchain-community
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    # Delete the collection if it already exists (clean slate)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: '{collection_name}'")

    # Create a fresh collection
    # nomic-embed-text produces 768-dimensional vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"Created collection: '{collection_name}'")

    # 5. Store chunks + embeddings using the new langchain-qdrant package
    print("Embedding and storing chunks (this may take a few minutes)...")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)

    print(f"Successfully stored {len(chunks)} chunks in '{collection_name}'")
    return vectorstore


if __name__ == "__main__":
    vectorstore = embed_and_store(
        markdown_path="data/parsed/report.md",
        collection_name=os.getenv("COLLECTION_NAME", "financial_docs")
    )

    # Smoke test
    results = vectorstore.similarity_search("What was the total revenue?", k=3)
    print("\n--- TOP 3 RETRIEVED CHUNKS FOR: 'What was the total revenue?' ---")
    for i, doc in enumerate(results):
        print(f"\n[Chunk {i+1}]")
        print(doc.page_content[:300])