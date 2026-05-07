# hybrid_search.py
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class HybridRetriever:
    def __init__(self, vectorstore: QdrantVectorStore, documents: list[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        print("I am starting")
        # Build a BM25 index from the raw text of all chunks
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def reciprocal_rank_fusion(
        self,
        vector_results: list[Document],
        bm25_results: list[Document],
        k: int = 60
    ) -> list[Document]:
        """
        RRF merges two ranked lists. A document's score is:
          score = 1/(k + rank_in_list_1) + 1/(k + rank_in_list_2)
        k=60 is the standard default — softens the penalty for lower-ranked items.
        """
        scores = {}
        
        for rank, doc in enumerate(vector_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, {"doc": doc, "score": 0})
            scores[key]["score"] += 1 / (k + rank + 1)
        
        for rank, doc in enumerate(bm25_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, {"doc": doc, "score": 0})
            scores[key]["score"] += 1 / (k + rank + 1)
        
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]
    
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # 1. Semantic vector search
        vector_results = self.vectorstore.similarity_search(query, k=top_k * 2)
        
        # 2. BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        bm25_results = [self.documents[i] for i in top_bm25_indices]
        
        # 3. Fuse using RRF
        fused = self.reciprocal_rank_fusion(vector_results, bm25_results)
        return fused[:top_k]


def demonstrate_hybrid_vs_vector(retriever: HybridRetriever, vectorstore: QdrantVectorStore):
    """EXERCISE: Run the same query through both methods and compare results."""
    test_query = "Total Net Sales 2025"  # Replace with a real code from your document
    
    print(f"\nQuery: '{test_query}'")
    print("=" * 60)
    
    print("\n[PURE VECTOR SEARCH RESULTS]")
    vector_only = vectorstore.similarity_search(test_query, k=3)
    for i, doc in enumerate(vector_only):
        print(f"  Result {i+1}: {doc.page_content[:150]}...")
    
    print("\n[HYBRID SEARCH RESULTS (Vector + BM25 + RRF)]")
    hybrid_results = retriever.retrieve(test_query, top_k=3)
    for i, doc in enumerate(hybrid_results):
        print(f"  Result {i+1}: {doc.page_content[:150]}...")
    
    print("\nObservation: Hybrid search should find the exact product code,")
    print("while vector search may return 'similar' but incorrect products.")