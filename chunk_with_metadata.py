# chunk_with_metadata.py
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import re

def extract_metadata_with_llm(chunk_text: str) -> dict:
    """
    Uses the local LLM to automatically extract financial metadata from a chunk.
    This is called 'auto-tagging' — letting AI label your data for you.
    """
    llm = ChatOllama(model="llama3", temperature=0)
    
    prompt = f"""You are a financial document analyst. 
Analyze the following text chunk and extract metadata.
Respond ONLY with a valid JSON object. No explanation, no markdown, just JSON.

Text:
{chunk_text[:800]}

Return this exact JSON structure (use null if not found):
{{
  "product_type": "string or null (e.g., 'equity', 'bond', 'ETF', 'mutual fund')",
  "risk_level": "string or null (e.g., 'low', 'medium', 'high')",
  "effective_date": "string or null (e.g., '2023-09-30')",
  "product_code": "string or null (e.g., 'AAPL', 'X123')",
  "section_type": "string or null (e.g., 'revenue', 'risk factors', 'balance sheet')"
}}"""

    response = llm.invoke(prompt)
    
    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


def chunk_with_metadata(markdown_path: str) -> list[Document]:
    with open(markdown_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.create_documents([text])
    
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Tagging chunk {i+1}/{len(chunks)}...")
        metadata = extract_metadata_with_llm(chunk.page_content)
        chunk.metadata.update(metadata)
        enriched_chunks.append(chunk)
    
    return enriched_chunks


if __name__ == "__main__":
    chunks = chunk_with_metadata("data/parsed/report.md")
    
    print("\n--- CHUNKS WITH METADATA ---")
    for chunk in chunks[:5]:
        if chunk.metadata:
            print(f"\nMetadata: {chunk.metadata}")
            print(f"Content preview: {chunk.page_content[:150]}...")