# chunk_documents.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document


def load_markdown(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# --- METHOD A: Recursive Character Splitting (recommended default) ---
def recursive_chunking(text: str) -> list[Document]:
    """
    Splits text by trying paragraph breaks first, then sentences, 
    then words — never cuts mid-word unless forced to.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,           # Target: ~512 characters per chunk
        chunk_overlap=64,         # 64 chars of overlap to preserve context across chunks
        separators=["\n\n", "\n", ". ", " ", ""]  # Priority order of split points
    )
    return splitter.create_documents([text])


# --- METHOD B: Fixed Window Chunking (bad practice — for comparison only) ---
def fixed_window_chunking(text: str) -> list[Document]:
    """
    Splits every 500 characters regardless of sentence boundaries.
    EXERCISE: Compare the chunk contents against recursive chunking.
    """
    chunks = []
    for i in range(0, len(text), 500):
        chunks.append(Document(page_content=text[i:i+500]))
    return chunks


# --- METHOD C: Header-Based Chunking (best for structured financial docs) ---
def header_based_chunking(text: str) -> list[Document]:
    """
    Splits the document at Markdown headers. Each chunk = one section.
    Best for documents where each section covers a distinct topic.
    """
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return splitter.split_text(text)


if __name__ == "__main__":
    text = load_markdown("data/parsed/report.md")
    
    recursive_chunks = recursive_chunking(text)
    fixed_chunks = fixed_window_chunking(text)
    header_chunks = header_based_chunking(text)

    print(f"Recursive chunking  → {len(recursive_chunks)} chunks")
    print(f"Fixed window        → {len(fixed_chunks)} chunks")
    print(f"Header-based        → {len(header_chunks)} chunks")

    print("\n--- RECURSIVE CHUNK SAMPLE ---")
    print(recursive_chunks[5].page_content)
    
    print("\n--- FIXED WINDOW CHUNK SAMPLE (may be cut mid-sentence) ---")
    print(fixed_chunks[5].page_content)
    
    print("\n--- HEADER CHUNK SAMPLE (with metadata) ---")
    print(header_chunks[2].page_content[:400])
    print("Metadata:", header_chunks[2].metadata)