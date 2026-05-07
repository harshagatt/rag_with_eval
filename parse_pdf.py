# parse_pdf.py
from docling.document_converter import DocumentConverter
import os

def parse_financial_document(input_path: str, output_path: str):
    """
    Converts a financial PDF into structured Markdown.
    Tables are preserved in proper Markdown table format.
    """
    print(f"Parsing: {input_path}")
    
    converter = DocumentConverter()
    result = converter.convert(input_path)
    
    # Export to Markdown (tables become proper Markdown tables)
    markdown_content = result.document.export_to_markdown()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"Saved parsed document to: {output_path}")
    print(f"Total characters: {len(markdown_content):,}")
    return markdown_content


if __name__ == "__main__":
    content = parse_financial_document(
        input_path="data/raw/report.pdf",
        output_path="data/parsed/report.md"
    )
    print("\n--- FIRST 2000 CHARACTERS OF PARSED OUTPUT ---")
    print(content[:2000])