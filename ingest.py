# ingest.py (updated)
import pdfplumber
import docx
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )
    return splitter.split_text(text)

def process_documents(folder_path="documents"):
    all_chunks = []
    folder = Path(folder_path)
    for file in folder.glob("*"):
        if file.suffix == ".pdf":
            raw_text = extract_text_from_pdf(str(file))
        elif file.suffix == ".docx":
            raw_text = extract_text_from_docx(str(file))
        elif file.suffix == ".txt":
            raw_text = extract_text_from_txt(str(file))
        else:
            print(f"Skipping unsupported file: {file}")
            continue
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    chunks = process_documents("documents")
    print(f"Extracted {len(chunks)} chunks from all docs.")
    print(chunks[:2])
