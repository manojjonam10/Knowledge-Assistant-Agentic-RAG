from typing import List, Dict
from pathlib import Path


def load_documents(doc_dir: str) -> List[Dict]:
    """
    Load data/docs into a list of {id, text, metadata}.
    """
    docs = []
    doc_path = Path(doc_dir)
    idx = 0
    for file in doc_path.glob("**/*"):
        if file.suffix.lower() not in [".md", ".txt"]:
            continue
        text = file.read_text(encoding="utf-8", errors="ignore")
        # Split into chunks
        chunks = split_into_chunks(text, chunk_size=600, overlap=100)
        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{file.name}-{idx}-{i}",
                "text": chunk,
                "metadata": {
                    "source_file": file.name,
                    "source_path": str(file),
                },
            })
        idx += 1
    return docs


def split_into_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
   Split by characters.
    """
    text = text.strip()
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks
