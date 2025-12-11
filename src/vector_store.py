from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.docs: List[Dict] = []

    def build(self, docs: List[Dict]):
        """
        docs: list of {id, text, metadata}
        """
        self.docs = docs
        texts = [d["text"] for d in docs]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))
        print("Vector index built.")

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb.astype("float32"), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            doc = self.docs[int(idx)]
            results.append((doc, float(dist)))
        return results
