import os
import json
import hashlib
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []
        self.document_hashes = set()
        self.loaded_documents = []
        self._load_document_metadata()

    def _compute_pdf_hash(self, pdf_path: str) -> str:
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_document_metadata(self):
        meta_path = "models/document_metadata.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.document_hashes = set(data.get("hashes", []))
                    self.loaded_documents = data.get("filenames", [])
            except Exception:
                self.document_hashes = set()
                self.loaded_documents = []
        else:
            self.document_hashes = set()
            self.loaded_documents = []

    def _save_document_metadata(self):
        os.makedirs("models", exist_ok=True)
        with open("models/document_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "hashes": list(self.document_hashes),
                "filenames": self.loaded_documents
            }, f, ensure_ascii=False, indent=2)

    def add_chunks(self, chunks: List[str]):
        clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
        if not clean_chunks:
            return
        self.chunks.extend(clean_chunks)
        embeddings = self.model.encode(clean_chunks, normalize_embeddings=True)
        self.index.add(embeddings)

    def save_index(self, folder: str = "models"):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "faiss_index.bin"))
        with open(os.path.join(folder, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, folder: str = "models"):
        index_path = os.path.join(folder, "faiss_index.bin")
        chunks_path = os.path.join(folder, "chunks.json")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            raise FileNotFoundError("Индекс не найден")

    def add_document(self, pdf_path: str) -> bool:
        pdf_hash = self._compute_pdf_hash(pdf_path)
        if pdf_hash in self.document_hashes:
            return False
        from src.pdf_loader import process_pdf_to_chunks
        chunks = process_pdf_to_chunks(pdf_path)
        if not chunks:
            return False
        self.add_chunks(chunks)
        filename = os.path.basename(pdf_path)
        self.loaded_documents.append(filename)
        self.document_hashes.add(pdf_hash)
        self._save_document_metadata()
        self.save_index()
        return True

    def get_loaded_documents(self) -> List[str]:
        return self.loaded_documents.copy()

    def ask(self, query: str) -> Tuple[str, str]:
        if self.index.ntotal == 0:
            return "Сначала загрузите инструкции.", ""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        k = min(20, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            formatted = chunk.replace(". ", ".\n\n").strip()
            return formatted, chunk
        return "Подходящий фрагмент не найден.", ""