# src/rag_engine.py
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []
        self.bad_chunks = set()  # чанки, помеченные как "неверно"
        self._load_bad_chunks()

    def _load_bad_chunks(self):
        path = "feedback/bad_chunks.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.bad_chunks = set(json.load(f))

    def _save_bad_chunks(self):
        os.makedirs("feedback", exist_ok=True)
        with open("feedback/bad_chunks.json", "w", encoding="utf-8") as f:
            json.dump(list(self.bad_chunks), f, ensure_ascii=False)

    def add_chunks(self, chunks: List[str]):
        self.chunks.extend(chunks)
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        self.index.add(embeddings)

    def save_index(self, folder="models"):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, f"{folder}/faiss_index.bin")
        with open(f"{folder}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, folder="models"):
        index_path = os.path.join(folder, "faiss_index.bin")
        chunks_path = os.path.join(folder, "chunks.json")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            raise FileNotFoundError("Индекс не найден")

    def ask(self, query: str) -> Tuple[str, str]:
        if self.index.ntotal == 0:
            return "Сначала загрузите инструкции.", ""
        
        emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(emb, 10)  # ищем 10, а не 1

        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if chunk not in self.bad_chunks:  # пропускаем "плохие" чанки
                return chunk, chunk
        
        return "Подходящий фрагмент не найден.", ""

    def mark_as_bad(self, chunk: str):
        """Вызывается при нажатии 'Неверно'."""
        self.bad_chunks.add(chunk)
        self._save_bad_chunks()