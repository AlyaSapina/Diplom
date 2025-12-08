"""
Итоговый RAG-движок для AI-помощника инженера 1-й линии.
Реализует:
- поиск по инструкциям (PDF),
- фильтрацию мусора,
- обучение на обратной связи,
- поддержку выделения плохих фрагментов.
"""

import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class RAGEngine:
    def __init__(self):
        # Мультиязычная модель эмбеддингов (поддерживает ru/en)
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []  # все фрагменты из документов
        self.bad_fragments = set()  # фрагменты, помеченные как нерелевантные
        self._load_bad_fragments()

    def _load_bad_fragments(self):
        """Загружает список нерелевантных фрагментов."""
        bad_path = "feedback/bad_fragments.json"
        if os.path.exists(bad_path):
            try:
                with open(bad_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.bad_fragments = set(data) if isinstance(data, list) else set()
            except Exception:
                self.bad_fragments = set()
        else:
            self.bad_fragments = set()

    def _is_chunk_bad(self, chunk: str) -> bool:
        """Проверяет, содержит ли чанк нерелевантный фрагмент."""
        if not self.bad_fragments:
            return False
        for bad in self.bad_fragments:
            if bad.strip() in chunk:
                return True
        return False

    def add_chunks(self, chunks: List[str]):
        """Добавляет фрагменты в индекс."""
        clean_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 20:  # пропускаем короткие фрагменты
                continue
            clean_chunks.append(chunk)

        if not clean_chunks:
            return

        self.chunks.extend(clean_chunks)
        embeddings = self.model.encode(clean_chunks, normalize_embeddings=True)
        self.index.add(embeddings)

    def save_index(self, folder: str = "models"):
        """Сохраняет FAISS-индекс и фрагменты."""
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "faiss_index.bin"))
        with open(os.path.join(folder, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, folder: str = "models"):
        """Загружает индекс и фрагменты."""
        index_path = os.path.join(folder, "faiss_index.bin")
        chunks_path = os.path.join(folder, "chunks.json")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            raise FileNotFoundError("Индекс не найден")

    def ask(self, query: str) -> Tuple[str, str]:
        """
        Возвращает (ответ, контекст) — всегда дословный фрагмент из документа.
        Исключает помеченные как плохие фрагменты.
        """
        if self.index.ntotal == 0:
            return "Сначала загрузите инструкции.", ""

        query_emb = self.model.encode([query], normalize_embeddings=True)
        # Ищем до 20 кандидатов, чтобы был выбор
        scores, indices = self.index.search(query_emb, min(20, self.index.ntotal))

        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if not self._is_chunk_bad(chunk):
                # Улучшаем читаемость
                formatted = chunk.replace(". ", ".\n\n").strip()
                return formatted, chunk

        return "Подходящий фрагмент не найден.", ""

    def mark_fragment_as_bad(self, fragment: str):
        """
        Помечает фрагмент как нерелевантный.
        Может быть частью большого чанка.
        """
        if not fragment or not fragment.strip():
            return
        fragment = fragment.strip()
        self.bad_fragments.add(fragment)

        # Сохраняем обновлённый список
        os.makedirs("feedback", exist_ok=True)
        with open("feedback/bad_fragments.json", "w", encoding="utf-8") as f:
            json.dump(list(self.bad_fragments), f, ensure_ascii=False, indent=2)