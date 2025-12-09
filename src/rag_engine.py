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
import re
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

    def _normalize_text(self, text: str) -> str:
        """Приводит текст к единому виду для сравнения."""
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _is_chunk_bad(self, chunk: str) -> bool:
        norm_chunk = self._normalize_text(chunk)
        for bad in self.bad_fragments:
            norm_bad = self._normalize_text(bad)
            if norm_bad and norm_bad in norm_chunk:
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
        Возвращает кортеж (ответ_на_русском, оригинальный_контекст).
        Ответ — это дословный фрагмент из документа, отформатированный для читаемости.
        Все чанки, содержащие ранее помеченные как нерелевантные подстроки, исключаются.
        """
        if self.index.ntotal == 0:
            return "Сначала загрузите инструкции.", ""

        # Генерируем эмбеддинг запроса
        query_emb = self.model.encode([query], normalize_embeddings=True)

        # Ищем до 20 кандидатов (или меньше, если документов мало)
        k = min(20, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)

        # Проходим по кандидатам от наиболее к наименее релевантному
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            # Проверяем, не содержит ли чанк запрещённых фрагментов
            if self._is_chunk_bad(chunk):
                continue  # пропускаем плохой чанк

            # Форматируем для лучшей читаемости (разбиваем на абзацы)
            formatted_answer = chunk.replace(". ", ".\n\n").strip()
            return formatted_answer, chunk

        # Если все кандидаты отфильтрованы
        return "Подходящий фрагмент не найден. Попробуйте переформулировать вопрос.", ""
    def mark_fragment_as_bad(self, fragment: str):
        """
        Помечает фрагмент как нерелевантный.
        Может быть частью большого чанка.
        """
        if not fragment or not fragment.strip():
            return
        fragment = fragment.strip()
        self.bad_fragments.add(fragment)

        # Очищаем от лишних пробелов и переносов
        clean_fragment = re.sub(r'\s+', ' ', fragment.strip())
        if len(clean_fragment) < 5:  # слишком коротко — игнорируем
            return
        self.bad_fragments.add(clean_fragment)

        # Сохраняем обновлённый список
        os.makedirs("feedback", exist_ok=True)
        with open("feedback/bad_fragments.json", "w", encoding="utf-8") as f:
            json.dump(list(self.bad_fragments), f, ensure_ascii=False, indent=2)
