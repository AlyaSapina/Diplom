"""
RAG-движок с поддержкой перевода на русский язык.
"""

import os
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Tuple

class RAGEngine:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Эмбеддинги
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

        # Модель перевода (английский → русский)
        print("Загрузка модели перевода en→ru...")
        self.translator = pipeline(
            "translation_en_to_ru",
            model="Helsinki-NLP/opus-mt-en-ru",
            device=0 if torch.cuda.is_available() else -1  # GPU если есть
        )
        print("Модель перевода загружена.")

    def add_chunks(self, chunks: List[str]):
        embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def save_index(self, folder_path: str = "models"):
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "faiss_index.bin"))
        with open(os.path.join(folder_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, folder_path: str = "models"):
        index_path = os.path.join(folder_path, "faiss_index.bin")
        chunks_path = os.path.join(folder_path, "chunks.json")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.dimension = self.index.d
        else:
            raise FileNotFoundError("Индекс не найден. Сначала загрузите инструкции.")

    def _is_english(self, text: str) -> bool:
        """Простая эвристика: если много латинских букв — английский."""
        latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        total = len(text)
        return total > 0 and latin / total > 0.3

    def _translate_to_russian(self, text: str) -> str:
        """Переводит текст на русский, если он на английском."""
        if self._is_english(text):
            try:
                result = self.translator(text, max_length=512)
                return result[0]['translation_text']
            except Exception as e:
                print(f"Ошибка перевода: {e}")
                return f"[Ошибка перевода] {text}"
        else:
            return text  # уже на русском или другом языке

    def retrieve(self, query: str, k: int = 1) -> str:
        """Возвращает самый релевантный чанк."""
        if self.index.ntotal == 0:
            return "Инструкции не загружены."
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(query_emb, k)
        best_chunk = self.chunks[indices[0][0]] if indices[0][0] < len(self.chunks) else "Не найдено."
        return best_chunk

    def ask(self, query: str) -> Tuple[str, str]:
        """
        Возвращает:
        - ответ на русском (переведённый, если нужно),
        - контекст (оригинал, для отладки).
        """
        context = self.retrieve(query)
        answer = self._translate_to_russian(context)
        return answer, context