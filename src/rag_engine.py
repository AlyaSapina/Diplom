"""
RAG-движок с переводом на русский через Argos Translate.
"""

import os
import json
import faiss
import argostranslate.package
import argostranslate.translate
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Устанавливаем перевод один раз при импорте
def _install_translation():
    """Устанавливает пакет перевода en→ru, если ещё не установлен."""
    installed_languages = {lang.code for lang in argostranslate.translate.get_installed_languages()}
    if "en" not in installed_languages or "ru" not in installed_languages:
        print("Установка пакета перевода en→ru (первый запуск, может занять 1–2 минуты)...")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            pkg for pkg in available_packages if pkg.from_code == "en" and pkg.to_code == "ru"
        )
        argostranslate.package.install_from_path(package_to_install.download())
        print("Пакет перевода установлен.")

# Выполняем установку при импорте модуля
_install_translation()

class RAGEngine:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.feedback_penalties = {}  # {chunk_text: штраф}
        self._load_feedback()

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
        else:
            raise FileNotFoundError("Индекс не найден. Сначала загрузите инструкции.")

    def _is_english(self, text: str) -> bool:
        """Простая эвристика: если много латинских букв — считаем английским."""
        if not text.strip():
            return False
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        total_chars = len(text)
        return total_chars > 0 and latin_chars / total_chars > 0.3

    def _translate_to_russian(self, text: str) -> str:
        """Переводит текст с английского на русский. Если не английский — возвращает как есть."""
        if self._is_english(text):
            try:
                return argostranslate.translate.translate(text, "en", "ru")
            except Exception as e:
                return f"[Ошибка перевода] {text}"
        return text

    def ask(self, query: str) -> Tuple[str, str]:
        """Возвращает (ответ на русском, оригинальный контекст)."""
        if self.index.ntotal == 0:
            return "Сначала загрузите инструкции.", ""

        original_context = self.retrieve_with_penalty(query)
        answer = self._translate_to_russian(original_context)

        # Улучшаем читаемость
        answer = answer.replace(". ", ".\n\n").strip()
        original_context = original_context.replace(". ", ".\n\n").strip()

        return answer, original_context

    def _load_feedback(self):
        """Загружает историю обратной связи."""
        feedback_path = "feedback/feedback.json"
        if os.path.exists(feedback_path):
            with open(feedback_path, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
            for fb in feedbacks:
                if not fb.get("is_correct", True):  # если НЕВЕРНО
                    chunk = fb["chunks"][0] if isinstance(fb["chunks"], list) else fb["chunks"]
                    self.feedback_penalties[chunk] = self.feedback_penalties.get(chunk, 0) + 1

    def retrieve_with_penalty(self, query: str, k: int = 3) -> str:
        """
        Находит самый релевантный чанк, учитывая штрафы за 'Неверно'.
        """
        if self.index.ntotal == 0:
            return "Инструкции не загружены."

        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_emb, k * 5)  # берём больше кандидатов

        candidates = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            raw_score = distances[0][i]  # чем выше, тем лучше
            penalty = self.feedback_penalties.get(chunk, 0) * 0.5  # штраф снижает оценку
            adjusted_score = raw_score - penalty
            candidates.append((adjusted_score, chunk))

        # Сортируем по скорректированной оценке
        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            return candidates[0][1]
        else:
            return "Не найдено."