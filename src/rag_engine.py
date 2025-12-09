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
        self.document_hashes = set()  # множество хешей загруженных PDF
        self.loaded_documents = []  # список имён файлов
        self._load_document_metadata()

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

        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(query_emb, k=1)
        idx = indices[0][0]

        if idx >= len(self.chunks):
            return "Не найдено.", ""

        original_context = self.chunks[idx]
        answer = self._translate_to_russian(original_context)

        # Улучшаем читаемость
        answer = answer.replace(". ", ".\n\n").strip()
        original_context = original_context.replace(". ", ".\n\n").strip()

        return answer, original_context

    def _compute_pdf_hash(self, pdf_path: str) -> str:
        """Вычисляет хеш содержимого PDF-файла."""
        import hashlib
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_document_metadata(self):
        """Загружает список загруженных документов и их хешей."""
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
        """Сохраняет метаданные о загруженных документах."""
        os.makedirs("models", exist_ok=True)
        with open("models/document_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "hashes": list(self.document_hashes),
                "filenames": self.loaded_documents
            }, f, ensure_ascii=False, indent=2)

    def add_document(self, pdf_path: str) -> bool:
        """
        Добавляет PDF в индекс, если он ещё не загружен.
        Возвращает: True — добавлен, False — дубликат.
        """
        pdf_hash = self._compute_pdf_hash(pdf_path)
        if pdf_hash in self.document_hashes:
            return False  # уже загружен

        # Добавляем чанки
        from src.pdf_loader import process_pdf_to_chunks
        chunks = process_pdf_to_chunks(pdf_path)
        if not chunks:
            return False

        self.add_chunks(chunks)
        filename = os.path.basename(pdf_path)
        self.loaded_documents.append(filename)
        self.document_hashes.add(pdf_hash)
        self._save_document_metadata()
        self.save_index()  # обновляем FAISS и chunks.json
        return True

    def get_loaded_documents(self) -> List:
        """Возвращает список имён загруженных документов."""
        return [[name] for name in self.loaded_documents]