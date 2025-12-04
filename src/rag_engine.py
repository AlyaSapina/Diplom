"""
RAG-движок: индексация, поиск, генерация ответа.
"""

import os
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from typing import List, Tuple

class RAGEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Инициализация RAG-движка.
        :param model_name: имя модели для эмбеддингов
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = косинус при нормализации
        self.chunks = []  # метаданные: исходные тексты

        # Загрузка LLM (4-битная квантованная)
        self.llm_tokenizer = None
        self.llm_model = None
        self.llm_pipeline = None
        self._load_llm()

    def _load_llm(self):
        """
        Загружает Phi-3-mini-4k-instruct с отключённым flash attention.
        """
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype="auto",  # автоматически выберет float16/bfloat16/cpu
            trust_remote_code=True,
            attn_implementation="eager"  # ← КЛЮЧЕВОЙ ПАРАМЕТР
        )
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )

        # Настройка пайплайна генерации
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False  # для более предсказуемых ответов
        )
    def add_chunks(self, chunks: List[str]):
        """
        Добавляет чанки в индекс.
        """
        embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def save_index(self, folder_path: str = "models"):
        """
        Сохраняет FAISS-индекс и чанки.
        """
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "faiss_index.bin"))
        with open(os.path.join(folder_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, folder_path: str = "models"):
        """
        Загружает индекс и чанки из файлов.
        """
        index_path = os.path(path.join(folder_path, "faiss_index.bin"))
        chunks_path = os.path.join(folder_path, "chunks.json")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.dimension = self.index.d
        else:
            raise FileNotFoundError("Индекс или чанки не найдены!")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Находит k наиболее релевантных чанков.
        """
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_emb, k)
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Генерирует ответ на основе контекста.
        """
        context = "\n".join(context_chunks)
        prompt = f"""Используя только информацию ниже, кратко и точно ответь на вопрос.

Контекст:
{context}

Вопрос:
{query}

Ответ:"""

        response = self.llm_pipeline(prompt)
        answer = response[0]["generated_text"].split("Ответ:")[-1].strip()
        return answer

    def ask(self, query: str) -> Tuple[str, List[str]]:
        """
        Полный цикл: ретривал + генерация.
        """
        chunks = self.retrieve(query)
        answer = self.generate_answer(query, chunks)
        return answer, chunks