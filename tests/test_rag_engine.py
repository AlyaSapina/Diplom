import os
import unittest
import tempfile
from src.rag_engine import RAGEngine


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = RAGEngine()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_chunks(self):
        """Проверка добавления чанков (длинных, >30 символов)."""
        chunks = [
            "Тестовый фрагмент 1 с достаточной длиной для прохождения фильтрации.",
            "Тестовый фрагмент 2 также содержит больше тридцати символов."
        ]
        self.engine.add_chunks(chunks)
        self.assertEqual(len(self.engine.chunks), 2)
        self.assertGreater(self.engine.index.ntotal, 0)

    def test_save_and_load_index(self):
        """Проверка сохранения и загрузки индекса."""
        chunks = ["Фрагмент для сохранения с длиной более тридцати символов."]
        self.engine.add_chunks(chunks)
        self.engine.save_index(self.temp_dir)

        new_engine = RAGEngine()
        new_engine.load_index(self.temp_dir)
        self.assertEqual(len(new_engine.chunks), 1)
        self.assertEqual(new_engine.chunks[0], "Фрагмент для сохранения с длиной более тридцати символов.")

    def test_ask_returns_relevant_chunk(self):
        """Проверка поиска по запросу."""
        chunks = [
            "Инструкция по подключению питания: используйте разъём DC-IN.",
            "Настройка VLAN: введите команду vlan database."
        ]
        self.engine.add_chunks(chunks)
        answer, context = self.engine.ask("Как подключить питание?")
        self.assertIn("DC-IN", answer)

    def test_compute_pdf_hash(self):
        """Проверка вычисления хеша (без PDF!)."""
        file1 = os.path.join(self.temp_dir, "file1.txt")
        file2 = os.path.join(self.temp_dir, "file2.txt")
        content = "Одинаковое содержимое для проверки хеша."
        with open(file1, "w", encoding="utf-8") as f:
            f.write(content)
        with open(file2, "w", encoding="utf-8") as f:
            f.write(content)

        hash1 = self.engine._compute_pdf_hash(file1)
        hash2 = self.engine._compute_pdf_hash(file2)
        self.assertEqual(hash1, hash2)

    def test_document_hashes_tracking(self):
        """Проверка отслеживания хешей без вызова add_document."""
        file_path = os.path.join(self.temp_dir, "doc.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Документ 1")

        file_hash = self.engine._compute_pdf_hash(file_path)
        self.engine.document_hashes.add(file_hash)
        self.engine.loaded_documents.append("doc.txt")

        self.assertIn(file_hash, self.engine.document_hashes)
        self.assertIn("doc.txt", self.engine.loaded_documents)

    def test_get_loaded_documents(self):
        """Проверка списка документов."""
        self.engine.loaded_documents = ["manual1.pdf", "guide2.pdf"]
        docs = self.engine.get_loaded_documents()
        self.assertEqual(docs, ["manual1.pdf", "guide2.pdf"])


if __name__ == "__main__":
    unittest.main()