import os
import unittest
import json
import shutil  # ← добавлен импорт
from src.feedback_handler import log_feedback, init_feedback_file

class TestFeedbackHandler(unittest.TestCase):

    def setUp(self):
        """Очищаем feedback-файл и папку перед каждым тестом."""
        self.feedback_file = "feedback/feedback.json"
        if os.path.exists(self.feedback_file):
            os.remove(self.feedback_file)
        if os.path.exists("feedback"):
            shutil.rmtree("feedback")  # ← ИСПРАВЛЕНО: теперь удаляется даже непустая папка

    def tearDown(self):
        """Очищаем после теста."""
        if os.path.exists(self.feedback_file):
            os.remove(self.feedback_file)
        if os.path.exists("feedback"):
            shutil.rmtree("feedback")

    def test_init_feedback_file(self):
        """Проверка создания файла обратной связи."""
        init_feedback_file()
        self.assertTrue(os.path.exists(self.feedback_file))
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, [])

    def test_log_feedback(self):
        """Проверка сохранения обратной связи."""
        init_feedback_file()
        log_feedback(
            query="Как подключить питание?",
            answer="Используйте разъём DC-IN.",
            chunks=["Используйте разъём DC-IN."],
            is_correct=True
        )
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        entry = data[0]
        self.assertEqual(entry["query"], "Как подключить питание?")
        self.assertEqual(entry["answer"], "Используйте разъём DC-IN.")
        self.assertEqual(entry["is_correct"], True)
        self.assertIn("timestamp", entry)

    def test_multiple_feedback_entries(self):
        """Проверка нескольких записей."""
        init_feedback_file()
        log_feedback("Вопрос 1", "Ответ 1", ["Чанк 1"], True)
        log_feedback("Вопрос 2", "Ответ 2", ["Чанк 2"], False)
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertTrue(data[0]["is_correct"])
        self.assertFalse(data[1]["is_correct"])


if __name__ == "__main__":
    unittest.main()