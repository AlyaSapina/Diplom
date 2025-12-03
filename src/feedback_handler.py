"""
Обработка обратной связи от пользователя.
"""

import json
import os
from datetime import datetime

FEEDBACK_FILE = "feedback/feedback.json"

def init_feedback_file():
    """Создаёт файл обратной связи, если его нет."""
    os.makedirs("feedback", exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def log_feedback(query: str, answer: str, chunks: List[str], is_correct: bool):
    """
    Сохраняет обратную связь в JSON.
    :param query: вопрос
    :param answer: ответ ИИ
    :param chunks: использованные чанки
    :param is_correct: True/False
    """
    init_feedback_file()
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "is_correct": is_correct
    }
    with open(FEEDBACK_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(feedback_entry)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)