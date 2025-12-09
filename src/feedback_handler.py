import json
import os
from datetime import datetime
from typing import List

FEEDBACK_FILE = "feedback/feedback.json"

def init_feedback_file():
    os.makedirs("feedback", exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def log_feedback(query: str, answer: str, chunks: List[str], is_correct: bool):
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