"""
Модуль для загрузки и обработки PDF-инструкций.
Извлекает текст, разбивает на чанки.
"""

import pdfplumber
import re
from typing import List


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из PDF, пропуская шапки/подвалы и повторяющиеся строки.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                continue

            # Разделяем на строки
            lines = page_text.splitlines()

            # Удаляем пустые строки и строки с только цифрами/символами
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # Пропускаем строки, которые выглядят как номер страницы или URL
                if re.match(r'^\d+$', stripped):  # только цифры
                    continue
                if 'http' in stripped or 'www.' in stripped:  # URL
                    continue
                if len(stripped) < 10:  # слишком короткие строки
                    continue
                clean_lines.append(stripped)

            text += "\n".join(clean_lines) + "\n"
    return text

def clean_text(text: str) -> str:
    """
    Очищает текст от лишних пробелов и переносов.
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def split_into_chunks(text: str, chunk_size: int = 256) -> List[str]:
    """
    Разбивает текст на чанки по количеству токенов (упрощённо — по словам).
    :param text: очищенный текст
    :param chunk_size: размер чанка в словах
    :return: список чанков
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdf_to_chunks(pdf_path: str) -> List[str]:
    """
    Полный пайплайн: PDF → текст → чанки.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw_text)
    chunks = split_into_chunks(clean)
    return chunks