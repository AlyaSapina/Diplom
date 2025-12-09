# src/pdf_loader.py
import pdfplumber
import re
from typing import List


def is_chapter_heading(line: str) -> bool:
    """
    Определяет, является ли строка заголовком главы.
    Поддерживает русские и английские шаблоны.
    """
    line = line.strip()
    if not line:
        return False
    # Шаблоны заголовков глав
    patterns = [
        r'^\s*(глава|chapter|раздел|section)\s+\d+[^a-zA-Zа-яА-Я]*',  # Глава 1, Chapter 2
        r'^\s*\d+(\.\d+)*\s+[A-ZА-Я]',  # 1.1 Введение, 3.2.1 Configuration
        r'^\s*[IVXLCDM]+\.\s+[A-Z]',  # I. Introduction (римские)
    ]
    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def is_junk_line(line: str) -> bool:
    """Фильтр мусора: телефоны, URL, номера страниц и т.д."""
    line = line.strip()
    if not line or len(line) < 8:
        return True
    if re.search(r'(http|www\.|@|8\-800|\d{3}\-\d{3}\-\d{4}|\d{10,})', line, re.IGNORECASE):
        return True
    if any(phrase in line.lower() for phrase in [
        'confidential', 'поддержка', 'телефон', 'support', 'page',
        '©', 'дата выпуска', 'technical documentation', 'document id'
    ]):
        return True
    if re.fullmatch(r'\d+', line):  # только цифры
        return True
    return False


def extract_chapters_from_pdf(pdf_path: str) -> List[str]:
    """
    Извлекает текст из PDF и разбивает его на главы.
    Возвращает список строк, где каждая строка — это глава (заголовок + содержимое).
    """
    chapters = []
    current_chapter_lines = []
    in_chapter = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                continue
            lines = page_text.splitlines()

            for line in lines:
                if is_chapter_heading(line):
                    # Сохраняем предыдущую главу, если она есть
                    if current_chapter_lines:
                        chapter_text = "\n".join(current_chapter_lines)
                        chapters.append(chapter_text)
                        current_chapter_lines = []
                    # Начинаем новую главу с заголовка
                    current_chapter_lines.append(line)
                    in_chapter = True
                else:
                    if in_chapter and not is_junk_line(line):
                        current_chapter_lines.append(line)

        # Не забываем последнюю главу
        if current_chapter_lines:
            chapters.append("\n".join(current_chapter_lines))

    return chapters


def split_chapter_into_chunks(chapter_text: str, chunk_size: int = 250) -> List[str]:
    """
    Разбивает текст одной главы на чанки фиксированного размера (в словах).
    """
    # Очищаем от лишних пробелов
    clean_text = re.sub(r'\s+', ' ', chapter_text.strip())
    if not clean_text:
        return []

    words = clean_opt_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks


def process_pdf_to_chunks(pdf_path: str) -> List[str]:
    """
    Основная функция: PDF → главы → чанки (без смешивания глав).
    """
    chapters = extract_chapters_from_pdf(pdf_path)
    all_chunks = []
    for chapter in chapters:
        chunks = split_chapter_into_chunks(chapter)
        all_chunks.extend(chunks)
    return all_chunks