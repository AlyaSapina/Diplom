import pdfplumber
import re
from typing import List

def is_chapter_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    patterns = [
        r'^\s*(глава|chapter|раздел|section)\s+\d+[^a-zA-Zа-яА-Я]*',
        r'^\s*\d+(\.\d+)*\s+[A-ZА-Я]',                          # 1.1 Введение
        r'^\s*[IVXLCDM]+[.\s_]+[A-ZА-Я]',                      # IV. Обслуживание, IV_Обслуживание
    ]
    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False

def is_junk_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 8:
        return True
    if re.search(r'(http|www\.|@|8\-800|\d{3}\-\d{3}\-\d{4}|\d{10,})', line, re.IGNORECASE):
        return True
    line_lower = line.lower()
    junk_phrases = [
        'confidential', 'поддержка', 'телефон', 'support', 'page',
        '©', 'дата выпуска', 'technical documentation'
    ]
    # Проверяем вхождение подстрок (а не точное совпадение)
    if any(phrase in line_lower for phrase in junk_phrases):
        return True
    if 'документ' in line_lower and 'id' in line_lower:
        return True
    if re.fullmatch(r'\d+', line):
        return True
    return False

def extract_chapters_from_pdf(pdf_path: str) -> List[str]:
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
                    if current_chapter_lines:
                        chapters.append("\n".join(current_chapter_lines))
                        current_chapter_lines = []
                    current_chapter_lines.append(line)
                    in_chapter = True
                else:
                    if in_chapter and not is_junk_line(line):
                        current_chapter_lines.append(line)

        if current_chapter_lines:
            chapters.append("\n".join(current_chapter_lines))

    return chapters

def split_chapter_into_chunks(chapter_text: str, chunk_size: int = 250) -> List[str]:
    clean_text = re.sub(r'\s+', ' ', chapter_text.strip())
    if not clean_text:
        return []
    words = clean_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks

def process_pdf_to_chunks(pdf_path: str) -> List[str]:
    chapters = extract_chapters_from_pdf(pdf_path)
    all_chunks = []
    for chapter in chapters:
        chunks = split_chapter_into_chunks(chapter)
        all_chunks.extend(chunks)
    return all_chunks