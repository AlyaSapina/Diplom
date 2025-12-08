# src/pdf_loader.py
import pdfplumber
import re

def is_content_line(line: str) -> bool:
    """Определяет, является ли строка полезным содержанием."""
    line = line.strip()
    if not line:
        return False
    if len(line) < 15:  # слишком коротко
        return False
    if re.match(r'^\d{1,3}$', line):  # номер страницы
        return False
    if 'http' in line or 'www.' in line or '.com' in line:
        return False
    if line.startswith(('©', 'Confidential', 'Page', 'Телефон', 'Support', 'www.', 'http')):
        return False
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', line):  # дата
        return False
    if line.count('.') > 5 or line.count('/') > 10:  # URL или путь
        return False
    return True

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                continue
            lines = page_text.splitlines()
            clean_lines = [line for line in lines if is_content_line(line)]
            text += "\n".join(clean_lines) + "\n"
    return text

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def split_into_chunks(text: str, chunk_size: int = 200) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:  # только осмысленные фрагменты
            chunks.append(chunk)
    return chunks

def process_pdf_to_chunks(pdf_path: str) -> list:
    raw = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw)
    return split_into_chunks(clean)