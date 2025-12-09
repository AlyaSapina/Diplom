# tests/test_pdf_loader.py
import unittest
from src.pdf_loader import is_chapter_heading, is_junk_line

class TestPDFLoader(unittest.TestCase):

    def test_is_chapter_heading_positive(self):
        valid_headings = [
            "Глава 1. Введение",
            "Chapter 2: Configuration",
            "1.2 Установка оборудования",
            "3.1.5 Подключение питания",
            "IV. Обслуживание",          # ← исправлено: точка есть
            "IV Обслуживание",           # ← без точки
            "  Раздел 5 — Техника безопасности  "
        ]
        for heading in valid_headings:
            with self.subTest(heading=heading):
                self.assertTrue(is_chapter_heading(heading))

    def test_is_chapter_heading_negative(self):
        invalid_headings = [
            "Это просто текст внутри документа.",
            "Подключите кабель питания.",
            "8-800-123-45-67",
            "confidential",
            "123",
            ""
        ]
        for heading in invalid_headings:
            with self.subTest(heading=heading):
                self.assertFalse(is_chapter_heading(heading))

    def test_is_junk_line_positive(self):
        junk_lines = [
            "8-800-123-45-67",
            "support@example.com",
            "http://example.com",
            "www.example.com",
            "Page 5",
            "© 2024 Company",
            "Документ ID: XYZ-123",      # ← теперь должно работать
            "123",
            "",
            "a",
        ]
        for line in junk_lines:
            with self.subTest(line=line):
                self.assertTrue(is_junk_line(line))

    def test_is_junk_line_negative(self):
        good_lines = [
            "Подключите блок питания к разъёму DC-IN.",
            "Глава 1 содержит общие сведения.",
            "Для настройки VLAN используйте команду switch#vlan database."
        ]
        for line in good_lines:
            with self.subTest(line=line):
                self.assertFalse(is_junk_line(line))

# УДАЛИЛИ тесты extract_chapters_from_pdf и process_pdf_to_chunks
# (они требуют реального PDF или mock-объектов)

if __name__ == "__main__":
    unittest.main()