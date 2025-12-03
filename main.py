"""
Точка входа. Запускает Gradio-интерфейс.
"""

import os
import gradio as gr
from src.rag_engine import RAGEngine
from src.pdf_loader import process_pdf_to_chunks
from src.feedback_handler import log_feedback

# Глобальный экземпляр движка
engine = RAGEngine()

def upload_pdfs(files):
    """
    Обработка загрузки PDF-файлов.
    """
    all_chunks = []
    for file in files:
        chunks = process_pdf_to_chunks(file.name)
        all_chunks.extend(chunks)
    engine.add_chunks(all_chunks)
    engine.save_index()
    return f"Загружено {len(all_chunks)} чанков из {len(files)} файлов."

def ask_question(query):
    """
    Отправка вопроса и получение ответа.
    """
    if engine.index.ntotal == 0:
        return "Сначала загрузите инструкции!", []
    answer, chunks = engine.ask(query)
    return answer, "\n---\n".join(chunks)

def handle_feedback(query, answer, chunks_str, is_correct):
    """
    Обработка кнопок 'Верно/Неверно'.
    """
    chunks = chunks_str.split("\n---\n")
    log_feedback(query, answer, chunks, is_correct)
    return "Спасибо за обратную связь!" if is_correct else "Понял, учту!"

# Загружаем индекс при старте, если есть
try:
    engine.load_index()
except FileNotFoundError:
    pass

# Gradio-интерфейс
with gr.Blocks(title="AI-помощник для инженера") as demo:
    gr.Markdown("# AI-помощник для инженера 1-й линии")
    gr.Markdown("Загрузите PDF-инструкции, затем задавайте вопросы.")

    with gr.Tab("Загрузка инструкций"):
        pdf_input = gr.File(file_count="multiple", file_types=[".pdf"])
        upload_btn = gr.Button("Загрузить и проиндексировать")
        upload_status = gr.Textbox(label="Статус")

    with gr.Tab("Задать вопрос"):
        query_input = gr.Textbox(label="Ваш вопрос", placeholder="Как перезагрузить сервер?")
        ask_btn = gr.Button("Получить ответ")
        answer_output = gr.Textbox(label="Ответ ИИ")
        context_output = gr.Textbox(label="Использованный контекст", lines=5)

        with gr.Row():
            yes_btn = gr.Button("✅ Верно")
            no_btn = gr.Button("❌ Неверно")
        feedback_status = gr.Textbox(label="Обратная связь")

    # Связывание событий
    upload_btn.click(upload_pdfs, inputs=pdf_input, outputs=upload_status)
    ask_btn.click(ask_question, inputs=query_input, outputs=[answer_output, context_output])
    yes_btn.click(
        handle_feedback,
        inputs=[query_input, answer_output, context_output, gr.State(True)],
        outputs=feedback_status
    )
    no_btn.click(
        handle_feedback,
        inputs=[query_input, answer_output, context_output, gr.State(False)],
        outputs=feedback_status
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)