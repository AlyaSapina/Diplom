"""
Точка входа. Запускает Gradio-интерфейс.
"""

import os
import gradio as gr
from src.rag_engine import RAGEngine
from src.pdf_loader import process_pdf_to_chunks
from src.feedback_handler import log_feedback

engine = RAGEngine()

def upload_pdfs(files):
    all_chunks = []
    for file in files:
        chunks = process_pdf_to_chunks(file.name)
        all_chunks.extend(chunks)
    engine.add_chunks(all_chunks)
    engine.save_index()
    return f"✅ Загружено {len(all_chunks)} фрагментов из {len(files)} файлов."

def ask_question(query):
    if engine.index.ntotal == 0:
        return "Сначала загрузите инструкции.", ""
    answer, context = engine.ask(query)
    return answer, context

def handle_feedback(query, answer, context, is_correct):
    if not is_correct:
        engine.mark_as_bad(context)  # ← ВОТ КЛЮЧ!
    log_feedback(query, answer, [context], is_correct)
    return "Учтено!" if is_correct else "Понял, больше не буду показывать этот фрагмент."

# Загружаем индекс при старте, если есть
try:
    engine.load_index()
    print("✅ Индекс загружен из models/")
except (FileNotFoundError, RuntimeError) as e:
    print("ℹ️ Индекс не найден. Загрузите PDF-инструкции для создания.")

with gr.Blocks(title="AI-помощник для инженера") as demo:
    gr.Markdown("# 🤖 AI-помощник для инженера 1-й линии")
    gr.Markdown("Загрузите PDF-инструкции (на любом языке). Ответ всегда будет на русском.")

    with gr.Tab("📄 Загрузка инструкций"):
        pdf_input = gr.File(file_count="multiple", file_types=[".pdf"])
        upload_btn = gr.Button("🔄 Загрузить и проиндексировать")
        upload_status = gr.Textbox(label="Статус")

    with gr.Tab("💬 Задать вопрос"):
        query_input = gr.Textbox(label="Ваш вопрос", placeholder="Как настроить VLAN на коммутаторе?")
        ask_btn = gr.Button("🔍 Получить ответ")

        answer_output = gr.Textbox(
            label="💬 Ответ ИИ (на русском)",
            lines=12,
            interactive=False,
        )
        context_output = gr.Textbox(
            label="📄 Использованный контекст (оригинал)",
            lines=12,
            interactive=False,
        )

        with gr.Row():
            yes_btn = gr.Button("✅ Верно")
            no_btn = gr.Button("❌ Неверно")
        feedback_status = gr.Textbox(label="Обратная связь")

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