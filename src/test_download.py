from huggingface_hub import snapshot_download
import os

# Создаём папку для модели
os.makedirs("models/phi3_mini", exist_ok=True)

print("Начинаю загрузку модели Phi-3-mini-4k-instruct...")
snapshot_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    local_dir="models/phi3_mini",
    local_dir_use_symlinks=False,
    resume_download=True  # позволяет докачать при обрыве
)
print("✅ Загрузка завершена! Модель сохранена в папку: models/phi3_mini")