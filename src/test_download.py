from huggingface_hub import snapshot_download

print("Начинаю загрузку модели...")
snapshot_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    local_dir="models/phi3_mini",
    local_dir_use_symlinks=False
)
print("Загрузка завершена!")