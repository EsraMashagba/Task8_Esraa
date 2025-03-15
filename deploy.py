from huggingface_hub import HfApi

hf_api = HfApi()
repo_name = "EsrMash/financial-news-model"

# Upload model
print("Uploading model to Hugging Face...")
hf_api.upload_file(
    path_or_fileobj="finbert_model.pkl",
    path_in_repo="finbert_model.pkl",
    repo_id=repo_name,
    repo_type="model"
)
print("Model uploaded successfully!")
